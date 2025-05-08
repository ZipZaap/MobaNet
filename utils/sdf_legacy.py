import os
import cv2 
import math
import json
import numpy as np
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn.functional as F

from configs import CONF



def compute_sobel_edges(mask):

    if not len(mask.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {mask.shape}")
    
    sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) 
                             
    sobel_y = torch.tensor([[-1, -2, -1], 
                            [ 0,  0,  0], 
                            [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3) 
    
    sobel_x = sobel_x.to(mask.device)
    sobel_y = sobel_y.to(mask.device)

    grad_x = F.conv2d(F.pad(mask, (1,1,1,1), 'reflect'), sobel_x, padding=0)
    grad_y = F.conv2d(F.pad(mask, (1,1,1,1), 'reflect'), sobel_y, padding=0)
    sobel_edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    sobel_edges = torch.sigmoid((sobel_edges-3)*100)
    return sobel_edges


def generate_kernel(k: int = 7, 
                    type: str = 'chebyshev', 
                    device: str = CONF.DEFAULT_DEVICE,
                    dtype: torch.dtype = torch.float32):
    
    if type == 'manhattan':
        center = (k // 2, k // 2)
        y_coords, x_coords = torch.meshgrid(torch.arange(k, dtype = dtype), torch.arange(k, dtype = dtype), indexing='ij')
        kernel = torch.abs(x_coords - center[0]) + torch.abs(y_coords - center[1])

    elif type == 'chebyshev':
        center = (k // 2, k // 2)
        y_coords, x_coords = torch.meshgrid(torch.arange(k, dtype = dtype), torch.arange(k, dtype = dtype), indexing='ij')
        kernel = torch.max(torch.abs(x_coords - center[0]), torch.abs(y_coords - center[1]))

    elif type == 'euclidean':
        coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
        dist2 = coords ** 2
        kernel = torch.sqrt(dist2.unsqueeze(0) + dist2.unsqueeze(1))

    kernel = kernel.to(device)

    return kernel


def SDF(
    mask: torch.Tensor, 
    kernel_size: int = CONF.SDM_KERNEL, 
    h: float = 0.35,
    normalization: str = CONF.NORMALIZATION
    ) -> torch.Tensor:
    """Approximates the Manhattan distance transform of images using cascaded convolution operations.

    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the image image.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension of the images.

    Args:
        image: Image with shape :math:`(B,C,H,W)`.
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    Returns:
        tensor with shape :math:`(B,C,H,W)`.
    """
    edges = compute_sobel_edges(mask)

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # n_iters is set such that the DT will be able to propagate from any corner of the image to its far,
    # diagonally opposite corner
    n_iters: int = math.ceil(max(edges.shape[2], edges.shape[3]) / math.floor(kernel_size / 2))
    kernel = generate_kernel(kernel_size, 'chebyshev', edges.device, edges.dtype)
    kernel = torch.exp(kernel / -h).unsqueeze(0).unsqueeze(0)
    sdm = torch.zeros_like(edges)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = edges.clone()
    for i in range(n_iters):
        cdt = F.conv2d(boundary, kernel, padding = kernel_size//2)
        cdt = -h * torch.log(cdt)
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        edges = torch.where(cdt > 0, 1.0, 0.0)
        if edges.sum() == 0:
            break

        offset: int = i * (kernel_size // 2)
        sdm += (offset + cdt) * edges
        boundary += edges

    sdm = normalize_sdm(mask, sdm)
    return sdm

def normalize_sdm(mask: torch.Tensor, sdm: torch.Tensor, mode: str = CONF.NORMALIZATION):
    B = sdm.shape[0]
    if mode == 'static_max':
        sdm = sdm/ (CONF.INPUT_IMAGE_SIZE*torch.where(mask == 0, -1 , 1))

    elif mode == 'dynamic_max':
        maxval = sdm.view(B , 1, -1).max(dim=2)[0]
        sdm = sdm/ (maxval.view(B, 1, 1, 1)*torch.where(mask == 0, -1 , 1))

    elif mode == 'minmax':
        minval = torch.multiply(sdm, mask == 1)
        minval = minval.view(B , 1, -1).max(dim=2)[0]
        minval = minval.view(B, 1, 1, 1)
        minval = torch.multiply(mask == 1, minval)

        maxval = torch.multiply(sdm, mask == 0)
        maxval= maxval.view(B , 1, -1).max(dim=2)[0]
        maxval = maxval.view(B, 1, 1, 1)
        maxval = torch.multiply(mask == 0, maxval)

        minmax = minval - maxval
        sdm = sdm/minmax
        
    return sdm

def save_sdms(imIDs, sdms, save_png = False):
    for imID, sdm in zip(imIDs, torch.split(sdms, 1, dim=0)):
        sdm = sdm[0][0].detach().cpu().numpy()
        np.save(f'{CONF.SDM_DATASET_PATH}/{imID}.npy', sdm)

        if save_png:
            sdm_8bit_img = (255*sdm/sdm.max()).astype(np.uint8)
            colormap = cv2.applyColorMap(sdm_8bit_img, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(f'{CONF.SDM_DATASET_PATH}/{imID}.png', colormap)

def generate_sdms(labels_json: str = CONF.LABELS_JSON,
                  mask_dataset_dir: str = CONF.MASK_DATASET_DIR,
                  sdm_dataset_dir: str = CONF.SDM_DATASET_DIR,
                  device: str = CONF.DEFAULT_DEVICE,
                  batch_size: int = CONF.BATCH_SIZE
                  ) -> None:
    
    if not os.path.exists(sdm_dataset_dir):
        os.makedirs(sdm_dataset_dir, exist_ok=True)


    if not os.listdir(sdm_dataset_dir):
        print('[PROC] SDMs >>> generating ...')

        with open(labels_json) as f:
            lblDict = json.load(f)
            # imIDs = lblDict['typ']['scarp']
            imIDs = lblDict['boundary']
        
        mask_batch = []
        imID_batch = []
        for i, imID in enumerate(tqdm(imIDs), start=1):
            mask = cv2.imread(os.path.join(mask_dataset_dir, f'{imID}.png'), cv2.IMREAD_GRAYSCALE)/255
            mask_batch.append(mask) 
            imID_batch.append(imID)
            if i % batch_size == 0 or i == len(imIDs):
                mask_batch = np.stack(mask_batch, axis=0)
                mask_batch_tensor = torch.tensor(mask_batch).to(torch.float32).unsqueeze(1)
                mask_batch_tensor = mask_batch_tensor.to(device)

                SDMs = SDF(mask_batch_tensor)
                save_sdms(imID_batch, SDMs)

                mask_batch = []
                imID_batch = []
    else:
        print('[PROC] SDMs >>> ready')