import os
import cv2 
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F

from configs.config_parser import CONF

class SDF:
    @classmethod
    def compute_sobel_edges(cls, 
                            mask: torch.Tensor
                            ) -> torch.Tensor:
        # if mask.ndim != 4:
        #     raise ValueError(f"Invalid image shape, expected BxCxHxW. Got: {mask.shape}")

        sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], 
                                [ 0,  0,  0], 
                                [ 1,  2,  1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(F.pad(mask, (1, 1, 1, 1), mode='reflect'), sobel_x, padding=0)
        grad_y = F.conv2d(F.pad(mask, (1, 1, 1, 1), mode='reflect'), sobel_y, padding=0)
        sobel_edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        sobel_edges = torch.sigmoid((sobel_edges - 3) * 100)
        return sobel_edges

    @classmethod
    def _generate_kernel(cls, 
                         device: str,
                         ksize: int = CONF.SDM_KERNEL_SIZE,
                         ktype: str = CONF.SDM_KERNEL_TYPE
                         ) -> torch.Tensor:

        if ktype == 'manhattan':
            center = ksize // 2
            y_coords, x_coords = torch.meshgrid(
                torch.arange(ksize, dtype=torch.float32, device=device), 
                torch.arange(ksize, dtype=torch.float32, device=device), 
                indexing='ij'
            )
            kernel = torch.abs(x_coords - center) + torch.abs(y_coords - center)

        elif ktype == 'chebyshev':
            center = ksize // 2
            y_coords, x_coords = torch.meshgrid(
                torch.arange(ksize, dtype=torch.float32, device=device), 
                torch.arange(ksize, dtype=torch.float32, device=device), 
                indexing='ij'
            )
            kernel = torch.max(torch.abs(x_coords - center), torch.abs(y_coords - center))

        elif ktype == 'euclidean':
            coords = torch.arange(ksize, dtype=torch.float32, device=device) - (ksize - 1) / 2
            dist2 = coords ** 2
            kernel = torch.sqrt(dist2.unsqueeze(0) + dist2.unsqueeze(1))

        return kernel.unsqueeze(0).unsqueeze(0)
    
<<<<<<< Updated upstream
    sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                             
    sobel_y = torch.tensor([[-1, -2, -1], 
                            [ 0,  0,  0], 
                            [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    sobel_x = sobel_x.to(mask.device)
    sobel_y = sobel_y.to(mask.device)

    grad_x = F.conv2d(F.pad(mask, (1,1,1,1), 'reflect'), sobel_x, padding=0)
    grad_y = F.conv2d(F.pad(mask, (1,1,1,1), 'reflect'), sobel_y, padding=0)
    sobel_edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    sobel_edges = torch.multiply(sobel_edges, mask)
    sobel_edges = torch.where(sobel_edges != 0, 1.0, 0.0)
    return sobel_edges

def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device(CONF.DEFAULT_DEVICE),
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:    
    """Generate a coordinate grid for an image.
    
    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.
    """

    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2

    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2

def SDF(
    mask: torch.Tensor, 
    kernel_size: int = 3, 
    h: float = 0.35
    ) -> torch.Tensor:
    r"""Approximates the Manhattan distance transform of images using cascaded convolution operations.

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
    grid = create_meshgrid(kernel_size, kernel_size, normalized_coordinates=False,
                           device=edges.device, dtype=edges.dtype)

    grid -= math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    sdm = torch.zeros_like(edges)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = edges.clone()
    signal_ones = torch.ones_like(boundary)
    for i in range(n_iters):
        cdt = F.conv2d(boundary, kernel.unsqueeze(0), padding = kernel_size//2)
        cdt = -h * torch.log(cdt)

        # We are calculating log(0) above.
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        edges = torch.where(cdt > 0, 1.0, 0.0)
        if edges.sum() == 0:
            break

        offset: int = i * kernel_size // 2
        sdm += (offset + cdt) * edges
        boundary = torch.where(edges == 1, signal_ones, boundary)

    return sdm

def normalize_sdm(mask: torch.Tensor, sdm: torch.Tensor, mode: str = 'minmax'):
    B = sdm.shape[0]
    if mode == 'max':
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

def save_sdms(imIDs, sdms, to_img = False):
    for imID, sdm in zip(imIDs, torch.split(sdms, 1, dim=0)):
        sdm = sdm[0][0].detach().cpu().numpy()
        np.save(f'{CONF.SDM_DATASET_PATH}/{imID}.npy', sdm)

        if to_img:
            sdm_8bit_img = (255*sdm/sdm.max()).astype(np.uint8)
            colormap = cv2.applyColorMap(sdm_8bit_img, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(f'{CONF.SDM_DATASET_PATH}/{imID}.png', colormap)

def generate_sdms(imIDs, ksize = CONF.SDM_KERNEL):
    imIDs = imIDs[0] + imIDs[1]
    if len(os.listdir(CONF.SDM_DATASET_PATH)) == 0:
        print('[PROC] SDMs >>> generating ...')
        mask_batch = []
        imID_batch = []
        for i, imID in enumerate(tqdm(imIDs), start=1):
            mask = cv2.imread(f'{CONF.MASK_DATASET_PATH}/{imID}.png', cv2.IMREAD_GRAYSCALE)/255
            mask_batch.append(mask) 
            imID_batch.append(imID)
            if i%CONF.BATCH_SIZE == 0 or i == len(imIDs):
                mask_batch = np.stack(mask_batch, axis=0)
                mask_batch_tensor = torch.tensor(mask_batch).to(torch.float32).unsqueeze(1)
                mask_batch_tensor = mask_batch_tensor.to(CONF.DEFAULT_DEVICE)
=======

    @classmethod
    def _normalize_sdm(cls, 
                       mask: torch.Tensor, 
                       sdm: torch.Tensor,
                       imsize: int = CONF.INPUT_IMAGE_SIZE,
                       normalization: str = CONF.SDM_NORMALIZATION
                       ) -> torch.Tensor:
        
        B = sdm.shape[0]

        if normalization == 'static_max':
            sdm = sdm / (imsize * torch.where(mask == 0, -1, 1))

        elif normalization== 'dynamic_max':
            maxval = sdm.view(B, 1, -1).max(dim=2)[0]
            sdm = sdm / (maxval.view(B, 1, 1, 1) * torch.where(mask == 0, -1, 1))

        elif normalization == 'minmax':
            minval = torch.multiply(sdm, mask == 1)
            minval = minval.view(B, 1, -1).max(dim=2)[0].view(B, 1, 1, 1)
            minval = torch.multiply(mask == 1, minval)

            maxval = torch.multiply(sdm, mask == 0)
            maxval = maxval.view(B, 1, -1).max(dim=2)[0].view(B, 1, 1, 1)
            maxval = torch.multiply(mask == 0, maxval)

            minmax = minval - maxval
            sdm = sdm / minmax

        return sdm

    @classmethod
    def _save_sdms(cls, 
                   imIDs: list[str], 
                   sdms: torch.Tensor, 
                   save_png: bool = False,
                   savedir: Path = CONF.SDM_DIR
                   ) -> None:
        
        for imID, sdm in zip(imIDs, torch.split(sdms, 1, dim=0)):
            sdm_np = sdm[0][0].detach().cpu().numpy()
            np.save(str(savedir / f'{imID}.npy'), sdm_np)
>>>>>>> Stashed changes

            if save_png:
                sdm_8bit_img = (255 * sdm_np / sdm_np.max()).astype(np.uint8)
                colormap = cv2.applyColorMap(sdm_8bit_img, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(str(savedir / f'{imID}.png'), colormap)

    @classmethod
    def sdf(cls, 
            mask: torch.Tensor, 
            h = 0.35,
            ksize: int = CONF.SDM_KERNEL_SIZE
            ) -> torch.Tensor:

        edges = cls.compute_sobel_edges(mask)

        n_iters = math.ceil(max(edges.shape[2], edges.shape[3]) / math.floor(ksize / 2))
        kernel = cls._generate_kernel(mask.device)
        kernel = torch.exp(kernel / -h)
        sdm = torch.zeros_like(edges)

        boundary = edges.clone()
        for i in range(n_iters):
            cdt = F.conv2d(boundary, kernel, padding=ksize // 2)
            cdt = -h * torch.log(cdt)
            cdt = torch.nan_to_num(cdt, posinf=0.0)

            edges = torch.where(cdt > 0, 1.0, 0.0)
            if edges.sum() == 0:
                break

            offset = i * (ksize // 2)
            sdm += (offset + cdt) * edges
            boundary += edges

        sdm = cls._normalize_sdm(mask, sdm)
        return sdm

    @classmethod
    def generate_sdms(cls,
                      batch_size: int = CONF.BATCH_SIZE,
                      device: str = CONF.DEFAULT_DEVICE,
                      lbl_json: Path = CONF.LBL_JSON,
                      sdm_dir: Path = CONF.SDM_DIR,
                      msk_dir: Path = CONF.MSK_DIR
                      ) -> None:
        
        if not sdm_dir.exists():
            sdm_dir.mkdir(parents=True, exist_ok=True)

        if not any(sdm_dir.iterdir()):
            print('[PROC] SDMs >>> generating ...')

            with lbl_json.open() as f:
                lblDict = json.load(f)
                imIDs = lblDict['boundary']

            mask_batch = []
            imID_batch = []
            for i, imID in enumerate(tqdm(imIDs), start=1):
                mask_path = str(msk_dir / f'{imID}.png')
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
                mask_batch.append(mask)
                imID_batch.append(imID)

                if i % batch_size == 0 or i == len(imIDs):
                    mask_batch_np = np.stack(mask_batch, axis=0)
                    mask_batch_tensor = torch.tensor(mask_batch_np, dtype=torch.float32, device = device).unsqueeze(1)

                    sdms = cls.sdf(mask_batch_tensor)
                    cls._save_sdms(imID_batch, sdms)
                    mask_batch = []
                    imID_batch = []
        else:
            print('[PROC] SDMs >>> ready')