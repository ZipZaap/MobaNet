import os
import cv2 
import math
import json
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