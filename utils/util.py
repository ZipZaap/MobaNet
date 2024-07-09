import os
import time
import math
import numpy as np
from typing import TYPE_CHECKING, List, Optional, Tuple
from model.UNet import UNet

import torch
from torch import Tensor, stack
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from configs import CONF

def ddp_setup(gpu_id):
    """
    Set up the distributed environment.
    
    Args:
        rank: The rank of the current process. Unique identifier for each process in the distributed training.
        world_size: Total number of processes participating in the distributed training.
    """
    
    # Address of the main node. Since we are doing single-node training, it's set to localhost.
    os.environ["MASTER_ADDR"] = CONF.MASTER_ADDR
    
    # Port on which the master node is expected to listen for communications from workers.
    os.environ["MASTER_PORT"] = CONF.MASTER_PORT
    
    # Set the current CUDA device to the specified device (identified by rank).
    # This ensures that each process uses a different GPU in a multi-GPU setup.
    torch.cuda.set_device(gpu_id)
    
    # Initialize the process group. 
    # 'backend' specifies the communication backend to be used, "nccl" is optimized for GPU training.
    init_process_group(backend="nccl", rank=gpu_id, world_size=CONF.GPU_COUNT)

def ddp_cleanup():
    destroy_process_group()

def initModel(gpu_id):
    model = UNet()
    if CONF.GPU_COUNT > 1:
        model = model.to('cuda')
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    else:
        model = model.to(gpu_id)

    return model

def timer(epoch, ts = None):
    if ts is not None:
        te = time.time()
        dt = np.round(te - ts)
        etc = np.round((CONF.NUM_EPOCHS - epoch - 1)*dt/60)
        return dt, etc
    else:
        return time.time()

def compute_sobel_edges(mask):

    if not len(mask.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {mask.shape}")
    
    # Define Sobel kernels for x and y direction
    sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                             
    sobel_y = torch.tensor([[-1, -2, -1], 
                            [ 0,  0,  0], 
                            [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Move the kernels to the same device as the input tensor
    device = mask.device
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)

    # Apply Sobel filters to get gradients in x and y directions
    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)

    # Compute gradient magnitude
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = torch.relu(torch.sign(grad_magnitude))
    
    return grad_magnitude

def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    ) -> Tensor:    
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

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: Tensor = stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2

def SDF(
    mask: torch.Tensor, 
    kernel_size: int = 3, 
    h: float = 0.35,
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
    surface = compute_sobel_edges(mask)


    # if not len(mask.shape) == 4:
    #     raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {mask.shape}")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # n_iters is set such that the DT will be able to propagate from any corner of the image to its far,
    # diagonally opposite corner
    n_iters: int = math.ceil(max(surface.shape[2], surface.shape[3]) / math.floor(kernel_size / 2))
    grid = create_meshgrid(kernel_size, kernel_size, normalized_coordinates=False,
                           device=surface.device, dtype=surface.dtype)

    grid -= math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    out = torch.zeros_like(surface)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = surface.clone()
    signal_ones = torch.ones_like(boundary)
    for i in range(n_iters):
        cdt = F.conv2d(boundary, kernel.unsqueeze(0), padding = kernel_size//2)
        cdt = -h * torch.log(cdt)

        # We are calculating log(0) above.
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        surface = torch.where(cdt > 0, 1.0, 0.0)
        if surface.sum() == 0:
            break

        offset: int = i * kernel_size // 2
        out += (offset + cdt) * surface
        boundary = torch.where(surface == 1, signal_ones, boundary)

    minval = -out[mask == 1].max()
    maxval = out[mask == 0].max()
    out = out / torch.where(mask == 1, minval, maxval)

    return out