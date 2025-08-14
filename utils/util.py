import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributed import all_gather

from configs.cfgparser  import Config


def load_png(impath: str | Path) -> np.ndarray:
    """
    Load a PNG image from the specified path and normalize it to [0, 1].
    If the image is 2D, it is reshaped to 3D with a single channel.
    If the image is 3D, it is loaded in BGR format, which is maintained
    throughout the rest of the processing/training.

    Args
    ----
        impath : str | Path
            Path to the .png image.

    Returns
    -------
        arr : np.ndarray (H, W, C)
            Normalized image array.

    Raises
    ------
        FileNotFoundError
            If the image file does not exist.

        ValueError
            If the image shape is not 2D or 3D.
    """

    arr = cv2.imread(str(impath), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Input path {impath} does not exist or is not a file.")

    if arr.ndim == 2:
        arr = arr[..., None]  # shape: (H, W) → (H, W, 1)
    elif arr.ndim > 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}. Expected a 2D or 3D image.")

    arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max  # Normalize to [0, 1]
    return arr


def load_mask(maskpath: str | Path, seg_classes: int = 1) -> np.ndarray:
    """
    Load a mask from the specified path. If the mask has multiple classes, it is converted to a one-hot encoded format.
    Alternatively, leave the default `seg_classes=1` to skip one-hot encoding and return the mask as is.

    Args
    ----
        maskpath : str | Path
            Path to the mask .png.

        seg_classes : int
            Number of channels in the segmentation mask

    Returns
    -------
        mask : np.ndarray
            - If `seg_classes > 1`:  One-hot encoded mask with shape (H, W, seg_classes). 
            - If `seg_classes == 1`: Mask with shape (H, W, 1).

    Raises
    ------
        FileNotFoundError
            If the mask file does not exist.

        ValueError
            If the mask shape is not 2D.
    """
    
    mask = cv2.imread(str(maskpath), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        FileNotFoundError(f"Input path {maskpath} does not exist or is not a file.")

    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask.shape}. Expected a 2D mask.")

    if seg_classes > 1:
        return np.eye((seg_classes), dtype=np.uint8)[mask] # 1Hot; shape: (H, W) → (H, W, C)
    else:
        return mask[..., None] # shape: (H, W) → (H, W, 1)


def load_sdm(sdmpath: Path, mask_shape: tuple[int, ...]) -> np.ndarray:
    """
    Loads a Signed Distance Map from the specified file path.
    If the given path does not exist, it is assumed that the mask with the 
    corresponding ID has no class-to-class boundaries. In such cases, a zero-filled SDM 
    with the provided `mask_shape` is returned.

    Args
    ----
        sdmpath : Path
            Path to the SDM .npy file.

        mask_shape : tuple[int, ...] (H, W, C)
            Shape of the mask to create an empty SDM if the file does not exist.
            
    Returns
    -------
        sdm: np.ndarray (H, W, C)
            - Loaded SDM if file exists.
            - Zeros SDM if file does not exist.

    Raises
    ------
        ValueError
            If `mask_shape` does not conform to (H, W, C).
    """

    if sdmpath.exists():
        return np.load(str(sdmpath))
    else:
        if len(mask_shape) != 3:
            raise ValueError(f"Expected mask_shape to have exactly 3 dimensions (H, W, C); got {mask_shape} instead.")

        return np.zeros((*mask_shape[:-1], 1))


def logits_to_msk(logits, mode: str) -> torch.Tensor:
    """
    Converts models segmentation logits to a segmentation mask based on the specified mode.

    Args
    ----
        logits : torch.Tensor (B, C, H, W)
            Model output logits from segmentation branch.

        mode : str
            Mode for converting logits to mask:
            - `1hot`:    One-hot encoding (X ∈ {0, 1}); non-differentiable.
            - `softmax`: Probability distribution (X ∈ [0; 1]); differentiable.
            - `argmax`:  Pixel-to-class (X ∈ {0, 1, ..., seg_classes}); non-differentiable.

    Returns
    -------
        pd_mask : torch.Tensor
            Segmentation mask after applying the specified mode.
            - If mode is `1hot`, shape: (B, C, H, W)
            - If mode is `softmax`, shape: (B, C, H, W)
            - If mode is `argmax`, shape: (B, 1, H, W)
    """

    if mode == '1hot':
        # 𝑿 ∈ {0, 1}; shape: (B, C, H, W)
        topk = logits.argmax(dim=1, keepdim=True)     
        pd_mask = torch.zeros_like(logits).scatter_(1, topk, 1.0)
    elif mode == 'prob':
        # 𝑿 ∈ [0; 1]; shape: (B, C, H, W)
        pd_mask = F.softmax(logits, dim=1)
    else: # argmax
        # 𝑿 ∈ {0, 1, ..., seg_classes}; shape: (B, 1, H, W)
        pd_mask = logits.argmax(dim=1, keepdim=True)

    return pd_mask


def logits_to_lbl(logits,
                  cls_threshold: float | None,
                  ) -> torch.Tensor:
    """
    Converts model classification logits to class labels based on the specified threshold.

    Args
    ----
        logits : torch.Tensor (B, C)
            Model output logits from classification branch.

        cls_threshold : float | None
            Threshold for classifying logits into labels.
            If the maximum probability is below this threshold, the label is set to `C - 1`, i.e. boundary class.
            If `None`, the argmax of the logits is used to determine the class labels.

    Returns
    -------
        pd_cls : torch.Tensor (B,)
            Predicted class labels based on the logits and threshold.
            - If `cls_threshold` is not None, uses the threshold to determine class labels.
            - Otherwise, uses argmax to determine class labels.
    """

    B, C = logits.shape

    # Convert logits to probabilities
    cls_probs = F.softmax(logits, dim=1)

    # Apply thresholding to determine class labels
    if cls_threshold:
        max_probs, max_lbls = cls_probs.max(dim=1)   
        pd_cls = torch.where(max_probs > cls_threshold,
                             max_lbls,
                             torch.full_like(max_lbls, C - 1))
        
    # Simply use the highest probability class
    else:
        pd_cls = cls_probs.argmax(dim=1)

    return pd_cls


def gather_tensors(tensor_dict: dict[str, torch.Tensor],
                   worldsize: int
                   ) -> dict[str, torch.Tensor]:
    """
    Gathers tensor values from all GPUs and averages them.

    Args
    ----
        tensor_dict : dict[str, torch.Tensor]
            Dictionary of tensor values (e.g., losses or metrics) per GPU.

        worldsize : int
            Number of processes (GPUs) in the distributed training.

    Returns
    -------
        tensor_dict : dict[str, torch.Tensor]
            Dictionary of averaged tensor values.
    """
    for name, tensor in tensor_dict.items():
        avgLst = [torch.zeros_like(tensor) for _ in range(worldsize)]
        all_gather(avgLst, tensor)
        tensor_dict[name] = torch.stack(avgLst).nanmean()
    return tensor_dict


def setup_dirs(cfg: Config):
    """
    Create necessary directories for the experiment based on the configuration.

    Args
    ----
        cfg : Config
            Configuration object containing paths and settings.
    """
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.LOG_LOCAL or cfg.LOG_WANDB or cfg.SAVE_MODEL:
        cfg.EXP_DIR.mkdir(parents=True, exist_ok=True)


def remap_to_sorted_indices(arr: np.ndarray) -> np.ndarray:
    """
    Replace each value in `a` by its index in the sorted list of uniques.
    Example: [0, 7, 100, 245] -> [0, 1, 2, 3]
    """
    _, inv = np.unique(arr, return_inverse=True)
    return inv.reshape(arr.shape)


def save_predictions(maskpath: Path, 
                     masks: torch.Tensor, 
                     ids: list[str]):
    """
    Save model predictions to the specified directory.

    Args
    ----
        maskpath : Path
            Path to the directory where masks will be saved.

        masks : torch.Tensor (B, C, H, W)
            Model output logits tensor.

        ids : list[str]
            List of image IDs corresponding to the outputs.
    """

    if not maskpath.exists():
        maskpath.mkdir(parents=True)

    for mask, id in zip(masks, ids):
        save_path = str(maskpath / f"{id}.png")
        mask = mask.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)  # (C, H, W) → (H, W, C)
        cv2.imwrite(save_path, mask)