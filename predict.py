import os
import torch
import numpy as np

from pathlib import Path
from utils.dataset import DatasetTools
from utils.util import load_png, save_predictions
from configs.cfgparser  import Config
from model.MobaNet import MobaNet

class Predictor:
    def __init__(self,
                 cfg: Config):
        """
        Initialize the predictor with the configuration.
        
        Args
        ----
            cfg : Config
                Configuration object containing the following attributes:
                - `.CHECKPOINT` (Path | None): Path to the model checkpoint.
                - `.DEFAULT_DEVICE` (str): Device to use for inference. Automatically set to the first GPU in `cfg.GPUs` list.
        
        Raises
        ------
            ValueError
                If the checkpoint path is not specified or does not exist.
        """

        self.device: str = cfg.DEFAULT_DEVICE
        self.checkpoint: Path | None = cfg.CHECKPOINT

        if self.checkpoint:
            # Load model weights from checkpoint
            weights = torch.load(self.checkpoint, 
                                map_location=self.device,
                                mmap=True,
                                weights_only=True)['weights']
            
            # Initialize model with weights, set to eval mode 
            self.model = MobaNet(cfg)
            self.model.load_state_dict(weights)
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError("Checkpoint path is not specified or does not exist.")

    def predict(self, input: str | Path | np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Predicts the output for the given input image.
        
        Args
        ----
            input : str | Path | np.ndarray | torch.Tensor
                Input image path, numpy array, or tensor.
                - np.ndarray: should be of shape (H, W) or (H, W, C) or (B, H, W, C).
                - torch.Tensor: should be of shape (H, W) or (H, W, C) or (B, H, W, C).

        Returns
        -------
            output : torch.Tensor (B, C, H, W)
                The model's output logits tensor.

        Raises
        ------
            ValueError
                If the input array or tensor has an unsupported shape.

            TypeError
                If the input type is unsupported.

        Example
        -------
        >>> predictor = Predictor(cfg)
        >>> output = predictor.predict('image.png')
        """
        
        if isinstance(input, (str, Path)):
            # Load image from file; (H, W, C)
            input = load_png(input)

            # add batch dimension; (H, W, C) → (1, H, W, C)
            input = input[None, ...] 
            
            # make pytorch compatible; (1, H, W, C) → (1, C, H, W)
            tensor = torch.from_numpy(input).permute(0, 3, 1, 2).float() 

        elif isinstance(input, (np.ndarray, torch.Tensor)):
            # Expand dimensions batch and channel dims (if necessary) 
            if input.ndim == 2:
                input = input[None, ..., None]  # (H, W) → (1, H, W, 1)
            elif input.ndim == 3:
                input = input[None, ...]  # (H, W, C) → (1, H, W, C)
            elif input.ndim > 4:
                raise ValueError(f"Unsupported input shape: {input.shape}. Expected an 2D, 3D or 4D (batched) input.")
                      
            # Convert input to tensor if it's a numpy array and permute dimensions for PyTorch
            tensor = torch.from_numpy(input).float() if isinstance(input, np.ndarray) else input
            tensor = tensor.permute(0, 3, 1, 2)

        else:
            raise TypeError(f"Unsupported input type: {type(input)}. Expected str, Path, np.ndarray, or torch.Tensor.")
        
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            return self.model(tensor)

def main(cfg: Config):
    """
    Main function to run the predictor.

    Args
    ----
        cfg : Config
            Configuration object containing the following attributes:
            - `.CHECKPOINT` (Path | None): Path to the model checkpoint.
            - `.DEFAULT_DEVICE` (str): Device to use for inference. Automatically set to the first GPU in `cfg.GPUs` list.
            - `.MSK_PATH` (Path): Path to save the predicted masks.
    """
    
    device: str = cfg.DEFAULT_DEVICE
    maskpath: Path = cfg.MSK_PATH

    model = Predictor(cfg)
    loader = DatasetTools.predict_dataloader(cfg)

    for batch in loader:
        ids, images = batch['id'], batch['image'].to(device)
        masks = model.predict(images)
        save_predictions(maskpath, masks, ids)


if __name__ == "__main__":
    cfg = Config('configs/config.yaml', inference = True, cli=True)

    if torch.cuda.is_available():
        main(cfg)
    else:
        raise RuntimeError('This library requires a GPU with CUDA support. '
                           'Please verify the PyTorch installation and ensure that a compatible GPU is available.') 