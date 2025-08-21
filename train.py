import os
import yaml
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam
from torch.multiprocessing.spawn import spawn
from torch.optim.lr_scheduler import LinearLR

from utils.sdf import SDF
# from utils.util import setup_dirs
from utils.loggers import Logger
from utils.dataset import DatasetTools
from utils.managers import ProcessManager
from engines.SegTrainer import SegTrainer

from configs.cfgparser import Config
from configs.cli import parse_cli_args

def _main_worker(rank: int, cfg: Config):
    """
    Main function to run the training process.

    Args:
        rank : int
            The rank of the process.

        cfg : Config
            Configuration object containing the training parameters.
    """
    
    # Bind rank to config and device
    cfg.RANK = rank
    process = ProcessManager(cfg)
    process.bind_to_device()

    # Model, optimizer, scheduler
    model = process.load_model()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.BASE_LR,
        weight_decay=cfg.L2_DECAY,
    )

    lr_scheduler = LinearLR(
        optimizer, 
        start_factor=cfg.INIT_LR/cfg.BASE_LR, 
        total_iters=cfg.WARMUP_EPOCHS
    )

    # Data + logging
    loaders = DatasetTools.train_dataloaders(cfg)
    logger = Logger(cfg)

    # Train
    SegTrainer(model, optimizer, lr_scheduler, loaders, logger, cfg).train()

    # Cleanup CUDA/Process resources
    process.cleanup()


def train(cfg: dict | str,
          *,
          dataset_dir: Optional[str] = None,
          train_set: Optional[str] = None,
          test_set: Optional[str] = None,
          model: Optional[str] = None,
          checkpoint: Optional[str] = None,
          batch_size: Optional[int] = None,
          train_epochs: Optional[int] = None,
          loss: Optional[str] = None,
          GPUs: Optional[list[int]] = None
          ):
    """
    Launch training programmatically (importable API), or from a notebook/script. 
    The function exposes only the basic configuration options. For a more granular control, 
    consider modifying the `config.yaml` file directly or using the CLI.

    Args
    ----
        config : dict | str
            Either a loaded parameter dictionary or a path to a `config.yaml` file.

        dataset_dir : str, optional
            Path to the dataset directory.

        train_set : str, optional
            Composition of the training set.

        test_set : str, optional
            Composition of the testing set.

        model : str, optional
            Name of the model to use.

        checkpoint : str, optional
            Path to the model checkpoint.

        batch_size : int, optional
            Batch size for training.

        train_epochs : int, optional
            Number of training epochs.

        loss : str, optional
            Loss function to use.

        GPUs : list[int], optional
            List of GPU IDs to use for training

    Raises
    ------
        RuntimeError
            If no GPU with CUDA support is available.

        ValueError
            If cfg is not a valid Config object or YAML file.

    Examples
    --------
    Using a config file path directly:

    >>> from train import train
    >>> train("configs/config.yaml", dataset_dir="path/to/dataset",
    ...       batch_size=16, train_epochs=100, loss="SoftDICE", GPUs=[0, 1])

    Loading the config with PyYAML first & modifying the parameters (e.g from CLI):

    >>> import yaml
    >>> from train import train
    >>> with open("configs/config.yaml") as f:
    ...     cfg = yaml.load(f, Loader=yaml.FullLoader)
    >>> cfg = parse_cli_args(cfg, inference=False)
    >>> train(cfg)
    """

    # Basic environment check
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This library requires a GPU with CUDA support. "
            "Please verify the PyTorch installation and ensure that a compatible GPU is available."
        )
    
    # Load cfg dictionary 
    if isinstance(cfg, dict):
        _cfg: dict = cfg
    elif isinstance(cfg, str) and os.path.exists(cfg) and cfg.endswith('.yaml'):
        with Path(cfg).open() as f:
            _cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError("cfg should either be a parameter dictionary or a path to a valid YAML file.")

    # Update config with local variables
    for param, value in locals().items():
        if param != 'cfg' and value is not None:
            _cfg[param.upper()]['default'] = value

    # Instantiate config object
    CONF: Config = Config(_cfg)

    # Setup
    DatasetTools.compose_dataset(CONF)
    SDF.generate_sdms(CONF)

    # Launch
    if CONF.WORLD_SIZE > 1:
        print(f"[INFO] Running in distributed mode on {CONF.WORLD_SIZE} GPUs ...")
        os.environ['NCCL_P2P_DISABLE'] = '0' if CONF.NCCL_P2P else '1'
        return spawn(_main_worker, args=(CONF,), nprocs=CONF.WORLD_SIZE)
    else:
        print(f"[INFO] Running in non-distributed mode on {CONF.DEFAULT_DEVICE} ...")
        return _main_worker(0, CONF)

if __name__ == "__main__":
    # Load the YAML into a dict
    with Path("configs/config.yaml").open() as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)

    # Default CLI behavior. Allows: python train.py --FOO=bar
    cfg = parse_cli_args(cfg, inference=False)

    # run training
    train(cfg)