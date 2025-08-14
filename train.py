import os

import torch
from torch.optim import Adam
from torch.multiprocessing.spawn import spawn
from torch.optim.lr_scheduler import LinearLR

from utils.sdf import SDF
from utils.util import setup_dirs
from utils.loggers import Logger
from utils.dataset import DatasetTools
from utils.managers import ProcessManager
from engines.SegTrainer import SegTrainer

from configs.cfgparser import Config

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


def train(cfg: Config | str):
    """
    Launch training programmatically (importable API), or from a notebook/script.

    Args
    ----
        cfg : Config | str
            Either a `Config` instance or a path to a YAML config file.

    Raises
    ------
        RuntimeError
            If no GPU with CUDA support is available.

        ValueError
            If cfg is not a valid Config object or YAML file.

    Examples
    --------
        >>> from configs.cfgparser import Config
        >>> from train import train
 
        >>> train('configs/config.yaml')
    """

    # Basic environment check
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This library requires a GPU with CUDA support. "
            "Please verify the PyTorch installation and ensure that a compatible GPU is available."
        )

    # Instantiate config
    if isinstance(cfg, Config):
        _cfg = cfg
    elif isinstance(cfg, str) and os.path.exists(cfg) and cfg.endswith('.yaml'):
        _cfg = Config(cfg)
    else:
        raise ValueError("cfg should either be a Config object or a path to a valid YAML file.")

    setup_dirs(_cfg)
    DatasetTools.compose_dataset(_cfg)
    SDF.generate_sdms(_cfg)

    # Launch
    if _cfg.WORLD_SIZE > 1:
        print(f"[INFO] Running in distributed mode on {_cfg.WORLD_SIZE} GPUs ...")
        os.environ['NCCL_P2P_DISABLE'] = '0' if _cfg.NCCL_P2P else '1'
        return spawn(_main_worker, args=(_cfg,), nprocs=_cfg.WORLD_SIZE)
    else:
        print(f"[INFO] Running in non-distributed mode on {_cfg.DEFAULT_DEVICE} ...")
        return _main_worker(0, _cfg)
    
if __name__ == "__main__":
    # Default CLI behavior. Allows: python train.py --FOO=bar (via Config(..., cli=True))
    cfg = Config("configs/config.yaml", cli=True)
    train(cfg)