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

from configs.cfgparser  import Config


def main(rank: int, cfg: Config):
    """
    Main function to run the training process.

    Args:
        rank : int
            The rank of the process.

        cfg : Config
            Configuration object containing the training parameters.
    """
    
    cfg.RANK = rank
    process = ProcessManager(cfg)
    process.bind_to_device()

    model = process.load_model()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                     lr=cfg.BASE_LR, 
                     weight_decay=cfg.L2_DECAY)
    lr_scheduler = LinearLR(optimizer, 
                            start_factor=cfg.INIT_LR/cfg.BASE_LR, 
                            total_iters=cfg.WARMUP_EPOCHS)
    loaders = DatasetTools.train_dataloaders(cfg)
    logger = Logger(cfg)
    SegTrainer(model, optimizer, lr_scheduler, loaders, logger, cfg).train()

    process.cleanup()

if __name__ == "__main__":
        
    cfg = Config('configs/config.yaml', cli=True)

    setup_dirs(cfg)
    DatasetTools.compose_dataset(cfg)
    SDF.generate_sdms(cfg)
    
    if torch.cuda.is_available():
        if cfg.WORLD_SIZE > 1:
            print(f'[INFO] Running in distributed mode on {cfg.WORLD_SIZE} GPUs ...')
            os.environ['NCCL_P2P_DISABLE'] = '0' if cfg.NCCL_P2P else '1'
            spawn(main, args = (cfg,), nprocs=cfg.WORLD_SIZE)
        else:
            print(f'[INFO] Running in non-distributed mode on {cfg.DEFAULT_DEVICE} ...')
            main(0, cfg)
    else:
        raise RuntimeError('This library requires a GPU with CUDA support. '
                           'Please verify the PyTorch installation and ensure that a compatible GPU is available.')