import os
import torch.multiprocessing as mp

from engines.SegTrainer import SegTrainer
from utils.dataset import getDataloaders
from utils.sdf import SDF
from utils.dataset import DatasetManager as DM
from utils.util import Subprocess
from configs.config_parser import CONF

from utils.loggers import Logger

from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

def main(rank: int | str):
    logger = Logger(rank)
    process = Subprocess(rank)
    process.setup()

    for fold in CONF.FOLD_IDs:
        model = process.load_model()
        loaders = DM.get_dataloaders(rank, fold) 
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONF.BASE_LR, weight_decay=CONF.L2_DECAY)
        lr_scheduler = LinearLR(optimizer, start_factor=CONF.INIT_LR/CONF.BASE_LR, total_iters=CONF.WARMUP_EPOCHS) if CONF.WARMUP_EPOCHS != 0 else None

        logger.init_run(fold)
        SegTrainer(model, optimizer, lr_scheduler, loaders, logger).train()
        logger.end_run()

    process.cleanup()

if __name__ == "__main__":
    print(f'[INFO] MODEL: {CONF.MODEL}, DATASET: {CONF.TRAIN}/{CONF.TEST}, LOSS: {CONF.LOSS}, BATCH(global): {CONF.BATCH_SIZE}')
    
    DM.compose_dataset()
    SDF.generate_sdms()
    
    if CONF.NUM_GPU > 1:
        print(f'[INFO] Running in distributed mode on {CONF.NUM_GPU} GPUs')
        os.environ['NCCL_P2P_DISABLE'] = '1' if CONF.NCCL_P2P_DISABLE else '0'
        mp.spawn(main, nprocs=CONF.NUM_GPU)
    else:
        print(f'[INFO] Running in non-distributed mode on {CONF.DEFAULT_DEVICE}')
        main(CONF.DEFAULT_DEVICE)