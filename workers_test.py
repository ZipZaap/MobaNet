import os
import torch
from time import time
from torch.optim import Adam
import torch.multiprocessing as mp

from model.loss import getTotalLoss
from configs.config_parser import Config
from data_loader.dataset import getDataloaders
from engines.trainer import AuxTrainer, ClsTrainer
from utils.util import initModel, ddp_setup, ddp_cleanup

os.environ['NCCL_P2P_DISABLE'] = '1'

def main(gpu_id, conf):
    conf.GPU_ID = gpu_id
    
    if conf.GPU_COUNT > 1:
        ddp_setup(conf)
    
    for num_workers in range(0, 16, 1):  
        conf.NUM_WORKERS = num_workers
        trainLoader, testLoader = getDataloaders(conf)
        start = time()

        for epoch in range(1, 3):
            nullcontext() if conf.GPU_COUNT < 2 else trainLoader.sampler.set_epoch(epoch)
            for i, data in enumerate(trainLoader, 0):
                pass

        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
        
    if conf.GPU_COUNT > 1:
        ddp_cleanup()
    
if __name__ == "__main__":
    conf = Config().getConfig()
    if conf.GPU_COUNT > 1:
        mp.spawn(main, nprocs=conf.GPU_COUNT, args = (conf,))
    else:
        main(0, conf)