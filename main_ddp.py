import os
import torch
from torch.optim import Adam
import torch.multiprocessing as mp

from model.loss import getTotalLoss
from configs.config_parser import Config
from data_loader.dataset import getDataloaders
from engines.AuxTrainer import AuxTrainer
from engines.ClsTrainer import ClsTrainer
from engines.SegTrainer import SegTrainer
from utils.util import initModel, ddp_setup, ddp_cleanup

# os.environ['NCCL_P2P_DISABLE'] = '1'

def main(gpu_id, conf):
    conf.GPU_ID = gpu_id
    if conf.GPU_COUNT > 1:
        ddp_setup(conf)
 
    trainLoader, testLoader = getDataloaders(conf)
    model = initModel(conf)
    lossFunc = getTotalLoss
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf.INIT_LR, weight_decay=1e-5)
    
    if conf.MODEL == 'AuxNet':
        AuxTrainer(conf, model, lossFunc, optimizer, testLoader, trainLoader).train()
    elif conf.MODEL == 'DenseNet':
        ClsTrainer(conf, model, lossFunc, optimizer, testLoader, trainLoader).train()
    elif conf.MODEL == 'UNet':
        SegTrainer(conf, model, lossFunc, optimizer, testLoader, trainLoader).train()
    
    if conf.GPU_COUNT > 1:
        ddp_cleanup()
    
    
if __name__ == "__main__":
    conf = Config().getConfig()
    if conf.GPU_COUNT > 1:
        mp.spawn(main, nprocs=conf.GPU_COUNT, args = (conf,))
    else:
        main(0, conf)
  