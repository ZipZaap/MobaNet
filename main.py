import os
import torch
from torch.optim import Adam
import torch.multiprocessing as mp

from utils.util import initModel
from engines.trainer import AuxTrainer, ClsTrainer
from model.loss import getTotalLoss
from configs.config_parser import Config
from data_loader.dataset import getDataloaders


def main(conf):
 
    trainLoader, testLoader = getDataloaders(conf)
    model = initModel(conf)
    lossFunc = getTotalLoss
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf.INIT_LR, weight_decay=1e-5)
    if conf.MODEL == 'AuxNet':
        AuxTrainer(conf, model, lossFunc, optimizer, testLoader, trainLoader).train()
    elif conf.MODEL == 'DenseNet':
        ClsTrainer(conf, model, lossFunc, optimizer, testLoader, trainLoader).train()
    
if __name__ == "__main__":
    conf = Config().getConfig()
    main(conf)
  