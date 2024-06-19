from torch.optim import Adam

from utils.util import initModel
from engines.AuxTrainer import AuxTrainer
from engines.ClsTrainer import ClsTrainer
from engines.SegTrainer import SegTrainer
from model.loss import getTotalLoss
from data_loader.dataset import getDataloaders

from configs import CONF
import time

def main():
    trainLoader, testLoader = getDataloaders(CONF.GPU_ID)
    model = initModel(CONF.GPU_ID)
    lossFunc = getTotalLoss
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONF.INIT_LR, weight_decay=1e-5)
    if CONF.MODEL == 'AuxNet':
        AuxTrainer(CONF.GPU_ID, model, lossFunc, optimizer, testLoader, trainLoader).train()
    elif CONF.MODEL == 'DenseNet':
        ClsTrainer(CONF.GPU_ID, model, lossFunc, optimizer, testLoader, trainLoader).train()
    elif CONF.MODEL == 'UNet':
        SegTrainer(CONF.GPU_ID, model, lossFunc, optimizer, testLoader, trainLoader).train()
    
if __name__ == "__main__":
    main()
  

    