from torch.optim import Adam
import torch.multiprocessing as mp

from engines.SegTrainer import SegTrainer
from data_loader.dataset import getDataloaders
from utils.util import initModel, ddp_setup, ddp_cleanup

from configs import CONF

# os.environ['NCCL_P2P_DISABLE'] = '1'

def main(gpu_id):

    print(f'[INFO] MODEL: {CONF.MODEL_ID}, DATASET: {CONF.DSET}, LOSS: {CONF.LOSS}')
    if CONF.GPU_COUNT > 1:
        ddp_setup(gpu_id)
 
    trainLoader, testLoader = getDataloaders(gpu_id)
    model = initModel(gpu_id)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONF.INIT_LR, weight_decay=1e-5)
    SegTrainer(gpu_id, model, optimizer, testLoader, trainLoader).train()

    if CONF.GPU_COUNT > 1:
        ddp_cleanup()
    
if __name__ == "__main__":
    if CONF.GPU_COUNT > 1:
        mp.spawn(main, nprocs=CONF.GPU_COUNT)
    else:
        main(CONF.GPU_ID)
  