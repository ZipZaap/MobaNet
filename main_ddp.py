from torch.optim import Adam
import torch.multiprocessing as mp

from engines.SegTrainer import SegTrainer
from dataset.data_loaders import getDataloaders
from dataset.data_prepare import generate_sdms, get_tts
from utils.util import initModel, ddp_setup, ddp_cleanup

from configs import CONF

# os.environ['NCCL_P2P_DISABLE'] = '1'

def main(gpu_id, imIDs):

    print(f'[INFO] MODEL: {CONF.MODEL}, DATASET: {CONF.DSET}, LOSS: {CONF.LOSS}')
    if CONF.NUM_GPU > 1:
        ddp_setup(gpu_id)
 
    loaders = getDataloaders(gpu_id, imIDs)
    model = initModel(gpu_id)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONF.INIT_LR, weight_decay=1e-5)
    SegTrainer(gpu_id, model, optimizer, loaders).train()

    if CONF.NUM_GPU > 1:
        ddp_cleanup()
    
if __name__ == "__main__":
    imIDs = get_tts()
    generate_sdms(imIDs)

    if CONF.NUM_GPU > 1:
        print(f'[INFO] Running in distributed mode on {CONF.NUM_GPU} GPUs')
        mp.spawn(main, args=(imIDs), nprocs=CONF.NUM_GPU)
    else:
        print(f'[INFO] Running in non-distributed mode on {CONF.DEFAULT_DEVICE}')
        main(CONF.DEFAULT_DEVICE, imIDs)
  