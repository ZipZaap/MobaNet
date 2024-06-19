import os
import time
import numpy as np
from model.AuxilaryNet import AuxNet
from model.DenseNet import DenseNet
from model.UNet import UNet

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from configs import CONF

def ddp_setup(gpu_id):
    """
    Set up the distributed environment.
    
    Args:
        rank: The rank of the current process. Unique identifier for each process in the distributed training.
        world_size: Total number of processes participating in the distributed training.
    """
    
    # Address of the main node. Since we are doing single-node training, it's set to localhost.
    os.environ["MASTER_ADDR"] = CONF.MASTER_ADDR
    
    # Port on which the master node is expected to listen for communications from workers.
    os.environ["MASTER_PORT"] = CONF.MASTER_PORT
    
    # Set the current CUDA device to the specified device (identified by rank).
    # This ensures that each process uses a different GPU in a multi-GPU setup.
    torch.cuda.set_device(gpu_id)
    
    # Initialize the process group. 
    # 'backend' specifies the communication backend to be used, "nccl" is optimized for GPU training.
    init_process_group(backend="nccl", rank=gpu_id, world_size=CONF.GPU_COUNT)

def ddp_cleanup():
    destroy_process_group()

def initModel(gpu_id):
    if CONF.MODEL == 'AuxNet':
        model = AuxNet(CONF)
        if CONF.LOAD_PATH is not None:
            pretrained = torch.load(CONF.LOAD_PATH)['state_dict']
            for name, child in model.named_children():
                if name in CONF.TO_FREEZE:
                    pretrained_dict = {key.replace(f'{name}.',''): value for key, value in pretrained.items() if name in key}
                    child.load_state_dict(pretrained_dict)
                    for param in child.parameters():
                        param.requires_grad = False
    elif CONF.MODEL == 'DenseNet':
        model = DenseNet()
    elif CONF.MODEL == 'UNet':
        model = UNet()

    if CONF.GPU_COUNT > 1:
        model = model.to('cuda')
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    else:
        model = model.to(gpu_id)

    return model

def gather_metrics(gpu_id, metrics: list) -> list:
    output = []
    for metric in metrics:
        AvgAccLst = [torch.zeros_like(torch.tensor(metric)).to(gpu_id) for _ in range(CONF.GPU_COUNT)]
        dist.all_gather(AvgAccLst, torch.tensor(metric).to(gpu_id))
        metric = np.round(torch.mean(torch.stack(AvgAccLst)).item(), 4)
        output.append(metric)
    return output

def timer(epoch, ts = None):
    if ts is not None:
        te = time.time()
        dt = np.round(te - ts)
        etc = np.round((CONF.NUM_EPOCHS - epoch - 1)*dt/60)
        return dt, etc
    else:
        return time.time()
