import os
from model.AuxilaryNet import AuxNet
from model.DenseNet import DenseNet
from model.UNet import UNet

import torch
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(conf):
    """
    Set up the distributed environment.
    
    Args:
        rank: The rank of the current process. Unique identifier for each process in the distributed training.
        world_size: Total number of processes participating in the distributed training.
    """
    
    # Address of the main node. Since we are doing single-node training, it's set to localhost.
    os.environ["MASTER_ADDR"] = conf.MASTER_ADDR
    
    # Port on which the master node is expected to listen for communications from workers.
    os.environ["MASTER_PORT"] = conf.MASTER_PORT
    
    # Set the current CUDA device to the specified device (identified by rank).
    # This ensures that each process uses a different GPU in a multi-GPU setup.
    torch.cuda.set_device(conf.GPU_ID)
    
    # Initialize the process group. 
    # 'backend' specifies the communication backend to be used, "nccl" is optimized for GPU training.
    init_process_group(backend="nccl", rank=conf.GPU_ID, world_size=conf.GPU_COUNT)

def ddp_cleanup():
    destroy_process_group()

def initModel(conf):
    if conf.MODEL == 'AuxNet':
        model = AuxNet(conf)
        if conf.LOAD_PATH is not None:
            pretrained = torch.load(conf.LOAD_PATH)['state_dict']
            for name, child in model.named_children():
                if name in conf.TO_FREEZE:
                    pretrained_dict = {key.replace(f'{name}.',''): value for key, value in pretrained.items() if name in key}
                    child.load_state_dict(pretrained_dict)
                    for param in child.parameters():
                        param.requires_grad = False
    
    elif conf.MODEL == 'DenseNet':
        model = DenseNet(conf)

    elif conf.MODEL == 'UNet':
        model = UNet(conf)

    return model
