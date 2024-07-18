import os
import time
import json
import random
import numpy as np

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model.UNet import UNet
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
    init_process_group(backend="nccl", rank=gpu_id, world_size=CONF.NUM_GPU)

def ddp_cleanup():
    destroy_process_group()

def initModel(gpu_id):
    model = UNet()
    if CONF.NUM_GPU > 1:
        model = model.to('cuda')
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    else:
        model = model.to(gpu_id)

    return model

def timer(epoch, ts = None):
    if ts is not None:
        te = time.time()
        dt = np.round(te - ts)
        etc = np.round((CONF.NUM_EPOCHS - epoch - 1)*dt/60)
        return dt, etc
    else:
        return time.time()
    
def get_tts():
    if CONF.DSET == 'npld':
        typs = ['npld']
    elif CONF.DSET == 'bu':
        typs = ['bu']
    elif CONF.DSET == 'scarp':
        typs = ['scarp']
    elif CONF.DSET == 'noscarp':
        typs = ['npld', 'bu']
    elif CONF.DSET == 'all':
        typs = ['bu', 'npld', 'scarp']

    if os.path.exists(CONF.TTS_PATH):
        with open(CONF.TTS_PATH) as f:
            TTS = json.load(f)
        trainIDs = TTS["trainIDs"]
        testIDs = TTS["testIDs"]
    else:
        with open(CONF.LABELS_JSON_PATH) as f:
            lblDict = json.load(f)
               
        testIDs = []
        trainIDs = []
        for typ in typs:
            random.seed(CONF.SEED)
            Z = random.sample(lblDict['typ'][typ], len(lblDict['typ']['scarp']))
            Y = random.sample(Z, int(len(lblDict['typ']['scarp']) * CONF.TEST_SPLIT))
            X = list(set(Z).difference(set(Y)))

            testIDs.extend(Y)
            trainIDs.extend(X)
            
        TTS = {'trainIDs': trainIDs, 'testIDs': testIDs}
        with open(CONF.TTS_PATH, 'w') as tts:
            json.dump(TTS, tts)

    print(f"[INFO] found {len(trainIDs)} examples in the training set...")
    print(f"[INFO] found {len(testIDs)} examples in the test set...")

    return (trainIDs, testIDs)

