import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.MobaNet import MobaNet
from configs.cfgparser  import Config

class ProcessManager():
    def __init__(self, cfg: Config):
        """
        Initialize the subprocess for distributed training.

        Args
        ----
            cfg : Config
                Configuration object containing the following attributes:
                - `.RANK` (int): The rank of the process.
                - `.GPU_LIST` (list[int]): List of GPUs to use.
                - `.WORLD_SIZE` (int): Number of processes (GPUs) in the distributed training.
                - `.MASTER_ADDR` (str): Master address for distributed training.
                - `.MASTER_PORT` (str): Master port for distributed training.
                - `.FREEZE_LAYERS` (list[str]): List of layers to freeze.
                - `.EXP_DIR` (Path): Save directory of the current experiment.
                - `.CHECKPOINT` (str | None): Name of the checkpoint file (without .pth extension).
        """
        
        self.cfg: Config = cfg
        self.rank: int = cfg.RANK
        self.gpu_list: list[int] = cfg.GPUs
        self.worldsize: int = cfg.WORLD_SIZE
        self.master_addr: str = cfg.MASTER_ADDR
        self.master_port: str = cfg.MASTER_PORT
        self.freeze_layers: list[str] = cfg.FREEZE_LAYERS
        self.exp_dir: Path = cfg.EXP_DIR
        self.checkpoint: Path | None = cfg.CHECKPOINT

    def bind_to_device(self):
        """
        Set the device and initialize distributed if needed.
        """
     
        if self.worldsize > 1:
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = self.master_port

            self.gpu_id = self.gpu_list[self.rank]
            self.device = f"cuda:{self.gpu_id}"
            torch.cuda.set_device(self.gpu_id)

            init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=len(self.gpu_list)
            )
        else:
            self.gpu_id = self.gpu_list[0]
            self.device = f"cuda:{self.gpu_id}"
            torch.cuda.set_device(self.gpu_id)

        self.cfg.DEVICE = self.device

    def cleanup(self):
        """
        Cleanup the process group for distributed training.
        """
        if self.worldsize > 1:
            destroy_process_group()

    def load_model(self):
        """
        Load the model and its pre-trained weights (if specified).

        Returns:
            model : torch.nn.Module
                The loaded model.
        """
        model = MobaNet(self.cfg)

        if self.checkpoint:
            weights = torch.load(self.checkpoint, 
                                 map_location=self.device, 
                                 mmap=True, 
                                 weights_only=True)['weights']
            model.load_state_dict(weights, strict=False)      
            
        for layer_name in self.freeze_layers:
            layer = getattr(model, layer_name)
            for param in layer.parameters():
                param.requires_grad = False

        if self.worldsize > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model.to(self.device), 
                        device_ids=[self.gpu_id], 
                        find_unused_parameters=False)
        else:
            model = model.to(self.device)

        return model