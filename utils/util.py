import os
import json
import random
import numpy as np
from scipy import ndimage
from sklearn.model_selection import KFold

import torch
from torch.distributed import init_process_group, destroy_process_group, all_gather
from torch.nn.parallel import DistributedDataParallel as DDP

from model.AuxNet import UNet
from configs.config_parser import CONF

class Subprocess():
    def __init__(self, 
                 rank,
                 gpu_list: list[int] = CONF.GPUs,
                 num_gpu: int = CONF.NUM_GPU,
                 master_addr: str = CONF.MASTER_ADDR,
                 master_port: str = CONF.MASTER_PORT,
                 load_layers: list[str] = CONF.LOAD,
                 freeze_layers: list[str] = CONF.FREEZE,
                 mdl_dir: str = CONF.MDL_DIR,
                 weights: str = CONF.PT_WEIGHTS
                 ) -> None:
        
        self.rank = rank
        self.gpu_list = gpu_list
        self.num_gpu = num_gpu
        self.master_addr = master_addr
        self.master_port = master_port
        self.load_layers = load_layers
        self.freeze_layers = freeze_layers
        self.mdl_dir = mdl_dir
        self.weights = weights
    
    def setup(self):
        if self.num_gpu > 1:
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = self.master_port

            self.device = self.gpu_list[self.rank]
            torch.cuda.set_device(self.device)
            init_process_group(backend="nccl", rank=self.rank, world_size=self.num_gpu)
        else:
            self.device = self.rank

    def cleanup(self):
        if self.num_gpu > 1:
            destroy_process_group()

    def load_model(self):
        model = UNet()
    
        # if (self.load_layers and self.weights):
        #     pretrained = torch.load(f"{self.mdl_dir}/{self.weights}.pth")
        #     for layer_name in self.load_layers:
        #         layer = getattr(model, layer_name)
        #         state_dict = {k.replace(f"{layer_name}.", ""): v for k, v in pretrained.items() if k.startswith(layer_name)}
        #         layer.load_state_dict(state_dict)
        
        if self.weights:
            checkpoint = torch.load(self.mdl_dir / f"{self.weights}.pth")
            model.load_state_dict(checkpoint, strict=False)      
        if self.freeze_layers:
            for layer_name in self.freeze_layers:
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False

        if self.num_gpu > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model.to('cuda'), 
                        device_ids=[self.device], 
                        find_unused_parameters=False)
        else:
            model = model.to(self.device)

def initModel(gpu_id):
    model = UNet()
    if CONF.NUM_GPU > 1:
        model = model.to('cuda')
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    else:
        model = model.to(gpu_id)

    return model

def gather_metrics(metrics: dict) -> dict:
    for metric, value in metrics.items():
        AvgAccLst = [torch.zeros_like(value) for _ in range(CONF.NUM_GPU)]
        all_gather(AvgAccLst, value)
        value = np.round(torch.nanmean(torch.stack(AvgAccLst)).item(), 4)
        metrics[metric] = value
    return metrics


def get_channel_dims(in_c, fdepth, ldepth):
    arr = (ldepth-1) - np.abs(np.arange(-(ldepth-1), 5))
    channels = [int(fdepth*(2**i)) for i in arr]
    out_c = channels + [in_c]
    in_c = [in_c] + channels
    channels = np.vstack((in_c, out_c)).T
    return channels


# def longest_edge_filter(mask_tensor):

#     def process_mask(mask):
#         edges, _ = ndimage.label(mask)
#         edge_len = np.bincount(edges.ravel())
#         edge_len[0] = 0
#         longest_edge_label = np.argmax(edge_len)
#         return (edges == longest_edge_label).astype(int)
    
#     mask_array = mask_tensor.detach().cpu().numpy()[:, 0]
#     vectorized_process = np.vectorize(process_mask, signature='(m,n)->(m,n)')
#     edges_array = vectorized_process(mask_array)
#     edge_tensor = torch.tensor(edges_array, device=mask_tensor.device).unsqueeze(1)
#     return edge_tensor


# def save_intermediate_masks(logits, gt_mask, source, epoch):

#     if epoch == CONF.WARMUP_EPOCHS + 1:
#         source = torch.split(source, 1, dim=0)
#         gt_mask = torch.split(gt_mask, 1, dim=0)
#         for  i, (sc, gt) in enumerate(zip(source, gt_mask)):
#             sc = sc[0][0].detach().cpu().numpy()
#             gt = gt[0][0].detach().cpu().numpy()

#             cv2.imwrite(f"{CONF.OUTPUT_PATH}/masks/im_{i}.png", sc * 255)
#             cv2.imwrite(f"{CONF.OUTPUT_PATH}/masks/gt_{i}.png", gt * 255)
#     else:
#         pred_mask = torch.relu(torch.sign(torch.sigmoid(logits)-CONF.THRESHOLD))
#         pred_mask = torch.split(pred_mask, 1, dim=0)
#         for  i, pd in enumerate(pred_mask):
#             pd = pd[0][0].detach().cpu().numpy()
#             cv2.imwrite(f"{CONF.OUTPUT_PATH}/masks/{i}.png", pd * 255)
