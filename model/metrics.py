import numpy as np

import torch
import torch.distributed as dist

from utils.sdf import SDF, compute_sobel_edges
from configs import CONF

def getDiceAcc(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    Inter = torch.sum(torch.multiply(gt_mask, pred_mask))
    Union = torch.sum(gt_mask) + torch.sum(pred_mask)
    DSC = (2*Inter + 1)/(Union + 1)
    # DSC = DSC.cpu().detach().tolist()
    return DSC

def getJaccardAcc(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    Inter = torch.sum(torch.multiply(gt_mask, pred_mask)) + 1 
    Union = torch.sum(gt_mask + pred_mask - torch.multiply(gt_mask, pred_mask)) + 1
    JCC = Inter/Union
    return JCC 

def getBoundaryAcc(pred_mask: torch.Tensor, pred_sdm: torch.Tensor, gt_mask: torch.Tensor, gt_sdm: torch.Tensor):
    max_dist = (np.sqrt(2)*CONF.INPUT_IMAGE_SIZE/2)/4
    dist_filter = torch.where(gt_sdm < max_dist, 1, 0)

    dP = compute_sobel_edges(pred_mask)*dist_filter
    dP = torch.abs(torch.multiply(gt_sdm, dP))
    dP = torch.where(dP == 0, float('nan'), dP)

    dG = compute_sobel_edges(gt_mask)*dist_filter
    dG = torch.abs(torch.multiply(pred_sdm, dG))
    dG = torch.where(dG == 0, float('nan'), dG)

    dPdG = torch.cat((dP, dG), dim=0).view(dP.shape[0]*2, 1, -1)

    HD95 = torch.nanmean(torch.nanquantile(dPdG, 0.95, dim=2))
    ASD = torch.nanmean(torch.nanmean(dPdG, dim=2))
    return ASD, HD95

class Accuracy:
    def __init__(self, sdm_logits: bool = CONF.SDM_LOGITS, threshold: float = CONF.THRESHOLD):
        self.sdm_logits = sdm_logits
        self.T = threshold

        self.DSC = 0
        self.IoU = 0
        self.HD95 = 0
        self.ASD = 0

    def update(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_sdm: torch.Tensor):
        with torch.no_grad():
            pred_mask = torch.relu(torch.sign(torch.sigmoid(logits)-self.T))
            pred_sdm = SDF(pred_mask, kernel_size = CONF.SDM_KERNEL)

            self.DSC += getDiceAcc(pred_mask, gt_mask)
            self.IoU += getJaccardAcc(pred_mask, gt_mask)
            asd, hd95 = getBoundaryAcc(pred_mask, pred_sdm, gt_mask, gt_sdm)
            self.HD95 += hd95
            self.ASD += asd

    def compute_avg(self, length: int):
        self.DSC = self.DSC/length
        self.IoU = self.IoU/length
        self.HD95 = self.HD95/length
        self.ASD = self.ASD/length
        metrics = {'dsc': self.DSC, 'iou': self.IoU, 'hd95': self.HD95, 'asd': self.ASD}

        if CONF.NUM_GPU > 1:
            return gather_metrics(metrics)
        else:
            return metrics

    def reset(self):
        self.DSC = 0
        self.IoU = 0
        self.HD95 = 0
        self.ASD = 0

def gather_metrics(metrics: dict) -> dict:
    for metric, value in metrics:
        AvgAccLst = [torch.zeros_like(value for _ in range(CONF.NUM_GPU))]
        dist.all_gather(AvgAccLst, value)
        value = np.round(torch.mean(torch.stack(AvgAccLst)).item(), 4)
        metrics[metric] = value
    return metrics

# def gather_metrics(gpu_id, metrics: list) -> list:
#     output = []
#     for metric in metrics:
#         AvgAccLst = [torch.zeros_like(torch.tensor(metric)).to(gpu_id) for _ in range(CONF.NUM_GPU)]
#         dist.all_gather(AvgAccLst, torch.tensor(metric).to(gpu_id))
#         metric = np.round(torch.mean(torch.stack(AvgAccLst)).item(), 4)
#         output.append(metric)
#     return output
    
    