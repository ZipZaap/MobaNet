import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

# SDM LOSSES
class WeightedBCE(nn.Module):
    """
    TYPE: pixel-lvl loss

    The gound truth SDM is inverted an normalized in range [1;2].
    It is then used to weight the BCEWithLogitsLoss() that calculates the pixel-wise 
    difference between the logits and the ground truth binary mask.

    Args:
        logits (torch.Tensor) - raw outputs from the last layer of UNet
        gt_sdm (torch.Tensor) - ground truth Surface Distance Map
        gt_mask (torch.Tensor) - ground truth binary segmentation mask
    Returns:
        loss (torch.Tensor) - scalar loss value
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, gt_sdm: torch.Tensor, gt_mask: torch.Tensor):
        gt_sdm = 2 - torch.abs(gt_sdm)
        loss = BCEWithLogitsLoss(weight = gt_sdm)(logits, gt_mask)
        return loss


class SDMDiceLoss(nn.Module):
    """
    TYPE: region-lvl loss
    This loss can be viewed as a relaxed, differentiable Dice Coefficient. 
    By setting k = 1 The predictions are not binarized {0,1} but relaxed and taken as probabilities [0,1]

    Args:
        logits (torch.Tensor) - raw outputs from the last layer of UNet
        gt_mask (torch.Tensor) - ground truth binary segmentation mask
        k (int) - variable which dictates the steepnes of the Sigmoid; k -> inf, Sigmoid -> Heavside
    Returns:
        loss (torch.Tensor) - scalar loss value
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, k: int = 1):
        pred_mask = torch.sigmoid(-k*logits)
        Inter = torch.sum(torch.multiply(gt_mask, pred_mask))
        Union = torch.sum(gt_mask) + torch.sum(pred_mask)
        loss = 1 - (2*Inter + 1)/(Union + 1)
        return loss


class SDMProductLoss(nn.Module):
    """
    TYPE: Sign loss v1
    This loss proposed in (Xue et al. 2019) is designed to penalize the output SDM 
    for having the wrong sign and is normnalized [-0.33, 1] -> negative loss is assigned to pixels that share the same sign as ground truth

    Args:
        logits (torch.Tensor) - raw outputs from the last layer of UNet
        gt_sdm (torch.Tensor) - ground truth Surface Distance Map
    Returns:
        loss (torch.Tensor) - scalar loss value
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, gt_sdm: torch.Tensor):
        pred_sdm = torch.tanh(logits)
        num = torch.multiply(pred_sdm, gt_sdm)
        den = torch.multiply(pred_sdm, gt_sdm) + torch.pow(pred_sdm, 2) + torch.pow(gt_sdm, 2)
        loss = -1*torch.mean(num/den)
        return loss
    

class SDMQuadLoss(nn.Module):
    """
    TYPE: Sign loss v2 (mine)
    This loss is also designed to penalize the output SDM for having the wrong sign but is normnalized [0;1] ->
    no loss is assigned to pixels that share the same sign as ground truth

    Args:
        logits (torch.Tensor) - raw outputs from the last layer of UNet
        gt_mask (torch.Tensor) - ground truth binary segmentation mask
        k (int) - variable which dictates the steepnes of the Sigmoid; k -> inf, Sigmoid -> Heavside
    Returns:
        loss (torch.Tensor) - scalar loss value
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, k: int = 20):
        pred_sdm = torch.tanh(logits)
        quad = torch.pow(pred_sdm, 2)
        sign = 1 - torch.sigmoid(pred_sdm*k*(1-2*gt_mask))
        loss = torch.mean(quad*sign)
        return loss
    

class SDMMAELoss(nn.Module):
    """
    TYPE: boundary-lvl loss
    Calculates the Mean Absolute distance between predicted and ground truth SDMs

    Args:
        logits (torch.Tensor) - raw outputs from the last layer of UNet
        gt_sdm (torch.Tensor) - ground truth Surface Distance Map
        delta (float) - calmping value that controls the distance from the boundary over which we expect to maintain a metric.
                        Smaller values of δ can be used to concentrate network capacity on details near the boundary.
    Returns:
        loss (torch.Tensor) - scalar loss value
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, gt_sdm: torch.Tensor, delta: float = 0.5):
        pred_sdm = torch.tanh(logits)
        loss = torch.mean(torch.abs(torch.clamp(pred_sdm, min=-delta, max=delta) - torch.clamp(gt_sdm, min=-delta, max=delta)))
        return loss
    

class SDMMSELoss(nn.Module):
    """
    TYPE: boundary-lvl loss
    Calculates the Mean Squared distance between predicted and ground truth SDMs

    Args:
        logits (torch.Tensor) - raw outputs from the last layer of UNet
        gt_sdm (torch.Tensor) - ground truth Surface Distance Map
        delta (float) - calmping value that controls the distance from the boundary over which we expect to maintain a metric.
                        Smaller values of δ can be used to concentrate network capacity on details near the boundary.
    Returns:
        loss (torch.Tensor) - scalar loss value
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, gt_sdm: torch.Tensor, delta: float = 0.5):
        pred_sdm = torch.tanh(logits)
        loss = torch.mean(torch.pow(torch.clamp(pred_sdm, min=-delta, max=delta) - torch.clamp(gt_sdm, min=-delta, max=delta), 2))
        return loss


class HDLoss(nn.Module):
    """
    TYPE: boundary-lvl loss
    Calculates the Hausodrf Distance between predicted and ground truth boundaries. Represents the worst case mistmatch.

    Args:
        logits (torch.Tensor) - raw outputs from the last layer of UNet
        gt_sdm (torch.Tensor) - ground truth Surface Distance Map
        gt_mask (torch.Tensor) - ground truth binary segmentation mask
        k (int) - variable which dictates the steepnes of the Sigmoid; k -> inf, Sigmoid -> Heavside
    Returns:
        loss (torch.Tensor) - scalar loss value
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, gt_sdm: torch.Tensor, gt_mask: torch.Tensor, k: int = 1000):
        pred_mask = torch.sigmoid(-k*logits)
        pred_sdm = torch.tanh(logits)

        dP = torch.max(torch.abs(torch.multiply(gt_sdm, pred_mask)))
        dG = torch.max(torch.abs(torch.multiply(pred_sdm, gt_mask)))
        loss = torch.max(dP, dG)

        return loss


# BINARY MASK LOSSES
class BinaryDiceLoss(nn.Module):
    def __init__(self, include_background: bool = False):
        super().__init__()
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor):
        pred_mask = torch.sigmoid(logits)

        if self.include_background == False:
            Inter = torch.sum(torch.multiply(gt_mask, pred_mask))
            Union = torch.sum(gt_mask) + torch.sum(pred_mask)
            loss = 1 - (2*Inter + 1)/(Union + 1)
        elif self.include_background == True:
            scarpPred, scarpTrue = pred_mask, gt_mask
            backgPred, backgTrue = torch.ones_like(pred_mask) - pred_mask, torch.ones_like(gt_mask) - gt_mask
            Inter = torch.sum(torch.multiply(scarpTrue, scarpPred) + torch.multiply(backgTrue, backgPred))
            Union = torch.sum(scarpTrue + backgTrue + scarpPred + backgPred)
            loss = 1 - (2*Inter + 1)/(Union + 1)

        return loss

class BinaryIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor): 
        pred_mask = torch.sigmoid(logits)
        Inter = torch.sum(torch.multiply(gt_mask, pred_mask)) + 1 
        Union = torch.sum(gt_mask + pred_mask - torch.multiply(gt_mask, pred_mask)) + 1
        loss = 0
        IoU = Inter/Union
        for thresh in np.arange(0.5, 1,  0.05):
            if IoU > thresh:
                loss = loss + (1 - IoU)
            else:
                loss = loss + 1
        loss = loss/10

        return loss

# def compLoss(pred, true):
     
#     DICEL = diceloss(pred, true)
#     IoUL = iouloss(pred, true)
#     BCL = BCEWithLogitsLoss(pred, true)

#     return (2*BCL + 2*IoUL + DICEL)/5

# def getTotalLoss(segPred = None, segTrue = None, clsPred = None, clsTrue = None, lossType = 'segmentation'):
#     if lossType == 'combined':
#         totalLoss = 0.5*compLoss(segPred, segTrue) + 0.5*CrossEntropyLoss(clsPred, clsTrue)
#     elif lossType == 'segmentation':
#         totalLoss = compLoss(segPred, segTrue)
#     elif lossType == 'classification':
#         totalLoss = CrossEntropyLoss(clsPred, clsTrue)
#     return totalLoss