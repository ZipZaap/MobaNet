import numpy as np

import torch
import torch.nn as nn
from torch.nn import BCELoss, BCEWithLogitsLoss

<<<<<<< Updated upstream
from configs import CONF
from utils.sdf import normalize_sdm

# SDM LOSSES
class CustomBCELoss(nn.Module):
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

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_sdm: torch.Tensor = None):
        if CONF.SDM_LOGITS:
            pred_sdm = torch.tanh(logits)
            loss = BCELoss()(pred_sdm, gt_sdm)
        else:
            if gt_sdm == None:
                loss = BCEWithLogitsLoss()(logits, gt_mask)
            else:
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
        pred_mask = torch.sigmoid(k*logits)
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

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, k: int = 1000):
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

    def forward(self, logits: torch.Tensor, gt_sdm: torch.Tensor, delta: float = None):
        pred_sdm = torch.tanh(logits)
        if delta is None:
            loss = torch.mean(torch.abs(pred_sdm - gt_sdm))
        else:
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
        pred_mask = torch.sigmoid(k*logits)
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
    
class Loss:
    def __init__(self, ltype = CONF.LOSS):
        self.ltype = ltype
        self.totalLoss = 0

    def update(self, epoch: int, logits: torch.Tensor, gt_mask: torch.Tensor, gt_sdm: torch.Tensor):
        gt_sdm = normalize_sdm(gt_mask, gt_sdm, mode = 'minmax')

        if self.ltype == 'bin_DICE':
            self.loss = BinaryDiceLoss()(logits, gt_mask)
        elif self.ltype == 'bin_BCE':
            self.loss = CustomBCELoss()(logits, gt_mask)
        elif self.ltype == 'bin_WeightedBCE':
            self.loss = CustomBCELoss()(logits, gt_mask, gt_sdm)
        elif self.ltype == 'bin_C1':
            self.loss = (BinaryDiceLoss()(logits, gt_mask) + 2*BinaryIoULoss()(logits, gt_mask) + 2*CustomBCELoss()(logits, gt_mask))/5
        elif self.ltype == 'sdm_DICE':
            self.loss = SDMDiceLoss()(logits, gt_mask)
        elif self.ltype == 'sdm_MAE':
            self.loss = SDMMAELoss()(logits, gt_sdm)
        elif self.ltype == 'sdm_C1':
            self.loss = (SDMDiceLoss()(logits, gt_mask) + SDMMAELoss()(logits, gt_sdm) + SDMProductLoss()(logits, gt_sdm))/3
        self.totalLoss += self.loss
        return self.loss
=======
from configs.config_parser import CONF
from utils.util import gather_metrics


# --- Standalone Loss Functions --- 
class WeightedBCE(nn.Module):
    def forward(self, inputs: SimpleNamespace):
        weights = torch.pow(2 - torch.abs(inputs.gt_sdm),2)
        loss = BCEWithLogitsLoss(weight = weights)(inputs.seg_logits, inputs.gt_mask)
        return loss
    
class BCE(nn.Module):
    def forward(self, inputs: SimpleNamespace):
        loss = BCEWithLogitsLoss()(inputs.seg_logits, inputs.gt_mask)
        return loss
    
class MAE(nn.Module):    
    def forward(self, inputs: SimpleNamespace):
        pred_sdm = torch.tanh(inputs.seg_logits)
        loss = torch.mean(torch.abs(pred_sdm - inputs.gt_sdm))
        return loss
    
class ClampedMAE(nn.Module):
    def __init__(self, delta: float = CONF.CLAMP_DELTA):
        super().__init__()
        self.delta = delta
    
    def forward(self, inputs: SimpleNamespace):
        pred_sdm = torch.tanh(inputs.seg_logits)
        loss = torch.abs(torch.clamp(pred_sdm, -self.delta, self.delta) - 
                         torch.clamp(inputs.gt_sdm, -self.delta, self.delta))
        loss = torch.mean(loss / (2 * self.delta))
        return loss
        
class Sign(nn.Module):
    def __init__(self, k: int = CONF.SIGMOID_STEEPNESS, q: int = 4):
        super().__init__()
        self.k = k
        self.q = q

    def forward(self, inputs: SimpleNamespace):
        pred_sdm = torch.tanh(inputs.seg_logits)
        sign = torch.sigmoid(pred_sdm * self.k * (1 - 2 * inputs.gt_mask))
        loss = torch.mean(self.q * sign * (pred_sdm ** 2))
        return loss

class Boundary(nn.Module):    
    def forward(self, inputs: SimpleNamespace):
        B = inputs.seg_logits.shape[0]
        pred_mask = torch.sigmoid(inputs.seg_logits)
        loss = (inputs.gt_sdm * pred_mask).view(B, -1).sum(dim=1)
        loss = loss.mean()
        return loss

class BaseDiceLoss(nn.Module):
    def __init__(self, 
                 k: int = CONF.SIGMOID_STEEPNESS, 
                 include_background: bool = CONF.INCLUDE_BACKGROUND):
        super().__init__()
        self.k = k
        self.include_background = include_background

    def forward(self, inputs: SimpleNamespace):
        raise NotImplementedError

    def compute_loss(self, gt_mask, pred_mask):
        if self.include_background:
            fg_loss = self._dice_score(gt_mask, pred_mask)
            bg_loss = self._dice_score(1 - gt_mask, 1 - pred_mask)
            dice_score = 0.5 * (fg_loss + bg_loss)
        else:
            dice_score = self._dice_score(gt_mask, pred_mask)
        return 1 - dice_score

    def _dice_score(self, gt, pred):
        intersection = torch.sum(gt * pred)
        union = torch.sum(gt) + torch.sum(pred)
        return (2 * intersection + 1e-6) / (union + 1e-6)

class SoftDice(BaseDiceLoss):
    def forward(self, inputs: SimpleNamespace):
        pred_mask = torch.sigmoid(inputs.seg_logits)
        return self.compute_loss(inputs.gt_mask, pred_mask)

class HardDice(BaseDiceLoss):
    def forward(self, inputs: SimpleNamespace):
        pred_mask = torch.sigmoid(self.k * inputs.seg_logits)
        return self.compute_loss(inputs.gt_mask, pred_mask)

class CrossEntropy(nn.Module):
    def forward(self, inputs: SimpleNamespace):
        loss = CrossEntropyLoss()(inputs.cls_logits, inputs.gt_cls)
        return loss
    
class IoU(nn.Module):
    def __init__(self, 
                 k = CONF.SIGMOID_STEEPNESS, 
                 include_background: bool = CONF.INCLUDE_BACKGROUND):
        super().__init__()
        self.k = k
        self.include_background = include_background
    
    def forward(self, inputs: SimpleNamespace):
        seg_logits, gt_mask = inputs.seg_logits, inputs.gt_mask
        pred_mask = torch.sigmoid(self.k * seg_logits)

        fg_iou = self.compute_iou(gt_mask, pred_mask)

        if self.include_background:
            bg_iou = self.compute_iou(1 - gt_mask, 1 - pred_mask)
            loss = 1 - (fg_iou + bg_iou) / 2
        else:
            loss = 1 - fg_iou

        return loss

    @staticmethod
    def compute_iou(mask_true, mask_pred, eps=1e-6):
        intersection = torch.sum(mask_true * mask_pred)
        union = torch.sum(mask_true + mask_pred - mask_true * mask_pred)
        return (intersection + eps) / (union + eps)

# --- Combined Loss Functions ---
class AdaptiveLoss(nn.Module):
    def __init__(self, 
                 losses: list, 
                 weights: list = CONF.WEIGHTS, 
                 adaptive: bool = CONF.ADAPTIVE_WEIGHTS):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.adaptive = adaptive

        if self.adaptive:
            self.weights = nn.Parameter(torch.ones(len(losses), dtype=torch.float32))
        else:
            if weights:
                self.weights = torch.tensor(weights, dtype=torch.float32)
            else:
                self.weights = torch.ones(len(losses), dtype=torch.float32)
        
    def forward(self, inputs):
        loss_values = [loss(inputs) for loss in self.losses]
        if self.adaptive:
            total_loss = sum((L / (2 * w ** 2)) + torch.log(w)
                             for w, L in zip(self.weights, loss_values))
        else:
            weighted_sum = sum(w * L for w, L in zip(self.weights, loss_values))
            total_loss =  weighted_sum / self.weights.sum()
        return total_loss

class SignMAE(AdaptiveLoss):
    def __init__(self):
        super().__init__([Sign(), MAE()])

class DiceMAE(AdaptiveLoss):
    def __init__(self):
        super().__init__([HardDice(), MAE()])

class DiceClampedMAE(AdaptiveLoss):
    def __init__(self):
        super().__init__([HardDice(), ClampedMAE()])

class DiceSignMAE(AdaptiveLoss):
    def __init__(self):
        super().__init__([HardDice(), Sign(), MAE])

class DiceBoundary(AdaptiveLoss):
    def __init__(self):
        super().__init__([HardDice(), Boundary()])

class DiceSignMAECE(AdaptiveLoss):
    def __init__(self):
        super().__init__([DiceSignMAE(), CrossEntropy()])

class DiceBCEIoU(AdaptiveLoss):
    def __init__(self):
        super().__init__([HardDice(), BCE(), IoU()])

class Loss:
    def __init__(self, loss: str = CONF.LOSS):
        loss_map = {
            'softDICE': SoftDice,
            'hardDICE': HardDice,
            'BCE': BCE,
            'wBCE': WeightedBCE,
            'MAE': MAE,
            'cMAE': ClampedMAE,
            'sMAE': SignMAE,
            'Boundary': Boundary,
            'CE': CrossEntropy,
            'DICE_Boundary': DiceBoundary,
            'DICE_MAE': DiceMAE,
            'DICE_cMAE': DiceClampedMAE,
            'DICE_sMAE': DiceSignMAE,
            'DICE_sMAE_CE': DiceSignMAECE,
            'DICE_BCE_IoU': DiceBCEIoU
        }

        self.lfunc = loss_map[loss]()

    def update(self, 
               logits: dict[str, torch.Tensor], 
               batch: dict[str, torch.Tensor]
               ):
        
        inputs = SimpleNamespace(
            seg_logits = logits.get('seg', None),
            cls_logits = logits.get('cls', None),
            gt_mask = batch.get('mask', None),
            gt_sdm = batch.get('sdm', None),
            gt_cls = batch.get('cls', None)
        )

        self.loss = self.lfunc(inputs)
        self.totalLoss += self.loss.detach()
>>>>>>> Stashed changes

    def compute_avg(self, length):
        self.totalLoss = self.totalLoss/length
        return self.totalLoss

    def reset(self):
        self.totalLoss = 0
