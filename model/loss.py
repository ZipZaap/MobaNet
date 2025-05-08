from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

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

    def compute_avg(self, length):
        self.totalLoss = self.totalLoss/length
        return self.totalLoss

    def reset(self):
        self.totalLoss = 0
