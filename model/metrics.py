import numpy as np

import torch
import torch.nn as nn

from utils.sdf import SDF
from utils.util import gather_metrics
from configs.config_parser import CONF

import torch
from utils.sdf import SDF


class SegmentationMetrics:
    @classmethod
    def dice(cls, 
             pd_mask: torch.Tensor, 
             gt_mask: torch.Tensor, 
             include_background: bool = CONF.INCLUDE_BACKGROUND
             ) -> torch.Tensor:
        
        if include_background:
            fg_inter = torch.sum(pd_mask * gt_mask)
            bg_inter = torch.sum((1 - pd_mask) * (1 - gt_mask))
            intersection = fg_inter + bg_inter
            union = torch.sum(gt_mask + (1 - gt_mask) + pd_mask + (1 - pd_mask))

        else:
            intersection = torch.sum(pd_mask * gt_mask)
            union = torch.sum(pd_mask) + torch.sum(gt_mask)

        return (2 * intersection + 1e-6) / (union + 1e-6)

    @classmethod
    def iou(cls, 
            pd_mask: torch.Tensor, 
            gt_mask: torch.Tensor,
            include_background: bool = CONF.INCLUDE_BACKGROUND
            ) -> torch.Tensor:

        fg_inter = torch.sum(pd_mask * gt_mask)
        fg_union = torch.sum(gt_mask + pd_mask - pd_mask * gt_mask)
        fg_iou = (fg_inter + 1e-6) / (fg_union + 1e-6)

        if include_background:
            bg_inter = torch.sum((1 - pd_mask) * (1 - gt_mask))
            bg_union = torch.sum((1 - gt_mask) + (1 - pd_mask) - (1 - pd_mask) * (1 - gt_mask))
            bg_iou = (bg_inter + 1e-6) / (bg_union + 1e-6)
            return (fg_iou + bg_iou) / 2
        else:
            return fg_iou

    @classmethod
    def boundary(cls, 
                 pd_mask: torch.Tensor, 
                 gt_mask: torch.Tensor, 
                 gt_sdm: torch.Tensor
                 ) -> tuple[torch.Tensor]:
        
        pd_sdm = torch.abs(SDF.sdf(pd_mask))
        gt_sdm = torch.abs(gt_sdm)

        gt_edges = SDF.compute_sobel_edges(gt_mask).bool()
        pd_edges = SDF.compute_sobel_edges(pd_mask).bool()

        fallback = torch.tensor(
            1.0, 
            dtype=torch.float32, 
            device=gt_sdm.device
        )

        asd, hd95, ad, d95 = [], [], [], []
        for gE, pE, gS, pS in zip(gt_edges, pd_edges, gt_sdm, pd_sdm):
            d1 = pS[gE != 0]
            d2 = gS[pE != 0]
            d = torch.cat((d1, d2))

            ad.append(d1.mean() if d1.numel() else fallback)
            d95.append(torch.quantile(d1, 0.95) if d1.numel() else fallback)
            asd.append(d.mean() if d.numel() else fallback)
            hd95.append(torch.quantile(d, 0.95) if d.numel() else fallback)

        asd = torch.stack(asd).mean()
        hd95 = torch.stack(hd95).mean()
        ad = torch.stack(ad).mean()
        d95 = torch.stack(d95).mean()
        return asd, hd95, ad, d95

        
class ClassificationMetrics:
    @classmethod
    def accuracy(cls,
                 cls_pred: torch.Tensor,
                 cls_gt: torch.Tensor):
        
        cls_gt_idx = cls_gt.argmax(dim=1)
        accuracy = (cls_pred == cls_gt_idx).float().mean()
        return accuracy
    
def logits2predictions(logits: dict[str, torch.Tensor], 
                       cls_threshold: float = CONF.CLS_THRESHOLD,
                       seg_threshold: float = CONF.SEG_THRESHOLD,
                       cls_classes: int = CONF.CLS_CLASSES
                       ) -> None:
    
    if cls_logits := logits.get('cls', None):
        cls_probs = nn.functional.softmax(cls_logits, dim=1)
        if cls_threshold > 1 / cls_classes:
            above_thresh = cls_probs > cls_threshold
            topk = above_thresh.int().argmax(dim=1)
            pd_cls = torch.where(~above_thresh.any(dim=1), 2, topk)
        else:
            pd_cls = cls_probs.argmax(dim=1)
    else:
        pd_cls = None

    if seg_logits := logits.get('seg', None):
        pd_mask = (torch.sigmoid(seg_logits) > seg_threshold).float()
        # if cls_mask:
        #     pd_mask[pd_cls == 0] = 0
        #     pd_mask[pd_cls == 1] = 1
    else:
        pd_mask = None

    return pd_cls, pd_mask

class Accuracy:
    def __init__(self,
                 metrics: list[str] = CONF.METRICS): 
        self.metrics = {metric:[] for metric in metrics}

    def update(self, 
               logits: dict[str, torch.Tensor], 
               batch: dict[str, torch.Tensor]
               ) -> None:
        
        with torch.no_grad():
            pd_cls, pd_mask = logits2predictions(logits)
            gt_mask = batch.get('mask', None)
            gt_cls = batch.get('cls', None)
            gt_sdm = batch.get('sdm', None)

            ttr = ClassificationMetrics.accuracy(pd_cls, gt_cls)
            dsc = SegmentationMetrics.dice(pd_mask, gt_mask)
            iou = SegmentationMetrics.iou(pd_mask, gt_mask)

            self.metrics['TTR'].append(ttr)
            self.metrics['DSC'].append(dsc)
            self.metrics['IoU'].append(iou)

            idx = gt_mask.ne(0).any((1,2,3)) & gt_mask.eq(0).any((1,2,3))
            if idx.any():
                asd, hd95, ad, d95 = SegmentationMetrics.boundary(pd_mask[idx], gt_mask[idx], gt_sdm[idx])
                cma = (dsc + iou + (1 - asd) + 2*(1 - ad)) / 5
                self.metrics['ASD'].append(asd)
                self.metrics['HD95'].append(hd95)
                self.metrics['AD'].append(ad)
                self.metrics['D95'].append(d95)
                self.metrics['CMA'].append(cma)

    def compute_avg(self, length: int) -> dict[str, float]:
        self.metrics = {k:torch.stack(v).mean() for k,v in self.metrics.items()}
        
        if CONF.NUM_GPU > 1:
            self.metrics = gather_metrics(self.metrics)

        self.metrics = {key:round(val.item(), 4) for key, val in self.metrics.items()}
        return self.metrics

    def reset(self):
        self.metrics = {metric:[] for metric in self.metrics.keys()}

# class SegmentationMetrics:
#     @classmethod
#     def dice(cls, 
#              pred_mask: torch.Tensor, 
#              gt_mask: torch.Tensor, 
#              include_background: bool = CONF.INCLUDE_BACKGROUND
#              ) -> torch.Tensor:
        
#         if include_background:
#             obj_inter = torch.sum(pred_mask * gt_mask)
#             bg_inter = torch.sum((1 - pred_mask) * (1 - gt_mask))
#             intersection = obj_inter + bg_inter
#             union = torch.sum(gt_mask + (1 - gt_mask) + pred_mask + (1 - pred_mask))

#         else:
#             intersection = torch.sum(pred_mask * gt_mask)
#             union = torch.sum(pred_mask) + torch.sum(gt_mask)

#         return (2 * intersection + 1e-6) / (union + 1e-6)

#     @classmethod
#     def iou(cls, 
#             pred_mask: torch.Tensor, 
#             gt_mask: torch.Tensor,
#             include_background: bool = CONF.INCLUDE_BACKGROUND
#             ) -> torch.Tensor:

#         fg_inter = torch.sum(pred_mask * gt_mask)
#         fg_union = torch.sum(gt_mask + pred_mask - pred_mask * gt_mask)
#         fg_iou = (fg_inter + 1e-6) / (fg_union + 1e-6)

#         if include_background:
#             bg_inter = torch.sum((1 - pred_mask) * (1 - gt_mask))
#             bg_union = torch.sum((1 - gt_mask) + (1 - pred_mask) - (1 - pred_mask) * (1 - gt_mask))
#             bg_iou = (bg_inter + 1e-6) / (bg_union + 1e-6)
#             return (fg_iou + bg_iou) / 2
#         else:
#             return fg_iou

#     @classmethod
#     def boundary(cls, 
#                  pred: dict[str, torch.Tensor], 
#                  batch: dict[str, torch.Tensor]
#                  ) -> tuple[torch.Tensor]:
        
#         pred_mask = pred['mask']
#         gt_mask = batch['mask']
#         gt_sdm = batch['sdm']
#         gt_cls = batch['cls']

#         # idx = (batch['cls'].argmax(dim=1) == 2)

#         pred_sdm = torch.abs(SDF.sdf(pred_mask))
#         gt_sdm = torch.abs(gt_sdm)

#         edges_gt = SDF.compute_sobel_edges(gt_mask)
#         edges_pred = SDF.compute_sobel_edges(pred_mask)

#         dG = torch.where(edges_gt != 0, pred_sdm, float('nan'))
#         dP = torch.where(edges_pred != 0, gt_sdm, float('nan'))
#         dGdP = torch.cat((dG, dP), dim=1).view(dP.shape[0], -1)

#         asd = torch.nanmean(torch.nanmean(dGdP, dim=1))
#         hd95 = torch.nanmean(torch.nanquantile(dGdP, 0.95, dim=1))

#         dG_flat = dG.view(dG.shape[0], -1)
#         ad = torch.nanmean(torch.nanmean(dG_flat, dim=1))
#         d95 = torch.nanmean(torch.nanquantile(dG_flat, 0.95, dim=1))
#         return asd, hd95, ad, d95


            # d1 = torch.where(gt_edges != 0, pd_sdm, float('nan'))
            # d2 = torch.where(pd_edges != 0, gt_sdm, float('nan'))
            # d = torch.cat((d1, d2), dim=1).view(d2.shape[0], -1)
            # asd = torch.nanmean(torch.nanmean(d, dim=1))
            # hd95 = torch.nanmean(torch.nanquantile(d, 0.95, dim=1))

            # d1_flat = d1.view(d1.shape[0], -1)
            # ad = torch.nanmean(torch.nanmean(d1_flat, dim=1))
            # d95 = torch.nanmean(torch.nanquantile(d1_flat, 0.95, dim=1))
            # return asd, hd95, ad, d95
    
# class ClassificationMetrics:
#     @classmethod
#     def accuracy(cls,
#                  cls_pred,
#                  cls_gt):
        
#         cls_gt_idx = cls_gt.argmax(dim=1)
#         accuracy = (cls_pred == cls_gt_idx).float().mean()
#         return accuracy

# def getDiceAcc(pred_mask: torch.Tensor, 
#                gt_mask: torch.Tensor, 
#                include_background: bool = CONF.INCLUDE_BACKGROUND):
    
#     if include_background == False:
#         Inter = torch.sum(torch.multiply(gt_mask, pred_mask))
#         Union = torch.sum(gt_mask) + torch.sum(pred_mask)
#     else: 
#         scarpPred, scarpTrue = pred_mask, gt_mask
#         backgPred, backgTrue = torch.ones_like(pred_mask) - pred_mask, torch.ones_like(gt_mask) - gt_mask
#         Inter = torch.sum(torch.multiply(scarpTrue, scarpPred) + torch.multiply(backgTrue, backgPred))
#         Union = torch.sum(scarpTrue + backgTrue + scarpPred + backgPred)

#     DSC = (2*Inter + 1e-6)/(Union + 1e-6)
#     return DSC

# def getJaccardAcc(pred_mask: torch.Tensor, 
#                   gt_mask: torch.Tensor, 
#                   include_background: bool = CONF.INCLUDE_BACKGROUND):
#     scarpPred, scarpTrue = pred_mask, gt_mask
#     backgPred, backgTrue = torch.ones_like(pred_mask) - pred_mask, torch.ones_like(gt_mask) - gt_mask
#     ScarpInter = torch.sum(torch.multiply(scarpTrue, scarpPred))
#     ScarpUnion = torch.sum(scarpTrue + scarpPred - torch.multiply(scarpTrue, scarpPred))
#     Scarp_IoU = (ScarpInter + 1e-6)/(ScarpUnion + 1e-6)
#     if include_background:
#         BackgInter = torch.sum(torch.multiply(backgTrue, backgPred))
#         BackgUnion = torch.sum(backgTrue + backgPred - torch.multiply(backgTrue, backgPred))
#         Backg_IoU = (BackgInter + 1e-6)/(BackgUnion + 1e-6)
#         JCC = (Backg_IoU + Scarp_IoU)/2
#     else:
#         JCC = Scarp_IoU

#     return JCC 
    
# def getBoundaryAcc(pred_mask: torch.Tensor,
#                    gt_mask: torch.Tensor, 
#                    gt_sdm: torch.Tensor):
    
#     pred_sdm = SDF.sdf(pred_mask)
#     pred_sdm, gt_sdm = torch.abs(pred_sdm), torch.abs(gt_sdm)
    
#     dG = SDF.compute_sobel_edges(gt_mask)
#     dP = SDF.compute_sobel_edges(pred_mask)

#     dG = torch.where(dG != 0, pred_sdm, float('nan'))
#     dP = torch.where(dP != 0, gt_sdm, float('nan'))
#     dGdP = torch.cat((dG, dP), dim=1).view(dP.shape[0],-1)

#     ASD = torch.nanmean(torch.nanmean(dGdP, dim=1))
#     HD95 = torch.nanmean(torch.nanquantile(dGdP, 0.95, dim=1))
#     D95 = torch.nanmean(torch.nanquantile(dG.view(dG.shape[0],-1) , 0.95, dim=1))
#     AD = torch.nanmean(torch.nanmean(dG.view(dG.shape[0],-1) , dim=1))
#     return ASD, HD95, AD, D95

# def getClsAcc(cls_logits: torch.Tensor, gt_cls: torch.Tensor):
#     pred_cls = nn.Softmax(dim=1)(cls_logits)
#     gt_cls = torch.argmax(gt_cls, dim=1)
#     if CONF.CLS_THRESHOLD > 1/CONF.CLS_CLASSES:
#         topk = torch.argmax((pred_cls > CONF.CLS_THRESHOLD).to(torch.int), dim=1)
#         pred_cls = torch.where(~(pred_cls > CONF.CLS_THRESHOLD).any(dim=1), 2, topk)
#     else:
#         pred_cls = torch.argmax(pred_cls, dim=1)
#     ttr = torch.mean((pred_cls == gt_cls).to(torch.float32))
#     return ttr, pred_cls

# def getPredMask(seg_logits: torch.Tensor, 
#                 pred_cls: torch.Tensor,
#                 thresh: float = CONF.SEG_THRESHOLD,
#                 cls_mask: bool = CONF.CLS_MASK):
#     pred_mask = torch.relu(torch.sign(torch.sigmoid(seg_logits)-thresh))
#     if cls_mask:
#         pred_mask = torch.where((pred_cls == 0).view(pred_mask.shape[0], 1, 1, 1), 
#                                 torch.zeros(pred_mask.shape[1:]).to(pred_mask.device), pred_mask)
#         pred_mask = torch.where((pred_cls == 1).view(pred_mask.shape[0], 1, 1, 1), 
#                             torch.ones(pred_mask.shape[1:]).to(pred_mask.device), pred_mask)
#     return pred_mask

# class Accuracy:
#     def __init__(self, device): 
#         self.device = device
#         self.metrics = ['TTR', 'CMA', 'DSC', 'IoU',
#                         'ASD', 'AD', 'HD95', 'D95']

#     def update(self, 
#                seg_logits: torch.Tensor, 
#                cls_logits: torch.Tensor, 
#                gt_mask: torch.Tensor, 
#                gt_sdm: torch.Tensor, 
#                gt_cls: torch.Tensor):
        
#         with torch.no_grad():
#             ttr, pred_cls = getClsAcc(cls_logits, gt_cls)
#             pred_mask = getPredMask(seg_logits, pred_cls)
#             dsc = getDiceAcc(pred_mask, gt_mask)
#             iou = getJaccardAcc(pred_mask, gt_mask)
#             self.metrics['TTR'].append(ttr)
#             self.metrics['DSC'].append(dsc)
#             self.metrics['IoU'].append(iou)

#             if CONF.BOUNDARY:
#                 sdm_idx = (torch.argmax(gt_cls, dim=1) == 2)
#                 if sdm_idx.sum() > 0:
#                     asd, hd95, ad, d95 = getBoundaryAcc(pred_mask[sdm_idx], gt_mask[sdm_idx], gt_sdm[sdm_idx])
#                     self.metrics['ASD'].append(asd)
#                     self.metrics['AD'].append(ad)
#                     self.metrics['HD95'].append(hd95)
#                     self.metrics['D95'].append(d95)      
            
#     def compute_avg(self):
#         if CONF.BOUNDARY:
#             for key, val in self.metrics.items():
#                 if val:
#                     self.metrics[key] = torch.nanmean(torch.stack(val))
#                 else:
#                     self.metrics[key] = torch.tensor(float('nan'), device = self.device) #val device
#             self.metrics['CMA'] = (self.metrics['DSC'] + self.metrics['IoU'] + (1 - self.metrics['ASD']) + 2*(1 - self.metrics['AD']))/5
#         else:
#             self.metrics = {key:torch.nanmean(torch.stack(val)) for key,val in self.metrics.items() if val}
        
#         if CONF.NUM_GPU > 1:
#             self.metrics = gather_metrics(self.metrics)
#         self.metrics = {key:round(val.item(), 4) for key, val in self.metrics.items()}
#         return self.metrics
    
#     def reset(self):
#         self.metrics = {metric:[] for metric in self.metrics}
