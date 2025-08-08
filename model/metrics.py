import torch
import torch.nn.functional as F

from utils.sdf import SDF
from utils.util import gather_tensors, logits_to_lbl, logits_to_msk
from configs.cfgparser  import Config

class SegmentationMetrics:
    """
    Class for computing segmentation metrics.
    """
    def __init__(self, cfg: Config):
        """
        Initializes the SegmentationMetrics class with the configuration object.

        Args
        ----
            cfg : Config
                Configuration object containing the following attributes:
                - `.CMA_COEFFICIENTS` (dict[str, int]): Coefficients for the Combined Mean Accuracy (CMA) calculation.
        """

        self.cfg: Config = cfg
        self.coef: dict[str, int] = cfg.CMA_COEFFICIENTS
        self.eps: float = 1e-6

    def dice(self, 
             pd_mask: torch.Tensor, 
             gt_mask: torch.Tensor
             ) -> torch.Tensor:
        """
        Computes the Dice Similarity Coefficient (DSC) between predicted and ground truth masks.

        Args
        ----
            pd_mask : torch.Tensor (B, C, H, W)
                1-hot encoded predicted mask.

            gt_mask : torch.Tensor (B, C, H, W)
                1-hot encoded ground truth mask.

        Returns
        -------
            dice : torch.Tensor
                Dice Similarity Coefficient.
        """

        # sum over batch and spatial dimensions 
        # pd/gt (B, C, H, W) → per_class_dice (C,)
        dims = (0, 2, 3)
        numer = torch.sum(gt_mask * pd_mask, dims)
        denom = torch.sum(gt_mask, dims) + torch.sum(pd_mask, dims)
        per_class_dice = (2 * numer + self.eps) / (denom + self.eps)

        # filter out classes with no ground truth
        # per_class_dice (C,) → scalar
        valid  = torch.sum(gt_mask, dims) > 0  
        return per_class_dice[valid].mean() 

    def iou(self, 
            pd_mask: torch.Tensor, 
            gt_mask: torch.Tensor
            ) -> torch.Tensor:
        """
        Computes the Intersection over Union (IoU) between predicted and ground truth masks.

        Args:
            pd_mask : torch.Tensor (B, C, H, W)
                1-hot encoded predicted mask.

            gt_mask : torch.Tensor (B, C, H, W)
                1-hot encoded ground truth mask.

        Returns
        -------
            iou : torch.Tensor
                Intersection over Union.
        """

        # sum over batch and spatial dimensions
        # pd/gt (B, C, H, W) → per_class_iou (C,)
        dims = (0, 2, 3)
        inter = torch.sum(gt_mask * pd_mask, dims)
        union = torch.sum(gt_mask, dims) + torch.sum(pd_mask, dims) - inter
        per_class_iou = (inter + self.eps) / (union + self.eps)

        # avoid IoU computation on empty classes
        # per_class_iou (C,) → scalar
        valid = torch.sum(gt_mask, (0, 2, 3)) > 0
        return per_class_iou[valid].mean()

    def boundary(self,
                 pd_mask: torch.Tensor,
                 gt_mask: torch.Tensor,
                 gt_sdm: torch.Tensor
                 ) -> tuple[torch.Tensor, ...]:
        """
        Computes the boundary (SDM-based) metrics between predicted and ground truth masks.

        Args
        ----
            pd_mask : torch.Tensor(B, C, H, W)
                1-hot encoded predicted mask.

            gt_mask : torch.Tensor(B, C, H, W)
                1-hot encoded ground truth mask.

            gt_sdm : torch.Tensor(B, 1, H, W)
                Ground truth Signed Distance Map.

        Returns
        -------
            asd, hd95, ad, d95 : tuple[torch.Tensor, ...]
                A tuple containing the following distance metrics:
                - Average Symmetric Distance (ASD)
                - Hausdorff Distance (HD95)
                - Average Distance one-way (AD)
                - Hausdorff Distance one-way (D95)
        """

        # mask (B, C, H, W) → edges (B, 1, H, W)
        gt_edges = SDF.compute_sobel_edges(gt_mask).bool()
        pd_edges = SDF.compute_sobel_edges(pd_mask).bool()

        # mask (B, C, H, W) → sdm (B, 1, H, W)
        pd_sdm = torch.abs(SDF.sdf(pd_mask, self.cfg))
        gt_sdm = torch.abs(gt_sdm)

        # if edges are empty, default to max penalty
        fallback = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=gt_sdm.device
        )

        # compute distances
        asd, hd95, ad, d95 = [], [], [], []
        for gE, pE, gS, pS in zip(gt_edges, pd_edges, gt_sdm, pd_sdm):
            d1 = pS[gE == True]
            d2 = gS[pE == True]
            d = torch.cat((d1, d2))

            asd.append(fallback if torch.isnan(d).any() else d.mean())
            hd95.append(fallback if torch.isnan(d).any()  else torch.quantile(d, 0.95))
            ad.append(fallback if torch.isnan(d1).any()  else d1.mean())
            d95.append(fallback if torch.isnan(d1).any()  else torch.quantile(d1, 0.95))

        # asd/hd95/ad/d95 (B,) → scalar
        asd = torch.stack(asd).mean()
        hd95 = torch.stack(hd95).mean()
        ad = torch.stack(ad).mean()
        d95 = torch.stack(d95).mean()
        return asd, hd95, ad, d95

    def combined_mean_accuracy(self, 
                               metrics: dict[str, torch.Tensor]
                               ) -> torch.Tensor:
        """
        Computes the Combined Mean Accuracy (CMA) based on the provided metrics and coefficients.

        Args
        ----
            metrics : dict[str, torch.Tensor]
                Dictionary containing the computed metrics.

        Returns
        -------
            cma : torch.Tensor
                Combined Mean Accuracy.
        """

        cma = metrics.get('DSC', torch.tensor(0.0)) * self.coef['DSC'] + \
            metrics.get('IoU', torch.tensor(0.0)) * self.coef['IoU'] + \
            (1 - metrics.get('ASD', torch.tensor(0.0))) * self.coef['ASD'] + \
            (1 - metrics.get('AD', torch.tensor(0.0))) * self.coef['AD'] + \
            (1 - metrics.get('HD95', torch.tensor(0.0))) * self.coef['HD95'] + \
            (1 - metrics.get('D95', torch.tensor(0.0))) * self.coef['D95']
        cma /= sum(self.coef.values())
        return cma

class ClassificationMetrics:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def accuracy(self,
                 pd_cls: torch.Tensor,
                 gt_cls: torch.Tensor
                 ) -> torch.Tensor:
        """
        Computes the accuracy of the classification predictions.

        Args
        ----
            pd_cls : torch.Tensor (B,)
                Predicted class labels.

            gt_cls : torch.Tensor (B, C)
                One-hot encoded ground truth class labels.

        Returns
        -------
            accuracy : torch.Tensor
                Accuracy of the predictions.
        """
        
        # gt_cls (B, C) → gt_cls_idx (B,)
        gt_cls_idx = gt_cls.argmax(dim=1)

        # (B,) → scalar
        accuracy = (pd_cls == gt_cls_idx).float().mean()
        return accuracy
    
class Accuracy:
    def __init__(self, cfg: Config): 
        """
        Initializes the Accuracy class with the configuration object.

        Args
        ----
            cfg : Config
                Configuration object containing the following attributes:
                - `.DISTANCE_METRICS` (bool): Whether to compute distance metrics.
                - `.CLS_THRESHOLD` (float | None): Threshold for classification.
                - `.WORLD_SIZE` (int): Number of GPUs used for training.
        """
        self.distance_metrics: bool = cfg.DISTANCE_METRICS
        self.cls_threshold: float | None = cfg.CLS_THRESHOLD
        self.worldsize: int = cfg.WORLD_SIZE

        self.metrics: dict[str, list[torch.Tensor]] = {}
        self.segMetrics = SegmentationMetrics(cfg)
        self.clsMetrics = ClassificationMetrics(cfg)

    def _logits2predictions(self,
                            logits: dict[str, torch.Tensor]
                            ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Converts logits to predictions for classification and segmentation tasks.
        For classification, if the maximum probability for a class exceeds the threshold,
        the class label is assigned; otherwise, the boundary-class label is assigned. 
        For segmentation, the predicted 1-hot mask is created by taking the class with the highest logit.

        Args
        ----
            logits : dict[str, torch.Tensor]
                Dictionary containing logits for classification and segmentation.

        Returns
        -------
            pd_cls : torch.Tensor (B,) | None
                Predicted class labels.
            
            pd_mask : torch.Tensor (B, C, H, W) | None
                Predicted segmentation masks.
        """

        cls_logits = logits.get('cls')
        if cls_logits is not None:
           pd_cls = logits_to_lbl(cls_logits, self.cls_threshold)
        else:
            pd_cls = None

        seg_logits = logits.get('seg')
        if seg_logits is not None:
            pd_mask = logits_to_msk(seg_logits, '1hot')
        else:
            pd_mask = None

        return pd_cls, pd_mask

    def update(self, 
               logits: dict[str, torch.Tensor], 
               batch: dict[str, torch.Tensor]):
        """
        Update the metrics dictionary with the values for the current batch.

        Args
        ----
            logits : dict[str, torch.Tensor]
                Dictionary containing logits for classification and segmentation.

            batch : dict[str, torch.Tensor]
                Dictionary containing ground truth labels.
        """
        
        with torch.inference_mode():
            pd_cls, pd_mask = self._logits2predictions(logits)
            gt_cls, gt_mask, gt_sdm = batch['cls'], batch['mask'], batch['sdm']

            if pd_cls is not None:
                ttr = self.clsMetrics.accuracy(pd_cls, gt_cls)
                self.metrics.setdefault('TTR', []).append(ttr)

            if pd_mask is not None: 
                dsc = self.segMetrics.dice(pd_mask, gt_mask)
                iou = self.segMetrics.iou(pd_mask, gt_mask)
                self.metrics.setdefault('DSC', []).append(dsc)
                self.metrics.setdefault('IoU', []).append(iou)
                
                if self.distance_metrics:
                    idx = (torch.argmax(gt_cls, dim=1) == 2)
                    if idx.any():
                        asd, hd95, ad, d95 = self.segMetrics.boundary(pd_mask[idx], gt_mask[idx], gt_sdm[idx])
                        self.metrics.setdefault('ASD', []).append(asd)
                        self.metrics.setdefault('AD', []).append(ad)
                        self.metrics.setdefault('HD95', []).append(hd95)
                        self.metrics.setdefault('D95', []).append(d95)

    def compute_avg(self, length: int) -> dict[str, float]:
        """
        Computes the average of the metrics over all batches.
        
        Args
        ----
            length : int
                Number of batches.

        Returns
        -------
            avgMetrics : dict[str, float]
                Dictionary of averaged metrics.
        """
        avgMetrics = {k:torch.stack(v).mean() for k,v in self.metrics.items()}
        avgMetrics['CMA'] = self.segMetrics.combined_mean_accuracy(avgMetrics)

        if self.worldsize > 1:
            avgMetrics = gather_tensors(avgMetrics, self.worldsize)

        return {k:round(v.item(), 4) for k, v in avgMetrics.items()}

    def reset(self):
        self.metrics: dict[str, list[torch.Tensor]] = {}