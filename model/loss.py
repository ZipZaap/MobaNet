import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace
from utils.util import gather_tensors
from configs.cfgparser import Config


# --- Cross-Entropy (pixel-wise) Losses ---
class SegCE(nn.Module):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates Cross-Entropy loss on the segmentation logits.

        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes:
                - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model
                - `gt_mask` | torch.Tensor (B, C, H, W) | 1-hot encoded ground truth segmentation mask.

        Returns
        -------
            loss : torch.Tensor
                The computed Segmentation Cross-Entropy loss.

        Raises
        ------
            ValueError
                If `inputs.seg_logits` or `inputs.gt_mask` is None.
        """

        if inputs.seg_logits is None or inputs.gt_mask is None:
            raise ValueError("Inputs must contain 'seg_logits' and 'gt_mask' attributes.")

        # (B, C, H, W) → (B, H, W)
        target =  inputs.gt_mask.argmax(dim=1)

        # compute CE loss → scalar
        loss = F.cross_entropy(inputs.seg_logits, target)
        return loss
    
class WeightedSegCE(nn.Module):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates Weighted Cross-Entropy loss on the segmentation logits.
        The weights are derived from the ground truth Signed Distance Map (`gt_sdm`).

        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes: 
                - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
                - `gt_mask` | torch.Tensor (B, C, H, W) | The ground truth segmentation mask.
                - `gt_sdm` | torch.Tensor (B, 1, H, W) | The ground truth Signed Distance Map.

        Returns
        -------
            loss : torch.Tensor
                The computed weighted binary cross-entropy loss.

        Raises
        ------
            ValueError
                If `inputs.seg_logits`, `inputs.gt_mask`, or `inputs.gt_sdm` is None.
        """

        if inputs.seg_logits is None or inputs.gt_mask is None or inputs.gt_sdm is None:
            raise ValueError("Inputs must contain 'seg_logits', 'gt_mask', and 'gt_sdm' attributes.")

        # (B, C, H, W) → (B, H, W)
        target = inputs.gt_mask.argmax(dim=1)

        # build per-pixel weight map (B, H, W)
        weight_map = ((2 - inputs.gt_sdm.abs()) ** 2).squeeze(1)

        # compute raw per-pixel CE loss → (B, H, W)
        per_pixel_loss = F.cross_entropy(
            inputs.seg_logits,
            target,
            reduction="none"
        )

        # apply per-pixel weights; average over batch & spatial dims → scalar
        loss = torch.mean(per_pixel_loss * weight_map)
        return loss

class ClsCE(nn.Module):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates cross-entropy loss on the classification logits.

        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes:
                - `cls_logits` | torch.Tensor (B, C) | The classification logits output by the model.
                - `gt_cls` | torch.Tensor (B, C) | The ground truth class labels.

        Returns
        -------
            loss : torch.Tensor
                The computed cross-entropy loss.

        Raises
        ------
            ValueError
                If `inputs.cls_logits` or `inputs.gt_cls` is None.
        """

        if inputs.cls_logits is None or inputs.gt_cls is None:
            raise ValueError("Inputs must contain 'cls_logits' and 'gt_cls' attributes.")

        loss = F.cross_entropy(inputs.cls_logits, inputs.gt_cls)
        return loss

# --- MAE (sdm-based) Losses ---
class BaseMAE(nn.Module):
    def __init__(self, cfg: Config):
        """ 
        Initializes the BaseMAE class.

        Args
        ----
            cfg : Config
                Configuration object containing the following attributes:
                - `.CLAMP_DELTA` (float): Clamping value for the clamped MAE loss.
                - `.SIGMOID_STEEPNESS` (int): Steepness of the sigmoid in the Sign loss component.
                - `.SDM_SMOOTHING` (bool): If True, the overall SDM is calculated smoothly using `logsumexp`.
                - `.SDM_SMOOTHING_ALPHA` (float): Smoothing factor for `logsumexp`.
        """

        super().__init__()
        self.delta: float = cfg.CLAMP_DELTA
        self.k: int = cfg.SIGMOID_STEEPNESS
        self.smooth: bool = cfg.SDM_SMOOTHING
        self.alpha: float = cfg.SDM_SMOOTHING_ALPHA
        self.q = 4

    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def sdm_union(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the Signed Distance Map (SDM) from the segmentation logits.
        Fuses the per-class SDMs into a single mask-wide map.

        Args
        ----
            logits : torch.Tensor (B, C, H, W)
                The segmentation logits output by the model.

        Returns
        -------
            sdm : torch.Tensor (B, 1, H, W)
                The computed signed distance map.   
        """

        # nomralize logits to [-1, 1] range
        sdm = torch.tanh(logits)

        # Fuse SDMs across channels: (B, C, H, W) → (B, 1, H, W)
        if self.smooth:
            sdm = torch.logsumexp(-self.alpha * sdm, dim=1, keepdim=True) / self.alpha
        else:
            sdm = torch.min(sdm, dim=1, keepdim=True).values

        return sdm
    
class MAE(BaseMAE):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates mean absolute error loss on the segmentation logits. 
        The loss is computed as the mean absolute difference between the predicted (`pd_sdm`)
        and the ground truth (`gt_sdm`) Signed Distance Maps.

        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes:
                - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
                - `gt_sdm` | torch.Tensor (B, 1, H, W) | The ground truth signed distance map.

        Raises
        ------
            ValueError
                If `inputs.seg_logits` or `inputs.gt_sdm` is None.
                
        Returns
        -------
            loss : torch.Tensor
                The computed mean absolute error loss.
        """

        if inputs.seg_logits is None or inputs.gt_sdm is None:
            raise ValueError("Inputs must contain 'seg_logits' and 'gt_sdm' attributes.")

        # logits (B, C, H, W) → SDM (B, 1, H, W)
        pd_sdm = self.sdm_union(inputs.seg_logits)

        # compute MAE loss → scalar
        loss = torch.mean(torch.abs(pd_sdm - inputs.gt_sdm))
        return loss
    
class ClampedMAE(BaseMAE):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates clamped mean absolute error loss on the segmentation logits.
        The loss is computed as the mean absolute difference between the predicted (`pd_sdm`)
        and the ground truth (`gt_sdm`) signed distance maps, clamped to a specified range
        defined by `cfg.CLAMP_DELTA`.

        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes:
                - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
                - `gt_sdm` | torch.Tensor (B, 1, H, W) | The ground truth Signed Distance Map.
            
        Returns
        -------
            loss : torch.Tensor
                The computed clamped mean absolute error loss.

        Raises
        ------
            ValueError
                If `inputs.seg_logits` or `inputs.gt_sdm` is None.
        """

        if inputs.seg_logits is None or inputs.gt_sdm is None:
            raise ValueError("Inputs must contain 'seg_logits' and 'gt_sdm' attributes.")

        # logits (B, C, H, W) → SDM (B, 1, H, W)
        pd_sdm = self.sdm_union(inputs.seg_logits)

        # clamp the pd/gt SDMs to [-δ; δ] range
        loss = torch.abs(torch.clamp(pd_sdm, -self.delta, self.delta) - 
                         torch.clamp(inputs.gt_sdm, -self.delta, self.delta))
        
        # average over batch & spatial dims → scalar
        loss = torch.mean(loss / (2 * self.delta))
        return loss
    
class SignMAE(BaseMAE):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates a custom Sign loss based on the signed distance map (`pd_sdm`) of the segmentation logits.
        This loss function intuitively applies additional penalties only to predicted distances that do not
        share the same sign as the ground truth. The penalty increases quadratically with
        the magnitude of the incorrect prediction and is modulated by the constant `q`.

        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes:
                - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
                - `gt_mask` | torch.Tensor (B, C, H, W) | The ground truth segmentation mask.
                - `gt_sdm` | torch.Tensor (B, 1, H, W) | The ground truth Signed Distance Map.

        Raises
        ------
            ValueError
                If `inputs.seg_logits`, `inputs.gt_mask`, or `inputs.gt_sdm` is None.

        Returns
        -------
            loss : torch.Tensor
                The computed Sign loss.
        """

        if inputs.seg_logits is None or inputs.gt_mask is None or inputs.gt_sdm is None:
            raise ValueError("Inputs must contain 'seg_logits', 'gt_mask', and 'gt_sdm' attributes.")

        # logits (B, C, H, W) → SDM (B, 1, H, W)
        pd_sdm = self.sdm_union(inputs.seg_logits)

        # calculate the Sign loss component → scalar
        sign = torch.sigmoid(pd_sdm * self.k * (1 - 2 * inputs.gt_mask))
        sign_loss = torch.mean(self.q * sign * (pd_sdm ** 2))

        # calculate the MAE loss component → scalar
        mae_loss = torch.mean(torch.abs(pd_sdm - inputs.gt_sdm))

        # combine the two components → scalar
        return (sign_loss + mae_loss) / 2

# --- DICE/IoU (area) Losses ---
class BaseDiceLoss(nn.Module):
    def __init__(self, cfg: Config):
        """
        Initializes the BaseDiceLoss class.

        Args
        ----
            cfg: Config
                Configuration object containing the following attributes:
                - `.SIGMOID_STEEPNESS` (int): Steepness of the sigmoid function for hard DICE loss.
        """
        super().__init__()
        self.k = cfg.SIGMOID_STEEPNESS
        self.eps = 1e-6

    def forward(self, inputs: SimpleNamespace):
        raise NotImplementedError

    def compute_loss(self, 
                     gt: torch.Tensor, 
                     pd: torch.Tensor) -> torch.Tensor:
        """
        Computes the DICE loss between the ground truth and predicted masks.

        Args
        ----
            gt : torch.Tensor (B, C, H, W).
                1-hot encoded ground truth mask.

            pd : torch.Tensor (B, C, H, W).
                Probabilistic (softmax) predicted mask.

        Returns
            loss: torch.Tensor
                The computed DICE loss.
        """

        # sum over batch and spatial dimensions
        # pd/gt (B, C, H, W) → per_class_dice (C,)
        dims = (0, 2, 3)
        numer = torch.sum(gt * pd, dims)
        denom = torch.sum(gt, dims) + torch.sum(pd, dims)
        per_class_dice = (2 * numer + self.eps) / (denom + self.eps)

        # avoid DICE computation on empty classes
        # loss (C, ) → scalar
        valid  = torch.sum(gt, dims) > 0         
        if valid.any():
            loss = 1.0 - per_class_dice[valid].mean() 
        else:                                            
            loss = torch.tensor(0., device=gt.device)
        return loss

class SoftDice(BaseDiceLoss):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates Soft (probabilistic) DICE loss on the segmentation logits.
        Every voxel contributes to the overlap score, so gradients
        remain informative even when the network is undecided. 
        
        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes:
                - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
                - `gt_mask` | torch.Tensor (B, C, H, W) | The ground truth segmentation mask.

        Returns
        -------
            loss : torch.Tensor
                The computed soft DICE loss.

        Raises
        ------
            ValueError
                If `inputs.seg_logits` or `inputs.gt_mask` is None.
        """

        if inputs.seg_logits is None or inputs.gt_mask is None:
            raise ValueError("Inputs must contain 'seg_logits' and 'gt_mask' attributes.")

        # logits (B, C, H, W) → probabilities (B, C, H, W)
        pd_mask = F.softmax(inputs.seg_logits, dim=1)

        # compute DICE loss; pd/gt (B, C, H, W) → scalar
        return self.compute_loss(inputs.gt_mask, pd_mask)

class HardDice(BaseDiceLoss):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """Dice with temperature-scaled softmax to sharpen predictions.

        Calculates Hard (discrete) DICE loss on the segmentation logits.
        Applies a steep sigmoid, so that the activation curve approaches a step function.  
        The narrow transition zone encourages the network to emit probabilities close to 0 or 1,
        effectively training it to behave as if hard thresholding had already been applied.
        This sharpens object contours but can make optimisation less forgiving.
        
        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes:
                - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
                - `gt_mask` | torch.Tensor  (B, C, H, W) | The ground truth segmentation mask.

        Returns
        -------
            loss : torch.Tensor
                The computed hard DICE loss.

        Raises
        ------
            ValueError
                If `inputs.seg_logits` or `inputs.gt_mask` is None.
        """

        if inputs.seg_logits is None or inputs.gt_mask is None:
            raise ValueError("Inputs must contain 'seg_logits' and 'gt_mask' attributes.")

        # logits (B, C, H, W) → probabilities (B, C, H, W)
        pd_mask = F.softmax(self.k * inputs.seg_logits, dim=1)

        # compute DICE loss; pd/gt (B, C, H, W) → scalar
        return self.compute_loss(inputs.gt_mask, pd_mask)

class IoU(nn.Module):
    def __init__(self, cfg: Config):
        """
        Initializes the IoU loss class.

        Args
        ----
            cfg : Config
                Configuration object containing the following attributes:
                - `.SIGMOID_STEEPNESS` (int): Steepness of the sigmoid function for hard DICE loss.
        """

        super().__init__()
        self.k = cfg.SIGMOID_STEEPNESS
        self.eps = 1e-6
    
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates Intersection over Union (IoU) loss on the segmentation logits.

        Args
        ----
            inputs : SimpleNamespace
                An object containing the following attributes:
                - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
                - `gt_mask` | torch.Tensor (B, C, H, W) | The ground truth segmentation mask.

        Returns
        -------
            loss : torch.Tensor
                The computed IoU loss.

        Raises
        ------
            ValueError
                If `inputs.seg_logits` or `inputs.gt_mask` is None.
        """

        if inputs.seg_logits is None or inputs.gt_mask is None:
            raise ValueError("Inputs must contain 'seg_logits' and 'gt_mask' attributes.")

        # logits (B, C, H, W) → probabilities (B, C, H, W)
        pd_mask = F.softmax(self.k * inputs.seg_logits, dim=1)

        # sum over batch and spatial dimensions
        # pd/gt (B, C, H, W) → per_class_iou (C,)
        dims = (0, 2, 3)
        inter = torch.sum(inputs.gt_mask * pd_mask, dims)
        union = torch.sum(inputs.gt_mask, dims) + torch.sum(pd_mask, dims) - inter
        per_class_iou = (inter + self.eps) / (union + self.eps)

        # avoid IoU computation on empty classes
        # loss (C,) → scalar
        valid = torch.sum(inputs.gt_mask, (0, 2, 3)) > 0
        if valid.any():
            loss = 1.0 - per_class_iou[valid].mean()
        else:
            loss = torch.tensor(0.0, device=inputs.gt_mask.device)
        return loss

# --- Combined Loss ---
class CombinedLoss(nn.Module):
    def __init__(self, 
                 losses: list[nn.Module], 
                 cfg: Config):
        """
        Combines multiple loss functions into a single loss function.
        The combined loss is computed as a weighted sum of the individual losses.
        The weights can be either fixed or adaptive, depending on the `cfg.ADAPTIVE_WEIGHTS`.

        Args
        ----
            losses : list[nn.Module]
                A list of loss functions to be combined.

            cfg : Config
                Configuration object containing the following attributes:
                - `.ADAPTIVE_WEIGHTS` (bool): If True, the weights are learned during training.
                - `.STATIC_WEIGHTS` (list): A list of static weights for each loss function.
                - `.DEVICE` (str): The device on which the loss functions will be computed.
        """

        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.adaptive: bool = cfg.ADAPTIVE_WEIGHTS
        self.device : str = cfg.DEVICE
        static_weights: list[int] = cfg.STATIC_WEIGHTS

        if self.adaptive:
            self.weights = nn.Parameter(torch.ones(len(self.losses), 
                                                   dtype=torch.float32, 
                                                   device=self.device))
        else:
            if static_weights:
                self.weights = torch.tensor(static_weights, 
                                            dtype=torch.float32, 
                                            device=self.device)
            else:
                self.weights = torch.ones(len(self.losses), 
                                          dtype=torch.float32, 
                                          device=self.device)

    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Computes the combined loss based on the individual losses and their weights.

        Args
        ----
            inputs : SimpleNamespace
                An inputs object containing the necessary attributes for the loss functions.

        Returns
        -------
            total_loss : torch.Tensor
                The computed combined loss.
        """

        # Collect individual loss values (N,) where N = len(self.losses)
        loss_values = torch.stack([loss(inputs) for loss in self.losses], dim=0)

        # Combine losses with weights; losses (N,) * weights (N,) → scalar
        if self.adaptive:
            total_loss = (loss_values / (2 * self.weights ** 2) + torch.log(self.weights)).sum()
        else:
            total_loss = (loss_values * self.weights).sum() / self.weights.sum()
        return total_loss

class Loss:
    def __init__(self, cfg: Config):
 
        """
        Initializes the loss function based on the specified `cfg.LOSS` string.
        Currently supported loss functions include:
        - `SoftDICE`: Soft DICE loss
        - `HardDICE`: Hard DICE loss
        - `IoU`: Intersection over Union loss
        - `SegCE`: Segmentation Cross-Entropy loss
        - `wSegCE`: Weighted Segmentation Cross-Entropy loss
        - `ClsCE`: Classification Cross-Entropy loss
        - `MAE`: Mean Absolute Error loss
        - `cMAE`: Clamped Mean Absolute Error loss
        - `sMAE`: Signed Mean Absolute Error loss
        - `Boundary`: Boundary loss
        
        Args
        ----
            cfg : Config
                Configuration object that contains the following attributes:
                - `.DEVICE` (str): The device on which the loss function will be computed (e.g. `cuda:0`).
                - `.LOSS` (str): A string representing the loss function to be used. Can either be a single
                                 loss function or a combination of multiple loss functions,
                                 separated by underscores (e.g., `hardDICE_MAE`). 
                - `.WORLD_SIZE` (int): The number of GPUs used for training.
        """

        self.device: str = cfg.DEVICE
        self.lname: str = cfg.LOSS
        self.worldsize: int = cfg.WORLD_SIZE
        self.totalLoss: torch.Tensor = torch.tensor(0, 
                                                    dtype=torch.float32, 
                                                    device=self.device)
        loss_map = {
            'SoftDICE': SoftDice(cfg),
            'HardDICE': HardDice(cfg),
            'IoU': IoU(cfg),
            'SegCE': SegCE(),
            'wSegCE': WeightedSegCE(),
            'ClsCE': ClsCE(),
            'MAE': MAE(cfg),
            'cMAE': ClampedMAE(cfg),
            'sMAE': SignMAE(cfg),
        }

        if '_' in self.lname:
            loss_lst = self.lname.split('_')
            self.lfunc = CombinedLoss([loss_map[lname] for lname in loss_lst], cfg)
        else:
            self.lfunc = loss_map[self.lname]

    def update(self, 
               logits: dict[str, torch.Tensor], 
               batch: dict[str, torch.Tensor]):
        """
        Updates the loss function with the current batch of logits and ground truth values.

        Args
        ----
            logits : dict[str, torch.Tensor]
                A dictionary containing the model's output logits.

            batch : dict[str, torch.Tensor]
                A dictionary containing the ground truth values.
        """
        
        inputs = SimpleNamespace(
            seg_logits=logits.get('seg'),
            cls_logits=logits.get('cls'),
            gt_mask=batch.get('mask'),
            gt_sdm=batch.get('sdm'),
            gt_cls=batch.get('cls')
        )

        self.loss = self.lfunc(inputs)
        self.totalLoss = self.totalLoss + self.loss.detach()

    def compute_avg(self, 
                    length: int
                    ) -> dict[str, float]:
        """
        Computes the average loss over the entire dataset.

        Args
        ----
            length : int
                The number of batches in the dataset.

        Returns
        -------
            avgLoss : dict[str, float]
                A dictionary containing the average loss values.
        """

        avgLoss = {'loss': self.totalLoss/length}

        if self.worldsize > 1:
            avgLoss = gather_tensors(avgLoss, self.worldsize) 

        return {k:round(v.item(), 4) for k,v in avgLoss.items()} 

    def backprop(self):
        self.loss.backward()

    def reset(self):
        self.totalLoss: torch.Tensor = torch.tensor(0, 
                                                    dtype=torch.float32, 
                                                    device=self.device)
        

# class Boundary(nn.Module):    
#     def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
#         """
#         Calculates a Boundary loss based on the Signed Distance Map (`pd_sdm`) of the segmentation logits.
#         See (Kevardec et. al, 2019; https://arxiv.org/abs/1812.07032) for more details.
        
#         Args
#         ----
#             inputs: SimpleNamespace
#                 An object containing the following attributes:
#                 - `seg_logits` | torch.Tensor(B, C, H, W) | The segmentation logits output by the model.
#                 - `gt_sdm` | torch.Tensor(B, 1, H, W) | The ground truth Signed Distance Map.
#         Raises
#         ------
#             ValueError
#                 If `inputs.seg_logits` or `inputs.gt_sdm` is None.

#         Returns
#         -------
#             loss : torch.Tensor
#                 The computed Boundary loss.
#         """
#         if inputs.seg_logits is None or inputs.gt_sdm is None:
#             raise ValueError("Inputs must contain 'seg_logits' and 'gt_sdm' attributes.")
        
#         B = inputs.seg_logits.shape[0]

#         pd_mask = F.softmax(inputs.seg_logits, dim=1)

#         loss = (inputs.gt_sdm * pd_mask).view(B, -1).sum(dim=1)
#         loss = loss.mean()
        # return loss