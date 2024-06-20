import numpy as np

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss

# After applying the sigmoid /predMask/ is normalized in the range [0,1].
def compLoss(pred, true):
    predMask = pred.squeeze()
    predMask = torch.relu(torch.sign(torch.sigmoid(predMask)-0.5))
    trueMask = true.squeeze()
    
    # DICE Loss
    # TP_SACRP = torch.sum(torch.multiply(trueMask, predMask))
    # TP_BACKG = torch.sum(torch.multiply(torch.ones_like(trueMask) - trueMask, torch.ones_like(predMask) - predMask))
    # FPFN = trueMask.nelement() + predMask.nelement()
    # DICEL = 1 - (2*(TP_SACRP+TP_BACKG) + 1)/(FPFN + 1)

    # DICE Loss
    # TP_SACRP = torch.sum(torch.multiply(trueMask, predMask))
    # TPFPFN = torch.sum(trueMask) + torch.sum(predMask)
    # DICEL = 1 - (2*TP_SACRP + 1)/(TPFPFN + 1)


    # # IoU loss
    # Inter = torch.sum(torch.multiply(trueMask, predMask)) + 1 
    # Union = torch.sum(trueMask + predMask - torch.multiply(trueMask, predMask)) + 1
    # IoUL = 0
    # IoU = Inter/Union
    # for thresh in np.arange(0.5, 1,  0.05):
    #     if IoU > thresh:
    #         IoUL = IoUL + (1 - IoU)
    #     else:
    #         IoUL = IoUL + 1
    # IoUL = IoUL/10
    
    # BCE loss
    BCEfun = BCEWithLogitsLoss()
    BCL = BCEfun(pred, true)

    # Aggregated loss
    # loss = (2*BCL + 2*IoUL)/5

    return BCL

def getTotalLoss(segPred = None, segTrue = None, clsPred = None, clsTrue = None, lossType = 'segmentation'):
    segLoss = compLoss
    clsLoss = CrossEntropyLoss()
    if lossType == 'combined':
        totalLoss = 0.5*segLoss(segPred, segTrue) + 0.5*clsLoss(clsPred, clsTrue)
    elif lossType == 'segmentation':
        totalLoss = segLoss(segPred, segTrue)
    elif lossType == 'classification':
        totalLoss = clsLoss(clsPred, clsTrue)
    return totalLoss