from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import torch

from utils.util import SDF

def getDiceAcc(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    Inter = torch.sum(torch.multiply(gt_mask, pred_mask))
    Union = torch.sum(gt_mask) + torch.sum(pred_mask)
    DSC = 1 - (2*Inter + 1)/(Union + 1)
    DSC = DSC.cpu().detach().tolist()
    return DSC

def getJaccardAcc(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    Inter = torch.sum(torch.multiply(gt_mask, pred_mask)) + 1 
    Union = torch.sum(gt_mask + pred_mask - torch.multiply(gt_mask, pred_mask)) + 1
    JCC = 0
    IoU = Inter/Union
    for thresh in np.arange(0.5, 1,  0.05):
        if IoU > thresh:
            JCC = JCC + (1 - IoU)
        else:
            JCC = JCC + 1
    JCC = JCC/10
    JCC = JCC.cpu().detach().tolist()
    return JCC 

def getHD95Acc(pred_mask: torch.Tensor, pred_sdm: torch.Tensor, gt_mask: torch.Tensor, gt_sdm: torch.Tensor):
    dP = torch.abs(torch.multiply(gt_sdm, pred_mask))
    dG = torch.abs(torch.multiply(pred_sdm, gt_mask))
    dPdG = torch.cat((dP, dG), dim=0)
    HD95 = torch.quantile(dPdG, 0.95)
    HD95 = HD95.cpu().detach().tolist()
    return HD95

def getAsdAcc(pred_mask: torch.Tensor, pred_sdm: torch.Tensor, gt_mask: torch.Tensor, gt_sdm: torch.Tensor):
    dP = torch.abs(torch.multiply(gt_sdm, pred_mask))
    dG = torch.abs(torch.multiply(pred_sdm, gt_mask))
    dPdG = torch.cat((dP, dG), dim=0)
    ASD = torch.mean(dPdG, 0.95)
    ASD = ASD.cpu().detach().tolist()
    return ASD

class Accuracy:
    def __init__(self, total, train_target: str = 'sdm', threshold: float = 0.5):
        self.T = threshold
        self.target = train_target
        self.total = total

        self.DSC = 0
        self.IoU = 0
        self.HD95 = 0
        self.ASD = 0

    def update(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_sdm: torch.Tensor):
        if self.target == 'sdm':
            pred_mask = torch.relu(torch.sign(torch.sigmoid(-1*logits)-self.T))
            pred_sdm = torch.tanh(logits)
        elif self.target == 'bin':
            pred_mask = torch.relu(torch.sign(torch.sigmoid(logits)-self.T))
            pred_sdm = SDF(pred_mask)

        self.DSC += getDiceAcc(pred_mask, gt_mask)
        self.IoU += getJaccardAcc(pred_mask, gt_mask)
        self.HD95 += getHD95Acc(pred_mask, pred_sdm, gt_mask, gt_sdm)
        self.ASD += getAsdAcc(pred_mask, pred_sdm, gt_mask, gt_sdm)

    def compute(self):
        self.DSC = self.DSC/self.total
        self.IoU = self.IoU/self.total
        self.HD95 = self.HD95/self.total
        self.ASD = self.ASD/self.total
        return (self.DSC, self.IoU, self.HD95, self.ASD)

    def reset(self):
        self.DSC = 0
        self.IoU = 0
        self.HD95 = 0
        self.ASD = 0



# def getSegAccuracy(pred, true):
#     predMask = pred.squeeze()
#     predMask = torch.relu(torch.sign(torch.sigmoid(predMask)-0.5))
#     trueMask = true.squeeze()
    
#     TP_SACRP = torch.sum(torch.multiply(trueMask, predMask))
#     TP_BACKG = torch.sum(torch.multiply(torch.ones_like(trueMask) - trueMask, torch.ones_like(predMask) - predMask))
#     FPFN = trueMask.nelement() + predMask.nelement()
#     DICE = (2*(TP_SACRP+TP_BACKG) + 1)/(FPFN + 1)
#     DICE = DICE.cpu().detach().tolist()
#     return DICE

# def getClsAccuracy(predCls, trueCls):
#     predCls = torch.nn.functional.softmax(predCls, dim=1)
#     predCls = torch.argmax(predCls, dim=1)
#     predCls = predCls.cpu().detach().numpy().reshape(-1,1)
#     predCls = OneHotEncoder().fit([[0],[1],[2]]).transform(predCls).toarray()

#     trueCls = trueCls.cpu().detach().numpy()

#     ACC = accuracy_score(trueCls, predCls)
#     return ACC

# def getAuxAccuracy(predMask, predCls, trueMask):
#     prob = torch.nn.functional.softmax(predCls, dim=1)
#     prob = prob.cpu().detach().numpy()
#     prob = np.round(prob, decimals=3)
#     lbl_pred = np.argmax(prob, axis=1)
    
#     for i in range(0, len(lbl_pred)):
#         if lbl_pred[i] == 0 and prob[i][lbl_pred[i]] > 0.5:
#             predMask[i] = torch.zeros(512, 512)
#         if lbl_pred[i] == 1 and prob[i][lbl_pred[i]] > 0.5:
#             predMask[i] = torch.ones(512, 512)
#         else:
#             pass

#     predMask = torch.relu(torch.sign(torch.sigmoid(predMask)-0.5))
    
#     RAND = torch.sum(torch.eq(trueMask, predMask))/(predMask.nelement())
#     RAND = RAND.cpu().detach().tolist()
    # RAND = 0
    # return RAND
    
    