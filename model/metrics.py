from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import torch

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

# def getSegAccuracy(predMask, trueMask):
#     predMask = torch.relu(torch.sign(torch.sigmoid(predMask)-0.5))
    
#     RAND = torch.sum(torch.eq(trueMask, predMask))/(predMask.nelement())
#     RAND = RAND.cpu().detach().tolist()
#     return RAND

def getSegAccuracy(pred, true):
    predMask = pred.squeeze()
    predMask = torch.relu(torch.sign(torch.sigmoid(predMask)-0.5))
    trueMask = true.squeeze()

    TP_SACRP = torch.sum(torch.multiply(trueMask, predMask)) + 1
    TPFPFN = torch.sum(trueMask) + torch.sum(predMask) + 1
    DICE = (2*TP_SACRP)/TPFPFN
    DICE = DICE.cpu().detach().tolist()

    return DICE


def getClsAccuracy(predCls, trueCls):
    predCls = torch.nn.functional.softmax(predCls, dim=1)
    predCls = torch.argmax(predCls, dim=1)
    predCls = predCls.cpu().detach().numpy().reshape(-1,1)
    predCls = OneHotEncoder().fit([[0],[1],[2]]).transform(predCls).toarray()

    trueCls = trueCls.cpu().detach().numpy()

    ACC = accuracy_score(trueCls, predCls)
    return ACC

def getAuxAccuracy(predMask, predCls, trueMask):
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
    RAND = 0
    return RAND
    
    