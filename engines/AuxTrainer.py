import json
import numpy as np #!!remove later
from tqdm import tqdm
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from model.metrics import getClsAccuracy, getSegAccuracy, getAuxAccuracy

class AuxTrainer():
    def __init__(self, conf, model, lossFunc, optimizer, testLoader, trainLoader):
        self.conf = conf
        self.lossFunc = lossFunc
        self.optimizer = optimizer
        self.testLoader = testLoader
        self.trainLoader = trainLoader
        self.minLoss = 1
        
        if conf.GPU_COUNT > 1:
            self.model = model.to('cuda')
            self.model = DDP(self.model, device_ids=[self.conf.GPU_ID], find_unused_parameters=True)
        else:
            self.model = model.to(self.conf.GPU_ID)
            
        self.H = {}
        self.H["train_loss"] = []
        self.H["test_loss"] = []
        self.H["cls_test_acc"] = []
        self.H["cls_train_acc"] = []
        self.H["seg_test_acc"] = []
        self.H["seg_train_acc"] = []
        self.H["aux_test_acc"] = []
        self.H["aux_train_acc"] = []

    def _save_model(self):
        with open(self.conf.LOG_PATH, 'w') as tts:
            json.dump(self.H, tts)
       
        if self.H[self.conf.SAVE_TRIG][-1] < self.minLoss:
            self.minLoss = self.H[self.conf.SAVE_TRIG][-1]
            
            best = {
                'INFO':{
                    'MODE': self.conf.TRAIN_MODE, 
                    'DATASET': self.conf.DSET, 
                    'FREEZING': self.conf.TO_FREEZE, 
                    'LOSS': self.conf.LOSS,
                    'ACC_METRIC': self.conf.SAVE_TRIG
                },
                'METRICS':{
                    'epoch' : self.epoch,
                    'train_loss': self.H['train_loss'][-1],
                    'test_loss': self.H['test_loss'][-1],
                    'cls_test_acc': self.H['cls_test_acc'][-1],
                    'cls_train_acc': self.H['cls_train_acc'][-1],
                    'seg_test_acc': self.H['seg_test_acc'][-1],
                    'seg_train_acc': self.H['seg_train_acc'][-1],
                    'aux_test_acc': self.H['aux_test_acc'][-1],
                    'aux_train_acc': self.H['aux_train_acc'][-1]
                },
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            
            torch.save(best, self.conf.MDL_PATH)
    
    def _run_epoch(self, loader, mode):
        self.model.train() if mode == "train" else self.model.eval()
      
        runningLoss = 0
        totalSegAcc = 0
        totalClsAcc = 0
        totalAuxAcc = 0
        with torch.no_grad() if mode == "test" else nullcontext():
            for source, seg_true, cls_true in loader:
                source, seg_true, cls_true = source.to(self.conf.GPU_ID), seg_true.to(self.conf.GPU_ID), cls_true.to(self.conf.GPU_ID)
                cls_pred, seg_pred = self.model(source)
                loss = self.lossFunc(seg_pred , seg_true, cls_pred, cls_true, self.conf.LOSS)
        
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                runningLoss += loss
                totalSegAcc += getSegAccuracy(seg_pred, seg_true)
                totalClsAcc += getClsAccuracy(cls_pred, cls_true)
                totalAuxAcc += getAuxAccuracy(seg_pred, cls_pred, seg_true)

        avgLoss = runningLoss/loader.__len__() 
        avgSegAcc = totalSegAcc/loader.__len__()
        avgClsAcc = totalClsAcc/loader.__len__()
        avgAuxAcc = totalAuxAcc/loader.__len__()
        
        return avgLoss, avgSegAcc, avgClsAcc, avgAuxAcc
    
    def train(self):
        for epoch in tqdm(range(self.conf.NUM_EPOCHS)):
            self.epoch = epoch
            
            nullcontext() if self.conf.GPU_COUNT < 2 else self.trainLoader.sampler.set_epoch(epoch)
            
            trainLoss, trainSegAcc, trainClsAcc, trainAuxAcc = self._run_epoch(self.trainLoader, "train")
            testLoss, testSegAcc, testClsAcc, testAuxAcc = self._run_epoch(self.testLoader, "test")
            
            if self.conf.GPU_ID == 0:
                self.H["train_loss"].append(trainLoss.item())
                self.H["test_loss"].append(testLoss.item())
                self.H["cls_test_acc"].append(testClsAcc)
                self.H["cls_train_acc"].append(trainClsAcc)
                self.H["seg_test_acc"].append(testSegAcc)
                self.H["seg_train_acc"].append(trainSegAcc)
                self.H["aux_test_acc"].append(testAuxAcc)
                self.H["aux_train_acc"].append(trainAuxAcc)

                self._save_model()

                print("[INFO] EPOCH: {}/{}".format(epoch + 1, self.conf.NUM_EPOCHS))
                print("Train loss: {:.6f}, Test loss: {:.4f}".format(trainLoss, testLoss))
                print("Train Seg acc: {:.4f}, Test Seg acc: {:.4f}".format(trainSegAcc, testSegAcc))
                print("Train Cls acc: {:.4f}, Test Cls acc: {:.4f}".format(trainClsAcc, testClsAcc))
                print("Train Aux acc: {:.4f}, Test Aux acc: {:.4f}".format(trainAuxAcc, testAuxAcc))


    