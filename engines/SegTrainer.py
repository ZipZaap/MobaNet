import json
import time
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model.metrics import getSegAccuracy

class SegTrainer():
    def __init__(self, conf, model, lossFunc, optimizer, testLoader, trainLoader):
        self.conf = conf
        self.lossFunc = lossFunc
        self.optimizer = optimizer
        self.testLoader = testLoader
        self.trainLoader = trainLoader
        self.minLoss = 1
        self.ts = None
        
        if conf.GPU_COUNT > 1:
            self.model = model.to('cuda')
            self.model = DDP(self.model, device_ids=[self.conf.GPU_ID], find_unused_parameters=False)
        else:
            self.model = model.to(self.conf.GPU_ID)
            
        self.H = {}
        self.H["train_loss"] = []
        self.H["test_loss"] = []
        self.H["seg_test_acc"] = []
        self.H["seg_train_acc"] = []


    def _save_model(self):
        with open(self.conf.LOG_PATH, 'w') as tts:
            json.dump(self.H, tts)
       
        if self.H[self.conf.SAVE_TRIG][-1] < self.minLoss:
            self.minLoss = self.H[self.conf.SAVE_TRIG][-1]
            
            best = {
                'INFO':{
                    'MODE': self.conf.TRAIN_MODE, 
                    'DATASET': self.conf.DSET, 
                    'LOSS': self.conf.LOSS,
                    'ACC_METRIC': self.conf.SAVE_TRIG
                },
                'METRICS':{
                    'epoch' : self.epoch,
                    'train_loss': self.H['train_loss'][-1],
                    'test_loss': self.H['test_loss'][-1],
                    'seg_test_acc': self.H['seg_test_acc'][-1],
                    'seg_train_acc': self.H['seg_train_acc'][-1],
                },
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            
            torch.save(best, self.conf.MDL_PATH)
    
    def _run_epoch(self, loader, mode):
        self.model.train() if mode == "train" else self.model.eval()
      
        runningLoss = 0
        totalSegAcc = 0
        with torch.no_grad() if mode == "test" else nullcontext():
            for source, seg_true in loader:
                source, seg_true = source.to(self.conf.GPU_ID), seg_true.to(self.conf.GPU_ID)
                seg_pred = self.model(source)
                loss = self.lossFunc(seg_pred, seg_true, self.conf.LOSS)
        
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                runningLoss += loss
                totalSegAcc += getSegAccuracy(seg_pred, seg_true)

        avgLoss = runningLoss.item()/loader.__len__() 
        avgSegAcc = totalSegAcc/loader.__len__()
        
        return avgLoss, avgSegAcc
    
    def _gather_metrics(self, metrics: list) -> list:
        output = []
        for metric in metrics:
            AvgAccLst = [torch.zeros_like(torch.tensor(metric)).to(self.conf.GPU_ID) for _ in range(self.conf.GPU_COUNT)]
            dist.all_gather(AvgAccLst, torch.tensor(metric).to(self.conf.GPU_ID))
            metric = np.round(torch.mean(torch.stack(AvgAccLst)).item(), 4)
            output.append(metric)
        return output
    
    def _timer(self):
        if self.ts is not None:
            te = time.time()
            dt = np.round(te - self.ts)
            etc = np.round((self.conf.NUM_EPOCHS - self.epoch - 1)*dt/60)
            self.ts = None
            return dt, etc
        else:
            self.ts = time.time()
        
    def train(self):
        for epoch in range(self.conf.NUM_EPOCHS):
            self.epoch = epoch
            self._timer()
            
            nullcontext() if self.conf.GPU_COUNT < 2 else self.trainLoader.sampler.set_epoch(epoch)
            
            trainLoss, trainSegAcc = self._run_epoch(self.trainLoader, "train")
            testLoss, testSegAcc = self._run_epoch(self.testLoader, "test")
    
            dt, etc = self._timer()
            
            if self.conf.GPU_COUNT > 1:
                testSegAcc, trainSegAcc, testLoss, trainLoss  = self._gather_metrics([testSegAcc, trainSegAcc, testLoss, trainLoss]) 

            if self.conf.GPU_ID == 0:
                self.H["train_loss"].append(trainLoss)
                self.H["test_loss"].append(testLoss)
                self.H["seg_test_acc"].append(testSegAcc)
                self.H["seg_train_acc"].append(trainSegAcc)

                self._save_model()

                print(f"[GPU{self.conf.GPU_ID}] Epoch {epoch + 1}/{self.conf.NUM_EPOCHS} | LR: {self.optimizer.defaults['lr']} | Train loss: {trainLoss:.4f} | Test loss: {testLoss:.4f} | Train Acc: {trainSegAcc:.4f} | Test Acc: {testSegAcc:.4f} | dt: {dt}s :: {etc}m")
                print("-------------------------------------------------------------------------------------------------------------------------------------")
