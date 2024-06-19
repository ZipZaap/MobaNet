import json
import torch
from contextlib import nullcontext
from model.metrics import getClsAccuracy
from utils.util import gather_metrics, timer
from configs import CONF

class ClsTrainer():
    def __init__(self, gpu_id, model, lossFunc, optimizer, testLoader, trainLoader):
        self.gpu_id = gpu_id
        self.model = model
        self.lossFunc = lossFunc
        self.optimizer = optimizer
        self.testLoader = testLoader
        self.trainLoader = trainLoader
        self.minLoss = 1
        
        self.H = {}
        self.H["train_loss"] = []
        self.H["test_loss"] = []
        self.H["cls_test_acc"] = []
        self.H["cls_train_acc"] = []

    def _save_model(self):
        with open(CONF.LOG_PATH, 'w') as tts:
            json.dump(self.H, tts)

        if self.H[CONF.SAVE_TRIG][-1] < self.minLoss:
            self.minLoss = self.H[CONF.SAVE_TRIG][-1]

            best = {
                'INFO':{
                    'MODE': CONF.TRAIN_MODE, 
                    'DATASET': CONF.DSET, 
                    'LOSS': CONF.LOSS,
                    'ACC_METRIC': CONF.SAVE_TRIG
                },
                'METRICS':{
                    'epoch' : self.epoch,
                    'train_loss': self.H['train_loss'][-1],
                    'test_loss': self.H['test_loss'][-1],
                    'cls_test_acc': self.H['cls_test_acc'][-1],
                    'cls_train_acc': self.H['cls_train_acc'][-1],
                },
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }

            torch.save(best, CONF.MDL_PATH)
    
    def _run_epoch(self, loader, mode):
        self.model.train() if mode == "train" else self.model.eval()
      
        runningLoss = 0
        totalClsAcc = 0
        with torch.no_grad() if mode == "test" else nullcontext():
            for source, cls_true in loader:
                source, cls_true = source.to(self.gpu_id), cls_true.to(self.gpu_id) 
                cls_pred = self.model(source)
                loss = self.lossFunc(cls_pred, cls_true, CONF.LOSS)
        
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                runningLoss += loss
                totalClsAcc += getClsAccuracy(cls_pred, cls_true)

        avgLoss = runningLoss/loader.__len__() 
        avgClsAcc = totalClsAcc/loader.__len__()
        
        return avgLoss, avgClsAcc
    
    
    def train(self):
        for epoch in range(CONF.NUM_EPOCHS):
            self.epoch = epoch
            ts = timer(epoch)
            
            nullcontext() if CONF.GPU_COUNT < 2 else self.trainLoader.sampler.set_epoch(epoch)
            
            trainLoss, trainClsAcc = self._run_epoch(self.trainLoader, "train")
            testLoss, testClsAcc = self._run_epoch(self.testLoader, "test")
    
            dt, etc = timer(epoch, ts)
            
            if CONF.GPU_COUNT > 1:
                testClsAcc, trainClsAcc, testLoss, trainLoss  = gather_metrics(self.gpu_id, [testClsAcc, trainClsAcc, testLoss, trainLoss]) 

            if self.gpu_id == 0:
                self.H["train_loss"].append(trainLoss)
                self.H["test_loss"].append(testLoss)
                self.H["cls_test_acc"].append(testClsAcc)
                self.H["cls_train_acc"].append(trainClsAcc)

                self._save_model()

                print(f"[GPU{self.gpu_id}] Epoch {epoch + 1}/{CONF.NUM_EPOCHS} | LR: {self.optimizer.defaults['lr']} | Train loss: {trainLoss:.4f} | Test loss: {testLoss:.4f} | Train Acc: {trainClsAcc:.4f} | Test Acc: {testClsAcc:.4f} | dt: {dt}s :: {etc}m")
                print("-------------------------------------------------------------------------------------------------------------------------------------")