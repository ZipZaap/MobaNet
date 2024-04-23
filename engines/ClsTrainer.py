import json
from tqdm import tqdm
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from model.metrics import getClsAccuracy

class ClsTrainer():
    def __init__(self, conf, model, lossFunc, optimizer, testLoader, trainLoader):
        self.conf = conf
        self.lossFunc = lossFunc
        self.optimizer = optimizer
        self.testLoader = testLoader
        self.trainLoader = trainLoader
        self.minLoss = 1
        
        if conf.GPU_COUNT > 1:
            self.model = model.to('cuda')
            self.model = DDP(self.model, device_ids=[self.conf.GPU_ID])
        else:
            self.model = model.to(self.conf.GPU_ID)
        
        self.H = {}
        self.H["train_loss"] = []
        self.H["test_loss"] = []
        self.H["cls_test_acc"] = []
        self.H["cls_train_acc"] = []

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
                    'cls_test_acc': self.H['cls_test_acc'][-1],
                    'cls_train_acc': self.H['cls_train_acc'][-1],
                },
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }

            torch.save(best, self.conf.MDL_PATH)
    
    def _run_epoch(self, loader, mode):
        self.model.train() if mode == "train" else self.model.eval()
      
        runningLoss = 0
        totalClsAcc = 0
        with torch.no_grad() if mode == "test" else nullcontext():
            for source, cls_true in loader:
                source, cls_true = source.to(self.conf.GPU_ID), cls_true.to(self.conf.GPU_ID) 
                cls_pred = self.model(source)
                loss = self.lossFunc(cls_pred, cls_true, self.conf.LOSS)
        
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
        for epoch in tqdm(range(self.conf.NUM_EPOCHS)):
            self.epoch = epoch

            nullcontext() if self.conf.GPU_COUNT < 2 else self.trainLoader.sampler.set_epoch(epoch)
            
            trainLoss, trainClsAcc = self._run_epoch(self.trainLoader, "train")
            testLoss, testClsAcc = self._run_epoch(self.testLoader, "test")
            
            if self.conf.GPU_ID == 0:
                self.H["train_loss"].append(trainLoss.item())
                self.H["test_loss"].append(testLoss.item())
                self.H["cls_test_acc"].append(testClsAcc)
                self.H["cls_train_acc"].append(trainClsAcc)

                self._save_model()

                print("[INFO] EPOCH: {}/{}".format(epoch + 1, self.conf.NUM_EPOCHS))
                print("Train loss: {:.6f}, Test loss: {:.4f}".format(trainLoss, testLoss))
                print("Train Cls acc: {:.4f}, Test Cls acc: {:.4f}".format(trainClsAcc, testClsAcc))