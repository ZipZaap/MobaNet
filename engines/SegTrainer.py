import json
import torch
from contextlib import nullcontext
from model.metrics import Accuracy
from model.loss import Loss
from utils.util import timer
from configs import CONF

class SegTrainer():
    def __init__(self, gpu_id, model, optimizer, testLoader, trainLoader):
        self.gpu_id = gpu_id
        self.model = model
        self.optimizer = optimizer
        self.testLoader = testLoader
        self.trainLoader = trainLoader
        self.minLoss = 1
        
    # def _save_model(self):
    #     with open(CONF.LOG_PATH, 'w') as tts:
    #         json.dump(self.H, tts)
       
    #     if self.H[CONF.SAVE_TRIG][-1] < self.minLoss:
    #         self.minLoss = self.H[CONF.SAVE_TRIG][-1]
            
    #         best = {
    #             'INFO':{
    #                 'MODE': CONF.TRAIN_MODE, 
    #                 'DATASET': CONF.DSET, 
    #                 'LOSS': CONF.LOSS,
    #                 'ACC_METRIC': CONF.SAVE_TRIG
    #             },
    #             'METRICS':{
    #                 'epoch' : self.epoch,
    #                 'train_loss': self.H['train_loss'][-1],
    #                 'test_loss': self.H['test_loss'][-1],
    #                 'seg_test_acc': self.H['seg_test_acc'][-1],
    #                 'seg_train_acc': self.H['seg_train_acc'][-1],
    #             },
    #             'state_dict': self.model.state_dict(),
    #             'optimizer': self.optimizer.state_dict()
    #         }
            
    #         torch.save(best, CONF.MDL_PATH)
    
    def _run_epoch(self, loader, mode):
        self.model.train() if mode == "train" else self.model.eval()
      
        acc = Accuracy('bin')
        lss = Loss('bin')
        with torch.no_grad() if mode == "test" else nullcontext():
            for source, gt_mask, gt_sdm in loader:
                pred_mask = self.model(source)
                loss = lss.update(pred_mask, gt_mask, gt_sdm)
                acc.update(pred_mask, gt_mask, gt_sdm)
        
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        avgLoss = lss.compute_avg(self.gpu_id, loader.__len__())
        avgSegAcc = acc.compute_avg(self.gpu_id, loader.__len__())
    
        return avgLoss, avgSegAcc
    
   
    def train(self):
        for epoch in range(CONF.NUM_EPOCHS):
            ts = timer(epoch)
            
            self.trainLoader.sampler.set_epoch(epoch) if CONF.GPU_COUNT > 1 else nullcontext()
            
            trainLoss, trainSegAcc = self._run_epoch(self.trainLoader, "train")
            testLoss, testSegAcc = self._run_epoch(self.testLoader, "test")
    
            dt, etc = timer(epoch, ts)
            
            if self.gpu_id == 0:

                print(f"[GPU{self.gpu_id}] Epoch {epoch + 1}/{CONF.NUM_EPOCHS} | LR: {self.optimizer.defaults['lr']} | Train loss: {trainLoss:.4f} | Test loss: {testLoss:.4f} | Train Acc: {trainSegAcc:.4f} | Test Acc: {testSegAcc:.4f} | dt: {dt}s :: {etc}m")
                print("-------------------------------------------------------------------------------------------------------------------------------------")