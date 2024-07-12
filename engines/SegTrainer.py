import json
import torch
from contextlib import nullcontext
from model.metrics import Accuracy
from model.loss import Loss
from utils.util import timer
from utils.util import SDF
from configs import CONF

class SegTrainer():
    def __init__(self, gpu_id, model, optimizer, loaders):
        self.gpu_id = gpu_id
        self.model = model
        self.optimizer = optimizer
        self.trainLoader, self.testLoader = loaders
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
        lss = Loss(CONF.LOSS)
        with torch.no_grad() if mode == "test" else nullcontext():
            for source, gt_mask, gt_sdm in loader:
                source, gt_mask, gt_sdm = source.to(self.gpu_id), gt_mask.to(self.gpu_id), gt_sdm.to(self.gpu_id)

                pred_mask = self.model(source)
                loss = lss.update(pred_mask, gt_mask, gt_sdm)
                acc.update(pred_mask, gt_mask, gt_sdm)
        
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        avgLoss = lss.compute_avg(loader.__len__())
        avgSegAcc = acc.compute_avg(loader.__len__())
    
        return avgLoss, avgSegAcc
    
   
    def train(self):
        for epoch in range(CONF.NUM_EPOCHS):
            ts = timer(epoch)
            
            self.trainLoader.sampler.set_epoch(epoch) if CONF.NUM_GPU > 1 else nullcontext()

            trainLoss, trainSegAcc = self._run_epoch(self.trainLoader, "train")
            testLoss, testSegAcc = self._run_epoch(self.testLoader, "test")
    
            dt, etc = timer(epoch, ts)
            
            if self.gpu_id == 0 or CONF.NUM_GPU <=1:

                print(f"[{self.gpu_id}] Epoch {epoch + 1}/{CONF.NUM_EPOCHS} > ETC: {etc}m ({dt}s / epoch)")
                print(f"Train -> loss: {trainLoss:.4f} | DSC: {trainSegAcc['dsc']:.4f} | JCC: {trainSegAcc['iou']:.4f} | HD95: {trainSegAcc['hd95']:.4f} | ASD: {trainSegAcc['asd']:.4f} |")
                print(f"Test  -> loss: {testLoss:.4f} | DSC: {testSegAcc['dsc']:.4f} | JCC: {testSegAcc['iou']:.4f} | HD95: {testSegAcc['hd95']:.4f} | ASD: {testSegAcc['asd']:.4f} |")
                print("--------------------------------------------------------------------------------")