import wandb
import torch
from contextlib import nullcontext

from model.metrics import Accuracy
from model.loss import Loss
from utils.util import timer
from configs import CONF

class SegTrainer():
    def __init__(self, gpu_id, model, optimizer, loaders):
        self.gpu_id = gpu_id
        self.model = model
        self.optimizer = optimizer
        self.trainLoader, self.testLoader = loaders
        self.minLoss = 1
    
    def _run_epoch(self, epoch, loader, mode):
        self.model.train() if mode == "train" else self.model.eval()
      
        acc = Accuracy()
        lss = Loss()
        with torch.no_grad() if mode == "test" else nullcontext():
            for source, gt_mask, gt_sdm in loader:
                source, gt_mask, gt_sdm = source.to(self.gpu_id), gt_mask.to(self.gpu_id), gt_sdm.to(self.gpu_id)

                pred_mask = self.model(source)
                loss = lss.update(epoch, pred_mask, gt_mask, gt_sdm)
                acc.update(pred_mask, gt_mask, gt_sdm)
        
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        avgLoss = lss.compute_avg(loader.__len__())
        avgSegAcc = acc.compute_avg(loader.__len__())
    
        return avgLoss, avgSegAcc
    
   
    def train(self):
        wandb.init(project=CONF.MODEL,
            name=CONF.RUN_ID, 
            config=CONF)

        for epoch in range(CONF.NUM_EPOCHS):
  
            ts = timer(epoch)
            
            self.trainLoader.sampler.set_epoch(epoch) if CONF.NUM_GPU > 1 else nullcontext()

            trainLoss, trainSegAcc = self._run_epoch(epoch, self.trainLoader, "train")
            testLoss, testSegAcc = self._run_epoch(epoch, self.testLoader, "test")
    
            dt, etc = timer(epoch, ts)
            
            if self.gpu_id == 0 or CONF.NUM_GPU <=1:

                print(f"[{self.gpu_id}] Epoch {epoch + 1}/{CONF.NUM_EPOCHS} > ETC: {etc}m ({dt}s / epoch)")
                print(f"Train -> loss: {trainLoss:.4f} | DSC: {trainSegAcc['dsc']:.4f} | JCC: {trainSegAcc['iou']:.4f} | HD95: {trainSegAcc['hd95']:.4f} | ASD: {trainSegAcc['asd']:.4f} |")
                print(f"Test  -> loss: {testLoss:.4f} | DSC: {testSegAcc['dsc']:.4f} | JCC: {testSegAcc['iou']:.4f} | HD95: {testSegAcc['hd95']:.4f} | ASD: {testSegAcc['asd']:.4f} |")
                print("--------------------------------------------------------------------------------")

                wandb.log({'loss/train': trainLoss, 'loss/test': testLoss}, step = epoch+1)
                wandb.log({'metrics/DSC-train': trainSegAcc['dsc'], 'metrics/DSC-test': testSegAcc['dsc']}, step = epoch+1)
                wandb.log({'metrics/JCC-train': trainSegAcc['iou'], 'metrics/JCC-test': testSegAcc['iou']}, step = epoch+1)
                wandb.log({'metrics/HD95-train': trainSegAcc['hd95'], 'metrics/HD95-test': testSegAcc['hd95']}, step = epoch+1)
                wandb.log({'metrics/ASD-train': trainSegAcc['asd'], 'metrics/ASD-test': testSegAcc['asd']}, step = epoch+1)
        
        wandb.finish()