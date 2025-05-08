import wandb
import torch
from contextlib import nullcontext

from model.metrics import Accuracy
from model.loss import Loss
<<<<<<< Updated upstream
from utils.util import timer
from configs import CONF

class SegTrainer():
    def __init__(self, gpu_id, model, optimizer, loaders):
        self.gpu_id = gpu_id
=======
from configs.config_parser import CONF

class SegTrainer():
    def __init__(self, model, optimizer, lr_scheduler, loaders, logger):
>>>>>>> Stashed changes
        self.model = model
        self.optimizer = optimizer
        self.trainLoader, self.testLoader = loaders
<<<<<<< Updated upstream
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
=======
        self.logger = logger
        self.device = next(model.parameters()).device
        self.loss = Loss()
        self.accu = Accuracy()

    def _run_epoch(self, loader, mode):

        train = (mode == 'train')
        warmup = (mode == 'warmup')
        self.model.train() if (train or warmup) else self.model.eval()
    
        self.accu.reset()
        self.loss.reset()
        with torch.set_grad_enabled(train or warmup):
            for batch in loader:
                batch = {k:v.to(self.device) for k,v in batch.items()}

                logits = self.model(batch['image'])
                self.loss.update(logits, batch)
                if not warmup:
                    self.accu.update(logits, batch) 

                if (train or warmup):
                    self.optimizer.zero_grad()
                    self.loss.backprop()
                    self.optimizer.step()

        if warmup:
            self.lr_scheduler.step() 
        else:
            avgLoss = self.loss.compute_avg(loader.__len__())
            avgAccu = self.accu.compute_avg(loader.__len__())
            return avgLoss, avgAccu
            
>>>>>>> Stashed changes
    
   
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

<<<<<<< Updated upstream
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
=======
            self.logger.info()


# import torch
# from contextlib import nullcontext

# from model.metrics import Accuracy
# from model.loss import Loss
# from configs.config_parser import CONF

# class SegTrainer():
#     def __init__(self, model, optimizer, lr_scheduler, loaders, logger):
#         self.model = model
#         self.optimizer = optimizer
#         self.lr_scheduler = lr_scheduler
#         self.trainLoader, self.testLoader = loaders
#         self.logger = logger
#         self.device = next(model.parameters()).device
#         self.lss = Loss()
#         self.acc = Accuracy()

#     def _run_epoch(self, loader, mode):

#         train = (mode == 'train')
#         warmup = (mode == 'warmup')
#         self.model.train() if (train or warmup) else self.model.eval()
      
#         self.acc.reset()
#         self.lss.reset()
#         with torch.set_grad_enabled(train or warmup):
#             for source, gt_mask, gt_sdm, gt_cls in loader:
#                 source, gt_mask, gt_sdm, gt_cls = source.to(self.device), gt_mask.to(self.device), gt_sdm.to(self.device), gt_cls.to(self.device)

#                 seg_logits, cls_logits = self.model(source)
#                 self.lss.update(seg_logits, cls_logits, gt_mask, gt_sdm, gt_cls)
#                 if not warmup:
#                     self.acc.update(seg_logits, cls_logits, gt_mask, gt_sdm, gt_cls) 

#                 if (train or warmup):
#                     self.optimizer.zero_grad()
#                     self.lss.backprop()
#                     self.optimizer.step()

#         if not warmup:
#             avgLoss = self.lss.compute_avg(loader.__len__())
#             avgSegAcc = self.acc.compute_avg()
#             return avgLoss, avgSegAcc
#         else:
#             self.lr_scheduler.step() 
    
    
#     def train(self):
#         for epoch in range(-CONF.WARMUP_EPOCHS + 1, CONF.TRAIN_EPOCHS + 1):
#             self.trainLoader.sampler.set_epoch(epoch) if CONF.NUM_GPU > 1 else nullcontext()
#             self.logger.set_epoch(epoch)
            
#             if epoch <= 0:
#                 self._run_epoch(self.trainLoader, "warmup")
#             else:
#                 self.logger.start_timer()
#                 trainLoss, trainSegAcc = self._run_epoch(self.trainLoader, "train")
#                 testLoss, testSegAcc = self._run_epoch(self.testLoader, "test")
#                 self.logger.reset_timer()

#                 self.logger.update_metrics((trainLoss, testLoss, trainSegAcc, testSegAcc))
#                 self.logger.log_metrics()
#                 self.logger.save_model(self.model)

#             self.logger.info()
>>>>>>> Stashed changes
