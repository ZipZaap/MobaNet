import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.loss import Loss
from utils.loggers import Logger
from model.metrics import Accuracy
from configs.cfgparser  import Config

class SegTrainer():
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: Optimizer, 
                 lr_scheduler: LRScheduler, 
                 loaders: tuple[DataLoader, DataLoader],
                 logger: Logger,
                 cfg: Config):
        """
        Initializes the SegTrainer. This class is responsible for training and evaluating the model.
        It handles the training loop, validation, and logging of metrics.

        Args
        ----
            model : torch.nn.Module
                Model to be trained.

            optimizer : Optimizer
                Optimizer for the model.

            lr_scheduler : LRScheduler
                Learning rate scheduler.

            loaders : tuple[DataLoader, DataLoader]
                Tuple of train and test loaders.

            logger : Logger
                Logger for logging metrics and saving model checkpoints.

            cfg : Config
                Configuration object that contains the following attributes:
                - `.DEVICE` (str): Device to run the model on.
                - `.TRAIN_EPOCHS` (int): Number of training epochs.
                - `.WARMUP_EPOCHS` (int): Number of warmup epochs.
                - `.EVAL_INTERVAL` (int): Evaluation interval in epochs.
        """

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.trainLoader, self.testLoader = loaders
        self.logger = logger

        self.device: str = cfg.DEVICE
        self.train_epochs: int = cfg.TRAIN_EPOCHS
        self.warmup_epochs: int = cfg.WARMUP_EPOCHS
        self.eval_interval: int = cfg.EVAL_INTERVAL

        self.loss: Loss = Loss(cfg)
        self.accu: Accuracy = Accuracy(cfg)
            
    def _warmup_epoch(self, loader: DataLoader):
        """
        Warmup training for the model by gradually increasing the learning rate.

        Args
        ----
            loader : DataLoader
                DataLoader for the training data.
        """
        
        self.loss.reset()
        self.model.train() 
        with torch.set_grad_enabled(True):

            for batch in loader:
                batch = {k:v.to(self.device) for k,v in batch.items()}
                logits = self.model(batch['image'])

                self.optimizer.zero_grad()
                self.loss.update(logits, batch)
                self.loss.backprop()
                self.optimizer.step()

        self.lr_scheduler.step()


    def _learn_epoch(self,
                     loader: DataLoader,
                     train: bool,
                     ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Runs a single epoch of training or evaluation

        Args
        ----
            loader : DataLoader
                DataLoader for the training or evaluation data.

            train : bool
                If True, runs training; if False, runs evaluation.
            
        Returns
        -------
            avgLoss : dict[str, float]
                Average loss for the epoch.
                
            avgAccu : dict[str, float]
                Average accuracy for the epoch.
        """
        
        self.model.train() if train else self.model.eval()

        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = {k:v.to(self.device) for k,v in batch.items()}
                logits = self.model(batch['image'])

                self.accu.update(logits, batch) 
                self.loss.update(logits, batch)

                if train:
                    self.optimizer.zero_grad()
                    self.loss.backprop()
                    self.optimizer.step()

        avgLoss = self.loss.compute_avg(loader.__len__())
        self.loss.reset()

        avgAccu = self.accu.compute_avg(loader.__len__())
        self.accu.reset()
        
        return avgLoss, avgAccu


    def train(self):
        """
        Main training loop; iterates over epochs and runs warmup, training & evaluation.
        """
        
        self.logger.init_run()
        for epoch in range(-self.warmup_epochs + 1, self.train_epochs + 1):
            
            self.logger.set_epoch(epoch)
            if isinstance(self.trainLoader.sampler, DistributedSampler):
                self.trainLoader.sampler.set_epoch(epoch)
            
            if epoch <= 0:
                self._warmup_epoch(self.trainLoader)
            else:
                self.logger.start_timer()
                
                trainLoss, trainAccu = self._learn_epoch(self.trainLoader, train=True)
                if epoch % self.eval_interval == 0 or epoch == self.train_epochs:
                    testLoss, testAccu = self._learn_epoch(self.testLoader, train=False)
                    self.logger.update((trainLoss, testLoss, trainAccu, testAccu), self.model)
                    self.logger.log_metrics()

                self.logger.reset_timer()

            self.logger.info()

        self.logger.end_run()