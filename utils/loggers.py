import os

import wandb
import time
import json
import torch
import numpy as np
from configs.config_parser import CONF

class Logger:
    def __init__(self, rank: int | str) -> None:
        
        self.rank0 = (rank == 0 or CONF.NUM_GPU <= 1)
        self.log_wandb = (CONF.LOG_WANDB and self.rank0)
        self.log_local = (CONF.LOG_LOCAL and self.rank0)

        if self.log_wandb:
            wandb.login()

    def init_run(self):
        if self.log_wandb:
            wandb.init(project = CONF.EXP_ID,
                       name = CONF.RUN_ID,
                       config = CONF.__dict__)
        
        if self.log_local:
            self.run_summary = {}

    def end_run(self):
        if self.log_wandb:
            wandb.finish()

        if self.log_local:
            with CONF.LOG_JSON.open('w') as f:
                json.dump(self.run_summary, f)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def update_metrics(self, data: tuple[dict, dict, dict, dict]):
        if self.rank0:
            self.trainLoss, self.testLoss, self.trainAcc, self.testAcc = data
            self.losses = {'loss/train': self.trainLoss['loss'], 'loss/test': self.testLoss['loss'], 'loss/diff': np.abs(self.trainLoss['loss'] - self.testLoss['loss'])}
            self.metrics = {f'metrics-train/{key}': value for key, value in self.trainAcc.items()} | {f'metrics-test/{key}': value for key, value in self.testAcc.items()}
            self.log_batch = self.losses | self.metrics

    def log_metrics(self):
        if self.log_wandb:
            wandb.log(self.log_batch, step = self.epoch)
        if self.log_local:
            for key, value in self.log_batch.items():
                self.run_summary.setdefault(key, []).append(value)

    def info(self):
        if self.rank0:
            if self.epoch == 1 - CONF.WARMUP_EPOCHS:
                print(f'Warming up for {CONF.WARMUP_EPOCHS} epochs ...')
            if self.epoch <= 0:
                print(f"[WARMUP] Epoch {self.epoch + CONF.WARMUP_EPOCHS}/{CONF.WARMUP_EPOCHS}")
            if self.epoch == 0:
                print(f'\nTraining for {CONF.TRAIN_EPOCHS} epochs ...')
            if self.epoch > 0:
                metrics = self.trainAcc.keys()
                train_metrics_str = " | ".join([f"{metric}: {self.trainAcc[metric]:.4f}" for metric in metrics])
                test_metrics_str = " | ".join([f"{metric}: {self.testAcc[metric]:.4f}" for metric in metrics])

                print(f"[DEV: {self.rank}] Epoch {self.epoch}/{CONF.TRAIN_EPOCHS} > ETC: {self.etc}m ({self.dt}s / epoch)")
                print(f"Train -> loss: {self.losses['loss/train']:.4f} | {train_metrics_str} |")
                print(f"Test  -> loss: {self.losses['loss/test']:.4f} | {test_metrics_str} |")
                print('-'*(len(test_metrics_str)+25) + '+')

    def start_timer(self):
        if self.rank0:
            self.ts = time.time()

    def reset_timer(self):
        if self.rank0:
            te = time.time()
            self.dt = np.round(te - self.ts)
            self.etc = np.round((CONF.TRAIN_EPOCHS - self.epoch)*self.dt/60)

    # def save_model(self, model, save = CONF.SAVE_MODEL):
    #     if self.master_proc and save:
    #         if self.testAcc[CONF.SAVE_METRIC] >= self.svmetric:
    #             self.svmetric = self.testAcc[CONF.SAVE_METRIC]
    #             self.metrics['epoch'] = self.epoch

    #             if CONF.NUM_GPU > 1:
    #                 torch.save(model.module.state_dict(), self.mdl_file)
    #             else:
    #                 torch.save(model.state_dict(), self.mdl_file)

    #             with open(self.best_logfile, 'w') as f:
    #                 json.dump(self.metrics, f)