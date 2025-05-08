# import os
# import yaml
# import torch
# from pathlib import Path
# from configs.validator import Validator

# class Config():
#     def __init__(self, cpath):

#         with open(cpath) as file:
#             cfg = yaml.load(file, Loader=yaml.FullLoader)
        
#         for key, value in cfg.items():
#             setattr(self, key, value['value'])
        
#         Validator.validate_cfg(self, cfg)
#         self._set_dependent_variables()

#     def _getExpID(self):
#         if os.path.exists(self.EXP_DIR):
#             id = len(os.listdir(self.EXP_DIR))
#         else:
#             id = 0
#         return f'exp_{id}'

#     def _set_dependent_variables(self):
#         self.NUM_KFOLDS = int(1/self.TEST_SPLIT)
#         self.FOLD_IDs = range(self.NUM_KFOLDS) if self.CROSS_VALIDATION else [self.DEFAULT_FOLD]
        
#         self.NUM_GPU = len(self.GPUs)
#         self.BATCH_SIZE = int(self.BATCH_SIZE/self.NUM_GPU)
#         self.DEFAULT_DEVICE = f'cuda:{self.GPUs[0] if self.GPUs else 0}' if torch.cuda.is_available() else "cpu"
#         self.PIN_MEMORY = True if torch.cuda.is_available() else False

#         self.DATASET_DIR = f'{self.DATASET_DIR}/{self.INPUT_IMAGE_SIZE}'
#         self.IMG_DIR = f'{self.DATASET_DIR}/images'
#         self.MSK_DIR = f'{self.DATASET_DIR}/masks'
#         self.SDM_DIR = f'{self.DATASET_DIR}/sdms'
#         self.EXP_DIR = f"{self.SAVE_DIR}/experiments"
#         self.TTS_JSON = f"{self.SAVE_DIR}/tts.json"
#         self.LBL_JSON = f'{self.DATASET_DIR}/labels.json'

#         self.MODEL_ID = f"{self.MODEL}_{self.LOSS}"
#         self.EXP_ID = self._getExpID() if self.EXP_ID == None else self.EXP_ID
#         self.RUN_ID = self.MODEL_ID if self.RUN_ID == None else self.RUN_ID

#         if self.MODEL == 'AuxNet-ED':
#             self.FREEZE_LAYERS = ['classifier']
#         elif self.MODEL == 'AuxNet-EDC':
#             self.FREEZE_LAYERS = []
#         elif self.MODEL == 'AuxNet-C':
#             self.FREEZE_LAYERS = ['encoder', 'decoder']
#         elif self.MODEL == 'AuxNet-D':
#             self.FREEZE_LAYERS = ['encoder', 'classifier']
#         elif self.MODEL == 'UNet':
#             self.FREEZE_LAYERS = []
#         elif self.MODEL == 'ClsCNN':
#             self.FREEZE_LAYERS = []

#         if 'Aux-Net' in self.MODEL:
#             self.METRICS = ['TTR', 'DSC', 'IoU', 'ASD', 'AD', 'HD95', 'D95', 'CMA']
#         elif 'UNet' in self.MODEL:
#             self.METRICS = ['DSC', 'IoU', 'ASD', 'AD', 'HD95', 'D95', 'CMA']
#         elif 'ClsCNN' in self.MODEL:
#             self.METRICS = ['TTR']


# CONF = Config('configs/config.yaml')

import os
import yaml
import torch
from pathlib import Path
from configs.validator import Validator

class Config:
    def __init__(self, cpath):
        cpath = Path(cpath)

        with cpath.open() as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        for key, value in cfg.items():
            setattr(self, key, value['value'])

        Validator.validate_cfg(self, cfg)
        self._set_dependent_variables()

    def _getExpID(self):
        if self.EXP_DIR.exists():
            id = sum(1 for d in self.EXP_DIR.iterdir() if d.is_dir())
        else:
            id = 0
        return f'exp_{id}'

    def _set_dependent_variables(self):
        # Derived numerical values
        self.NUM_KFOLDS = int(1 / self.TEST_SPLIT)
        self.FOLD_IDs = range(self.NUM_KFOLDS) if self.CROSS_VALIDATION else [self.DEFAULT_FOLD]
        self.NUM_GPU = len(self.GPUs)
        self.BATCH_SIZE = int(self.BATCH_SIZE / self.NUM_GPU)
        self.DEFAULT_DEVICE = f'cuda:{self.GPUs[0] if self.GPUs else 0}' if torch.cuda.is_available() else "cpu"
        self.PIN_MEMORY = torch.cuda.is_available()

        # Path-related attributes
        self.DATASET_DIR = Path(self.DATASET_DIR)
        self.RESULTS_DIR = Path(self.RESULTS_DIR)

        self.IMG_DIR = self.DATASET_DIR / 'images'
        self.MSK_DIR = self.DATASET_DIR / 'masks'
        self.SDM_DIR = self.DATASET_DIR / 'sdms'
        self.EXP_DIR = self.RESULTS_DIR / 'experiments'
        self.TTS_JSON = self.RESULTS_DIR / 'tts.json'
        self.LBL_JSON = self.DATASET_DIR / 'labels.json'

        # ID strings
        self.MODEL_ID = f"{self.MODEL}_{self.LOSS}"
        self.RUN_ID = self.MODEL_ID if self.RUN_ID is None else self.RUN_ID
        self.EXP_ID = self._getExpID() if self.EXP_ID is None else self.EXP_ID

        # Paths for saving model and logs
        self.MODEL_PTH = CONF.EXP_DIR / CONF.EXP_ID / f"{CONF.RUN_ID}-model.pth"
        self.LOG_JSON = CONF.EXP_DIR / CONF.EXP_ID / f"{CONF.RUN_ID}-log.json"

        # self.PT_WEIGHTS = CONF.PT_WEIGHTS 

        # Model-specific settings
        model_freeze_layers = {
            'AuxNet-ED': ['classifier'],
            'AuxNet-EDC': [],
            'AuxNet-C': ['encoder', 'decoder'],
            'AuxNet-D': ['encoder', 'classifier'],
            'UNet': [],
        }
        self.FREEZE_LAYERS = model_freeze_layers.get(self.MODEL, [])

        # Metrics
        if 'Aux-Net' in self.MODEL:
            self.METRICS = ['TTR', 'DSC', 'IoU', 'ASD', 'AD', 'HD95', 'D95', 'CMA']
        elif 'UNet' in self.MODEL:
            self.METRICS = ['DSC', 'IoU', 'ASD', 'AD', 'HD95', 'D95', 'CMA']

# Example usage
CONF = Config('configs/config.yaml')
