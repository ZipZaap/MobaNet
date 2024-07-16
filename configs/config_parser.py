import os
import yaml
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Config():
    def __init__(self, cpath):
        self.conf = AttrDict()
        with open(cpath) as file:
            yaml_cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.conf.update(yaml_cfg)

    def createFolders(self, lst: list) -> None:
        for path in lst:
            os.makedirs(path, exist_ok = True)
        
    def getConfig(self):
        self.conf.RUN_ID = f"{self.conf.MODEL}-{self.conf.LOSS}"
        self.conf.SDM_LOGITS = 'sdm' in self.conf.LOSS
        print('SDM_LOGITS: ', self.conf.SDM_LOGITS )

        self.conf.DATASET_PATH = f'{self.conf.DATASET_PATH}/{self.conf.INPUT_IMAGE_SIZE}'
        self.conf.IMAGE_DATASET_PATH = f'{self.conf.DATASET_PATH}/images'
        self.conf.MASK_DATASET_PATH = f'{self.conf.DATASET_PATH}/masks'
        self.conf.SDM_DATASET_PATH = f'{self.conf.DATASET_PATH}/sdms'
        self.conf.LABELS_JSON_PATH =f'{self.conf.DATASET_PATH}/labels.json'

        self.conf.DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"
        self.conf.PIN_MEMORY = True if torch.cuda.is_available() else False

        self.conf.MDL_PATH = f"{self.conf.OUTPUT_PATH}/{self.conf.MODEL}-model.pth"
        self.conf.LOG_PATH = f"{self.conf.OUTPUT_PATH}/{self.conf.MODEL}-log.json"
        self.conf.TTS_PATH = f"{self.conf.OUTPUT_PATH}/splits/tts_{self.conf.DSET}.json"
            
        self.createFolders([self.conf.OUTPUT_PATH, f'{self.conf.OUTPUT_PATH}/splits', self.conf.SDM_DATASET_PATH ])

        return self.conf
    


