import os
import yaml
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Config():
    def __init__(self):
        self.conf = AttrDict()
        with open('configs/config.yaml') as file:
            yaml_cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.conf.update(yaml_cfg)

    def createFolders(self, lst: list) -> None:
        for path in lst:
            os.makedirs(path, exist_ok = True)

    def printModelConfing(self):
        print(f'[INFO] MODEL: {self.conf.MODEL_ID}, DATASET: {self.conf.DSET}, LOSS: {self.conf.LOSS}')

    def getConfig(self):
        self.conf.MODEL_ID = f"{self.conf.MODEL}:{self.conf.TRAIN_MODE}"

        self.conf.DATASET_PATH = f'{self.conf.DATASET_PATH}/{self.conf.INPUT_IMAGE_SIZE}'
        self.conf.IMAGE_DATASET_PATH = f'{self.conf.DATASET_PATH}/images'
        self.conf.MASK_DATASET_PATH = f'{self.conf.DATASET_PATH}/masks'
        self.conf.LABELS_JSON_PATH =f'{self.conf.DATASET_PATH}/labels.json'

        self.conf.GPU_ID = 0 if torch.cuda.is_available() else "cpu"
        self.conf.PIN_MEMORY = True if torch.cuda.is_available() else False
        self.conf.NUM_WORKERS = 1
        
        if self.conf.MODEL == 'AuxNet':
            if self.conf.TRAIN_MODE == 'encoder':
                self.conf.DSET = 'scarp'
                self.conf.LOAD_PATH = None 
                self.conf.TO_FREEZE = ['out1']
                self.conf.LOSS = 'segmentation'
                self.conf.SAVE_TRIG = 'test_loss'

            elif self.conf.TRAIN_MODE == 'decoder':
                self.conf.DSET  = 'scarp'
                self.conf.LOAD_PATH = f"{self.conf.OUTPUT_PATH}/_train/{self.conf.RUN_NAME}/encoder_model.pth"
                self.conf.TO_FREEZE = ['enc1', 'enc2', 'enc3', 'enc4', 'bneck', 'out1']
                self.conf.LOSS = 'segmentation'
                self.conf.SAVE_TRIG = 'test_loss'

            elif self.conf.TRAIN_MODE == 'dense':
                self.conf.DSET = 'all'
                self.conf.LOAD_PATH = f"{self.conf.OUTPUT_PATH}/_train/{self.conf.RUN_NAME}/encoder_model.pth"
                self.conf.TO_FREEZE = ['enc1', 'enc2', 'enc3', 'enc4', 'bneck', 'dec1', 'dec2', 'dec3', 'dec4', 'out2']
                self.conf.LOSS = 'classification'
                self.conf.SAVE_TRIG = 'test_loss'           
        
        elif self.conf.MODEL == 'DenseNet': 
            self.conf.TRAIN_MODE = 'default'
            self.conf.DSET = 'all'
            self.conf.LOSS = 'classification'
            self.conf.SAVE_TRIG = 'test_loss'

        elif self.conf.MODEL == 'UNet':
            self.conf.TRAIN_MODE = 'default'
            self.conf.DSET = 'scarp'
            self.conf.LOSS = 'segmentation'
            self.conf.SAVE_TRIG = 'test_loss'

        self.conf.MDL_PATH = f"{self.conf.OUTPUT_PATH}/_train/{self.conf.RUN_NAME}/{self.conf.MODEL_ID}_model.pth"
        self.conf.LOG_PATH = f"{self.conf.OUTPUT_PATH}/_train/{self.conf.RUN_NAME}/{self.conf.MODEL_ID}_log.json"
        self.conf.TTS_PATH = f"{self.conf.OUTPUT_PATH}/splits/tts_{self.conf.DSET}.json"
            
        self.createFolders([f"{self.conf.OUTPUT_PATH}/_train/{self.conf.RUN_NAME}"])
        self.printModelConfing()

        return self.conf
    


