import os
import cv2
import json
import random
import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from configs import CONF
from utils.util import SDF

class AuxillaryDataset(Dataset):
    def __init__(self, imIDs, transforms):
        self.imIDs = imIDs
        self.transforms = transforms
        self.typs = json.load(open(CONF.LABELS_JSON_PATH))['imIDs']
        self.impath = CONF.IMAGE_DATASET_PATH
        self.maskpath = CONF.MASK_DATASET_PATH

    def __len__(self):
        return len(self.imIDs)

    def __getitem__(self, idx):
        imID = self.imIDs[idx]
        image = cv2.imread(f'{self.impath}/{imID}.png', cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(f'{self.maskpath}/{imID}.png', cv2.IMREAD_GRAYSCALE)
        typ = np.zeros(3)
        typ[self.typs[imID]] = 1

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = torch.tensor(image/255).unsqueeze(0).to(torch.float32)
        mask = torch.tensor(mask/255).unsqueeze(0).to(torch.float32)
        typ = torch.tensor(typ).to(torch.float32)
        return (image, mask, typ)
    
class SegmentationDataset(Dataset):
    def __init__(self, gpu_id, imIDs, transforms):
        self.gpu_id = gpu_id
        self.imIDs = imIDs
        self.transforms = transforms
        self.impath = CONF.IMAGE_DATASET_PATH
        self.maskpath = CONF.MASK_DATASET_PATH

    def __len__(self):
        return len(self.imIDs)

    def __getitem__(self, idx):
        imID = self.imIDs[idx]
        image = cv2.imread(f'{self.impath}/{imID}.png', cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(f'{self.maskpath}/{imID}.png', cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = torch.tensor(image/255).to(self.gpu_id).unsqueeze(0).to(torch.float32)
        mask = torch.tensor(mask/255).to(self.gpu_id).unsqueeze(0).to(torch.float32)
        sdm = SDF(mask)
        return (image, mask, sdm)

class ClassificationDataset(Dataset):
    def __init__(self, imIDs, transforms):
        self.imIDs = imIDs
        self.transforms = transforms
        self.typs = json.load(open(CONF.LABELS_JSON_PATH))['imIDs']
        self.impath = CONF.IMAGE_DATASET_PATH

    def __len__(self):
        return len(self.imIDs)

    def __getitem__(self, idx):
        imID = self.imIDs[idx]
        image = cv2.imread(f'{self.impath}/{imID}.png', cv2.IMREAD_GRAYSCALE)
        typ = np.zeros(3)
        typ[self.typs[imID]] = 1

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']

        image = torch.tensor(image/255).unsqueeze(0).to(torch.float32) 
        typ = torch.tensor(typ).to(torch.float32)
        
        return (image, typ)
    

def get_tts():
    if CONF.DSET == 'npld':
        typs = ['npld']
    elif CONF.DSET == 'bu':
        typs = ['bu']
    elif CONF.DSET == 'scarp':
        typs = ['scarp']
    elif CONF.DSET == 'noscarp':
        typs = ['npld', 'bu']
    elif CONF.DSET == 'all':
        typs = ['bu', 'npld', 'scarp']

    if os.path.exists(CONF.TTS_PATH):
        with open(CONF.TTS_PATH) as f:
            TTS = json.load(f)
        trainIDs = TTS["trainIDs"]
        testIDs = TTS["testIDs"]
    else:
        with open(CONF.LABELS_JSON_PATH) as f:
            lblDict = json.load(f)
               
        testIDs = []
        trainIDs = []
        for typ in typs:
            random.seed(CONF.SEED)
            Z = random.sample(lblDict['typ'][typ], len(lblDict['typ']['scarp']))
            random.seed(CONF.SEED)
            Y = random.sample(Z, int(len(lblDict['typ']['scarp']) * CONF.TEST_SPLIT))
            X = list(set(Z).difference(set(Y)))

            testIDs.extend(Y)
            trainIDs.extend(X)
            
        TTS = {'trainIDs': trainIDs, 'testIDs': testIDs}
        with open(CONF.TTS_PATH, 'w') as tts:
            json.dump(TTS, tts)

    return trainIDs, testIDs


def getDataloaders(gpu_id):
    trainIDs, testIDs = get_tts()

    train_transform = A.Compose(
        [
            A.VerticalFlip(p=0.5),  
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
            # A.RandomBrightnessContrast(p=0.8),  
            # A.RandomGamma(p=0.8)
        ])
        
    if CONF.MODEL == 'AuxNet':
        trainDS = AuxillaryDataset(imIDs=trainIDs, transforms=train_transform)
        testDS = AuxillaryDataset(imIDs=testIDs, transforms=None)
    elif CONF.MODEL == 'DenseNet':
        trainDS = ClassificationDataset(imIDs=trainIDs, transforms=train_transform)
        testDS = ClassificationDataset(imIDs=testIDs, transforms=None)
    elif CONF.MODEL == 'UNet':
        trainDS = SegmentationDataset(imIDs=trainIDs, transforms=train_transform)
        testDS = SegmentationDataset(imIDs=testIDs, transforms=None)

    if gpu_id == 0:
        print(f"[INFO] found {len(trainDS)} examples in the training set...")
        print(f"[INFO] found {len(testDS)} examples in the test set...")
        print("------------------------------------------------------------")

        
    trainSampler = DistributedSampler(trainDS, num_replicas=CONF.GPU_COUNT, rank=gpu_id, 
                                      shuffle=True, drop_last=True) if CONF.GPU_COUNT > 1 else None
    testSampler = DistributedSampler(testDS, num_replicas=CONF.GPU_COUNT, rank=gpu_id, 
                                     shuffle=False, drop_last=True) if CONF.GPU_COUNT > 1 else None

    trainLoader = DataLoader(trainDS,
        batch_size=CONF.BATCH_SIZE, pin_memory=CONF.PIN_MEMORY, shuffle = CONF.GPU_COUNT < 2,
        sampler = trainSampler, num_workers=CONF.NUM_WORKERS)
    testLoader = DataLoader(testDS,
        batch_size=CONF.BATCH_SIZE, pin_memory=CONF.PIN_MEMORY, shuffle = False,
        sampler = testSampler, num_workers=CONF.NUM_WORKERS)

    return trainLoader, testLoader


# train_transform = A.Compose(
#     [
#         A.VerticalFlip(p=0.5),  
#         A.HorizontalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.OneOf([
#             A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
#             A.GridDistortion(p=0.5),
#             A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
#             ], p=0.8),
#         A.CLAHE(p=0.8),
#         A.RandomBrightnessContrast(p=0.8),  
#         A.RandomGamma(p=0.8)
#     ])

