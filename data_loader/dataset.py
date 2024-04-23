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

class AuxillaryDataset(Dataset):
    def __init__(self, imIDs, transforms, conf):
        self.imIDs = imIDs
        self.transforms = transforms
        self.typs = json.load(open(conf.LABELS_JSON_PATH))['imIDs']
        self.impath = conf.IMAGE_DATASET_PATH
        self.maskpath = conf.MASK_DATASET_PATH

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
    def __init__(self, imIDs, transforms, conf):
        self.imIDs = imIDs
        self.transforms = transforms
        self.impath = conf.IMAGE_DATASET_PATH
        self.maskpath = conf.MASK_DATASET_PATH

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

        image = torch.tensor(image/255).unsqueeze(0).to(torch.float32)
        mask = torch.tensor(mask/255).unsqueeze(0).to(torch.float32)
        return (image, mask)

class ClassificationDataset(Dataset):
    def __init__(self, imIDs, transforms, conf):
        self.imIDs = imIDs
        self.transforms = transforms
        self.typs = json.load(open(conf.LABELS_JSON_PATH))['imIDs']
        self.impath = conf.IMAGE_DATASET_PATH

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
    

def get_tts(conf):
    if conf.DSET == 'npld':
        typs = ['npld']
    elif conf.DSET == 'bu':
        typs = ['bu']
    elif conf.DSET == 'scarp':
        typs = ['scarp']
    elif conf.DSET == 'noscarp':
        typs = ['npld', 'bu']
    elif conf.DSET == 'all':
        typs = ['bu', 'npld', 'scarp']

    if os.path.exists(conf.TTS_PATH):
        with open(conf.TTS_PATH) as f:
            TTS = json.load(f)
        trainIDs = TTS["trainIDs"]
        testIDs = TTS["testIDs"]
    else:
        with open(conf.LABELS_JSON_PATH) as f:
            lblDict = json.load(f)
               
        testIDs = []
        trainIDs = []
        for typ in typs:
            random.seed(conf.SEED)
            Z = random.sample(lblDict['typ'][typ], conf.IM_PER_CAT)
            random.seed(conf.SEED)
            Y = random.sample(Z, int(conf.IM_PER_CAT * conf.TEST_SPLIT))
            X = list(set(Z).difference(set(Y)))

            testIDs.extend(Y)
            trainIDs.extend(X)
            
        TTS = {'trainIDs': trainIDs, 'testIDs': testIDs}
        with open(conf.TTS_PATH, 'w') as tts:
            json.dump(TTS, tts)

    return trainIDs, testIDs


def getDataloaders(conf):
    trainIDs, testIDs = get_tts(conf)

    train_transform = A.Compose(
        [
            A.VerticalFlip(p=0.5),  
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
            # A.RandomBrightnessContrast(p=0.8),  
            # A.RandomGamma(p=0.8)
        ])
        
    if conf.MODEL == 'AuxNet':
        trainDS = AuxillaryDataset(imIDs=trainIDs, transforms=train_transform, conf=conf)
        testDS = AuxillaryDataset(imIDs=testIDs, transforms=None, conf=conf)
    elif conf.MODEL == 'DenseNet':
        trainDS = ClassificationDataset(imIDs=trainIDs, transforms=train_transform, conf=conf)
        testDS = ClassificationDataset(imIDs=testIDs, transforms=None, conf=conf)
    elif conf.MODEL == 'UNet':
        trainDS = SegmentationDataset(imIDs=trainIDs, transforms=train_transform, conf=conf)
        testDS = SegmentationDataset(imIDs=testIDs, transforms=None, conf=conf)

    if conf.GPU_ID == 0:
        print(f"[INFO] found {len(trainDS)} examples in the training set...")
        print(f"[INFO] found {len(testDS)} examples in the test set...")
        print("------------------------------------------------------------")

        
    trainSampler = DistributedSampler(trainDS, num_replicas=conf.GPU_COUNT, rank=conf.GPU_ID, 
                                      shuffle=True, drop_last=True) if conf.GPU_COUNT > 1 else None
    testSampler = DistributedSampler(testDS, num_replicas=conf.GPU_COUNT, rank=conf.GPU_ID, 
                                     shuffle=False, drop_last=True) if conf.GPU_COUNT > 1 else None

    trainLoader = DataLoader(trainDS,
        batch_size=conf.BATCH_SIZE, pin_memory=conf.PIN_MEMORY, shuffle = conf.GPU_COUNT < 2,
        sampler = trainSampler, num_workers=conf.NUM_WORKERS)
    testLoader = DataLoader(testDS,
        batch_size=conf.BATCH_SIZE, pin_memory=conf.PIN_MEMORY, shuffle = False,
        sampler = testSampler, num_workers=conf.NUM_WORKERS)

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

