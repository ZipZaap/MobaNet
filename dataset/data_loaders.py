import cv2
import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class SegmentationDataset(Dataset):
    def __init__(self, config, imIDs, transforms):
        self.imIDs = imIDs
        self.transforms = transforms
        self.impath = config.IMAGE_DATASET_PATH
        self.maskpath = config.MASK_DATASET_PATH
        self.sdmpath = config.SDM_DATASET_PATH

    def __len__(self):
        return len(self.imIDs)

    def __getitem__(self, idx):
        imID = self.imIDs[idx]
        image = cv2.imread(f'{self.impath}/{imID}.png', cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(f'{self.maskpath}/{imID}.png', cv2.IMREAD_GRAYSCALE)
        sdm = np.load(f'{self.sdmpath}/{imID}.npy')

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask, sdm = sdm)
            image = transformed['image']
            mask = transformed['mask']
            sdm = transformed['sdm']

        image = torch.tensor(image/255).unsqueeze(0).to(torch.float32)
        mask = torch.tensor(mask/255).unsqueeze(0).to(torch.float32)
        sdm = torch.tensor(sdm).unsqueeze(0).to(torch.float32)
        return (image, mask, sdm)

def getDataloaders(config, gpu_id, imIDs):
    trainIDs, testIDs = imIDs

    train_transform = A.Compose(
        [
            A.VerticalFlip(p=0.5),  
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
            # A.RandomBrightnessContrast(p=0.8),  
            # A.RandomGamma(p=0.8)
        ], additional_targets={'mask': 'mask', 'sdm': 'mask'})
        
    trainDS = SegmentationDataset(config = config, imIDs=trainIDs, transforms=train_transform)
    testDS = SegmentationDataset(config = config, imIDs=testIDs, transforms=None)

    trainSampler = DistributedSampler(trainDS, num_replicas=config.NUM_GPU, rank=gpu_id, 
                                      shuffle=True, drop_last=True) if config.NUM_GPU > 1 else None
    testSampler = DistributedSampler(testDS, num_replicas=config.NUM_GPU, rank=gpu_id, 
                                     shuffle=False, drop_last=True) if config.NUM_GPU > 1 else None
    
    trainLoader = DataLoader(trainDS,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, shuffle = config.NUM_GPU <= 1,
        sampler = trainSampler, num_workers=config.NUM_WORKERS, persistent_workers=True)
    testLoader = DataLoader(testDS,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, shuffle = False,
        sampler = testSampler, num_workers=config.NUM_WORKERS, persistent_workers=True)

    return trainLoader, testLoader


