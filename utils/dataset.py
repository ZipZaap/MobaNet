import os
import cv2 
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import TypedDict
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
from sklearn.model_selection import KFold

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from configs.config_parser import CONF

class FullDataset(Dataset):
    def __init__(self,
                 imIDs: list[str],
                 transforms: list | None,
                 img_dir: Path = CONF.IMG_DIR,
                 msk_dir: Path = CONF.MSK_DIR,
                 sdm_dir: Path = CONF.SDM_DIR,
                 lbl_json: Path = CONF.LBL_JSON,
                 cls_classes: int = CONF.CLS_CLASSES
                 ) -> dict[str, torch.Tensor]:

        self.imIDs = imIDs
        self.transforms = transforms
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.sdm_dir = sdm_dir
        self.labels = json.load(lbl_json.open())['labels']
        self.num_classes = cls_classes

    def __len__(self):
        return len(self.imIDs)

    def __getitem__(self, idx: int):
        imID = self.imIDs[idx]
        label = self.labels[imID]

        impath = self.img_dir / f"{imID}.png"
        image = cv2.imread(str(impath), cv2.IMREAD_GRAYSCALE)

        maskpath = self.msk_dir / f"{imID}.png"
        mask = cv2.imread(str(maskpath), cv2.IMREAD_GRAYSCALE)

        sdmpath = self.sdm_dir / f"{imID}.npy"
        sdm = np.load(str(sdmpath)) if sdmpath.exists() else np.zeros(image.shape)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask, sdm=sdm)
            image = transformed['image']
            mask = transformed['mask']
            sdm = transformed['sdm']

        image = torch.tensor(image/255).unsqueeze(0).float()
        mask = torch.tensor(mask/255).unsqueeze(0).float()
        sdm = torch.tensor(sdm).unsqueeze(0).float()
        label = F.one_hot(torch.tensor(label), self.num_classes).float()

        return {'image': image, 'mask': mask, 'sdm': sdm, 'label': label}
    

class LblDict(TypedDict):
    background: list[str]
    foreground: list[str]
    boundary:   list[str]
    labels:     dict[str, int]


def generate_class_labels(img_dir: Path = CONF.IMG_DIR,
                          msk_dir: Path = CONF.MSK_DIR,
                          threshold: float = 0.01
                          ) -> LblDict:

    imIDs = [id.stem for id in img_dir.glob('*.png')]

    lbl_dict = {
        'background': [],
        'foreground': [],
        'boundary': [],
        'labels': {}
    }

    for id in tqdm(imIDs, desc="[PROC] Generating class labels"):
        msk_path = msk_dir / f"{id}.png"
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)

        mean_val = np.mean(mask / 255)
        if mean_val < threshold:
            lbl_dict['background'].append(id)
            lbl_dict['labels'][id] = 0
        elif mean_val > 1 - threshold:
            lbl_dict['foreground'].append(id)
            lbl_dict['labels'][id] = 1
        else:
            lbl_dict['boundary'].append(id)
            lbl_dict['labels'][id] = 2

    return lbl_dict


def compose_dataset(seed: int = CONF.SEED,
                    lbl_json: Path = CONF.LBL_JSON, 
                    tts_json: Path = CONF.TTS_JSON,
                    num_kfolds: int = CONF.NUM_KFOLDS
                    ) -> None:

    if lbl_json.exists():
        with lbl_json.open() as f:
            lbl_dict = json.load(f)
    else:
        lbl_dict = generate_class_labels()
        with lbl_json.open('w') as f:
            json.dump(lbl_dict, f)

    if tts_json.exists(): 
        print(f'[INFO] Using cached {tts_json}. To update the train/test split, delete the existing file.')
    else:
        random.seed(seed)
        num_samples = len(lbl_dict['boundary'])
        boundary = random.sample(lbl_dict['boundary'], num_samples)
        background = random.sample(lbl_dict['background'], num_samples)
        foreground = random.sample(lbl_dict['foreground'], num_samples)

        TTS = {}
        kf = KFold(n_splits=num_kfolds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(kf.split(boundary)):
            TTS[fold] = {
                'boundary_train': [boundary[i] for i in train_idx], 
                'boundary_test': [boundary[i] for i in test_idx],
                'complete_train': [imID for i in train_idx for imID in (boundary[i], background[i], foreground[i])],
                'complete_test': [imID for i in test_idx for imID in (boundary[i], background[i], foreground[i])]
                }
        
        with tts_json.open('w') as f:
            json.dump(TTS, f)
     
        
def get_dataloaders(rank: int, 
                    fold: int = CONF.DEFAULT_FOLD,
                    num_gpu: int = CONF.NUM_GPU,
                    batch_size: int = CONF.BATCH_SIZE,
                    num_workers: int = CONF.NUM_WORKERS,
                    pin_memory: bool = CONF.PIN_MEMORY,
                    tts_json: Path = CONF.TTS_JSON,
                    train_set: str = CONF.TRAIN_SET_COMPOSITION,
                    test_set: str = CONF.TEST_SET_COMPOSITION
                    ) -> tuple[DataLoader, DataLoader]:
    
    with tts_json.open() as f:
        TTS = json.load(f)

    trainIDs = TTS[str(fold)][f'{train_set}_train']
    testIDs = TTS[str(fold)][f'{test_set}_test']

    train_transform = A.Compose(
        [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5)
        ],
        additional_targets={'mask': 'mask', 'sdm': 'mask'}
    )

    trainDS = FullDataset(imIDs=trainIDs, transforms=train_transform)
    testDS = FullDataset(imIDs=testIDs, transforms=None)

    trainSampler = DistributedSampler(
        trainDS, num_replicas=num_gpu, rank=rank,
        shuffle=True, drop_last=True) if num_gpu > 1 else None

    testSampler = DistributedSampler(
        testDS, num_replicas=num_gpu, rank=rank,
        shuffle=False, drop_last=True) if num_gpu > 1 else None

    trainLoader = DataLoader(
        trainDS,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=num_gpu <= 1,
        sampler=trainSampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    testLoader = DataLoader(
        testDS,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=False,
        sampler=testSampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    return trainLoader, testLoader