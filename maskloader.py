import os
import cv2
import math
import json
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F

from utils.sdf import SDF
from configs.cfgparser import Config
from torch.utils.data import Dataset, DataLoader

cfg  = Config('configs/config.yaml', inference = False, cli = False)

def main():
    class MaskDataset(Dataset):
        def __init__(self,
                    imIDs: list[str],
                    msk_dir: Path,
                    C: int):
            self.imIDs = imIDs
            self.msk_dir = msk_dir
            self.C = C

        def __len__(self) -> int:
            return len(self.imIDs)

        def __getitem__(self, idx: int) -> torch.Tensor:
            maskpath = self.msk_dir / f"{self.imIDs[idx]}.png"
            mask = cv2.imread(str(maskpath), cv2.IMREAD_GRAYSCALE)
            mask = torch.from_numpy(mask).float()
            return mask

    batch: int = cfg.BATCH_SIZE
    lbl_json: Path = cfg.LBL_JSON
    msk_dir: Path = cfg.MSK_DIR
    workers: int = cfg.NUM_WORKERS
    C: int = cfg.SEG_CLASSES

    with lbl_json.open() as f:
        lblDict = json.load(f)
        imIDs = lblDict['boundary']

    dset = MaskDataset(imIDs, msk_dir, C)
    maskLoader = DataLoader(
        dset,
        batch_size=16,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=1,
        persistent_workers=True
    )

    for X in maskLoader:
        print(X.shape)
        break

if __name__ == "__main__":
    main()