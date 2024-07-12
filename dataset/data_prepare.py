import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm

import torch

from configs import CONF
from utils.util import SDF

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

    return (trainIDs, testIDs)

def save_sdms(imIDs, sdms):
    for imID, sdm in zip(imIDs, torch.split(sdms, 1, dim=0)):
        sdm = torch.abs(sdm[0][0]).detach().cpu().numpy()
        np.save(f'{CONF.SDM_DATASET_PATH}/{imID}.npy', sdm)

        colormap = cv2.applyColorMap((sdm*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(f'{CONF.SDM_DATASET_PATH}/{imID}.png', colormap)

def generate_sdms(imIDs):
    if len(os.listdir(CONF.SDM_DATASET_PATH)) == 0:
        print('[PROC] SDMs >>> generating ...')
        mask_batch = []
        imID_batch = []
        for i, imID in enumerate(tqdm(imIDs), start=1):
            mask = cv2.imread(f'{CONF.MASK_DATASET_PATH}/{imID}.png', cv2.IMREAD_GRAYSCALE)/255
            mask_batch.append(mask) 
            imID_batch.append(imID)
            if i%20 == 0 or i == len(imIDs):
                mask_batch = np.stack(mask_batch, axis=0)
                mask_batch_tensor = torch.tensor(mask_batch).to(torch.float32).unsqueeze(1)
                mask_batch_tensor = mask_batch_tensor.to(CONF.DEFAULT_DEVICE)

                SDMs = SDF(mask_batch_tensor, kernel_size = 7)
                save_sdms(imID_batch, SDMs)

                mask_batch = []
                imID_batch = []
    else:
        print('[PROC] SDMs >>> ready')