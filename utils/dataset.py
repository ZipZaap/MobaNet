import os
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Iterator, Sequence
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from configs.cfgparser  import Config
from utils.util import load_png, load_mask, load_sdm

class KFold:
    """
    Lightweight replacement for sklearn.model_selection.KFold.
    """

    def __init__(self,
                 n_splits: int,
                 *,
                 shuffle: bool = False,
                 random_state: int | None = None):
        """
        Initialize the KFold object.

        Args
        ----
            n_splits : int
                Number of splits for K-Fold cross-validation.

            shuffle : bool
                Whether to shuffle the data before splitting.

            random_state : int | None
                Random seed for reproducibility; if None, no shuffling is performed.
        """

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: Sequence | int) -> Iterator[tuple[list[int], list[int]]]:
        """
        Generate indices to split data into training and test sets.

        Args
        ----
            X : Sequence | int
                Data to split; can be a sequence or an integer representing the number of samples.
        
        Returns
        -------
            Iterator[tuple[list[int], list[int]]]
                An iterator yielding tuples of (train_indices, test_indices) for each fold.
        """

        # Accept len(X) or treat X as the sample count
        n_samples = len(X) if not isinstance(X, int) else X

        # build & optionally shuffle the master index list
        indices = list(range(n_samples))
        if self.shuffle:
            rng = random.Random(self.random_state)
            rng.shuffle(indices)

        # compute fold sizes (identical to sklearn logic)
        fold_sizes = [n_samples // self.n_splits] * self.n_splits
        for i in range(n_samples % self.n_splits):
            fold_sizes[i] += 1

        # yield each (train_idx, test_idx) pair
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx  = indices[start:stop]
            train_idx = indices[:start] + indices[stop:]
            yield train_idx, test_idx
            current = stop

# ---------------------------------------------------------

class FullDataset(Dataset):
    def __init__(self,
                 imIDs: list[str],
                 transforms: A.Compose | None,
                 cfg: Config):
        """
        Initializes the Pytorch Dataset.

        Args
        ----
            imIDs : list[str]
                List of image IDs to be used in the dataset.

            transforms : A.Compose | None
                Albumentations transformations to be applied to the images and masks.

            cfg : Config
                Configuration object with the following attributes:
                - `.IMG_DIR` (Path): Directory containing the input images.
                - `.MSK_DIR` (Path): Directory containing the input masks.
                - `.SDM_DIR` (Path): Directory containing the input SDMs.
                - `.LBL_JSON` (Path): Path to the class labels JSON file.
                - `.CLS_CLASSES` (int): Number of classes for classification.
        """

        self.imIDs = imIDs
        self.transforms = transforms

        self.img_dir: Path = cfg.IMG_DIR
        self.msk_dir: Path = cfg.MSK_DIR
        self.sdm_dir: Path = cfg.SDM_DIR
        self.labels: dict = json.load(cfg.LBL_JSON.open())['id_to_label']
        self.cls_classes: int = cfg.CLS_CLASSES
        self.seg_classes: int = cfg.SEG_CLASSES

    def __len__(self) -> int:
        return len(self.imIDs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        imID = self.imIDs[idx]
        cls = self.labels[imID]

        # load image → (H, W, C)
        impath = self.img_dir / f"{imID}.png"
        image = load_png(impath)

        # load mask → (H, W, C)
        maskpath = self.msk_dir / f"{imID}.png"
        mask = load_mask(maskpath, self.seg_classes)

        # load SDM → (H, W, 1)
        sdmpath = self.sdm_dir / f"{imID}.npy"
        sdm = load_sdm(sdmpath, mask.shape)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask, sdm=sdm)
            image = transformed['image']
            mask = transformed['mask']
            sdm = transformed['sdm']

        # make PyTorch compatibel: (H, W, C) → (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        sdm = torch.from_numpy(sdm).permute(2, 0, 1).float()
        cls = torch.tensor(cls).long()

        return {'image': image, 'mask': mask, 'sdm': sdm, 'cls': cls}
    
class BoundaryMasksDataset(Dataset):
    def __init__(self,
                 imIDs: list[str],
                 msk_dir: Path,
                 seg_classes: int):
        """
        Initializes the BoundaryMasksDataset, which is used when generating SDMs

        Args
        ----
            imIDs : list[str]
                List of image IDs to be used in the dataset.

            msk_dir : Path
                Directory containing the mask images.

            seg_classes : int
                Number of segmentation classes (including background).
        """

        self.imIDs = imIDs
        self.msk_dir = msk_dir
        self.seg_classes = seg_classes
        
    def __len__(self) -> int:
        return len(self.imIDs)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        imID = self.imIDs[idx]

        # load mask → (H, W, C)
        mask = load_mask(self.msk_dir / f"{imID}.png", self.seg_classes)

        # make PyTorch compatible: (H, W, C) → (C, H, W)
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return {'id': imID, 'mask': mask}

class PredictDataset(Dataset):
    def __init__(self, impaths: list[Path]):
        """
        Initializes the PredictDataset for inference.

        Args
        ----
            impaths : list[Path]
                List of image paths to be used for prediction.
        """

        self.impaths = impaths

    def __len__(self) -> int:
        return len(self.impaths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        impath = self.impaths[idx]

        # load image → (H, W, C)
        image = load_png(impath)

        # # make PyTorch compatible: (H, W, C) → (C, H, W)
        # image = torch.from_numpy(image).permute(2, 0, 1).float()

        image = torch.from_numpy(image).float()

        return {'id': impath.stem, 'image': image}

# ---------------------------------------------------------

class DatasetTools():

    @classmethod
    def _fetch_IDs(cls, 
                   D: dict[str, list[str]],
                   idx: list[int]
                   ) -> tuple[list[str], list[str]]:
        """
        Splits the dataset into boundary and full images based on the provided indices.

        Args
        ----
            D : dict
                Dictionary mapping labels to image IDs.

            idx : np.ndarray
                Indices of the images to be split.

        Returns
        -------
            boundary : list[str]
                List containing boundary image IDs

            full : list[str]
                List containing all image IDs 
        """

        # Boundary class is (should be) the last one in the sorted dictionary keys.
        boundary_class : str = sorted(D.keys())[-1]

        full, boundary  = [], []
        for i in idx:
            for lbl, imID in D.items():
                full.append(imID[i])
                if lbl == boundary_class:
                    boundary.append(imID[i])
        
        return boundary, full

    @classmethod
    def _generate_class_labels(cls,
                               msk_dir: Path,
                               seg_classes: int,
                               threshold: float = 0.99,
                               ) -> dict:
        """
        Generates class labels for the dataset based on their respective masks.
        If any class label occupies the area greater than `threshold` portion of the mask, 
        the image is classified as that class. Otherwise, image is classified as boundary.

        Args
        ----
            msk_dir : Path
                Directory containing the mask images.

            seg_classes : int
                Number of classes in the segmentation task (incl. background).
                Also becomes the `id` of the boundary class.
                
            threshold : float
                Threshold value to classify the images.

        Returns
        -------
            lbl_dict : LblDict
                A dictionary containing image IDs organized by class label & boundary status.
        """

        imIDs = [id.stem for id in msk_dir.glob('*.png')]

        label_to_id = {}
        id_to_label = {}

        for id in tqdm(imIDs, desc="[PREP] Generating class labels"):
            maskpath = msk_dir / f"{id}.png"
            mask = load_mask(maskpath)

            counts = np.bincount(mask.ravel(), minlength=seg_classes)
            max_label = int(counts.argmax())
            max_fraction = counts[max_label] / mask.size

            if max_fraction >= threshold:
                lbl = max_label
            else:
                # Class 1 → 0, class 2 → 1, …, so the (N + 1)-th class is labeled N.
                lbl = seg_classes 

            label_to_id.setdefault(str(lbl), []).append(id)
            id_to_label[id] = lbl

        return {'label_to_id': label_to_id, 'id_to_label': id_to_label}

    @classmethod
    def compose_dataset(cls, cfg: Config):
        """
        Generate the class labels and the train/test split files. 
        Maintains 1:1 ratio of all classes in the full dataset.

        Args
        ----
            cfg : Config
                Configuration object with following attributes:
                - `.SEED` (int): Random seed for reproducibility.
                - `.LBL_JSON` (Path): Path to the class labels JSON file.
                - `.TTS_JSON` (Path): Path to the train/test splits JSON file.
                - `.MSK_DIR` (Path): Directory containing the input masks.
                - `.NUM_KFOLDS` (int): Number of folds for K-Fold cross-validation.
        """

        seed: int = cfg.SEED
        lbl_json: Path = cfg.LBL_JSON
        tts_json: Path = cfg.TTS_JSON
        msk_dir: Path = cfg.MSK_DIR
        num_kfolds: int = cfg.NUM_KFOLDS
        seg_classes: int = cfg.SEG_CLASSES

        if lbl_json.exists():
            with lbl_json.open() as f:
                lbl_dict = json.load(f)
        else:
            lbl_dict = cls._generate_class_labels(msk_dir, seg_classes)
            with lbl_json.open('w') as f:
                json.dump(lbl_dict, f)

        label_to_id = lbl_dict['label_to_id']

        if tts_json.exists():
            print('[INFO] Using cached `tts.json`. To update the train/test split, delete the existing file.')
        else:
            # random generator
            rng = random.Random(seed)

            # Ensure each class has the same number of samples
            n_samples = min(len(samples) for samples in label_to_id.values())

            # Sample the minimum number of samples from each class
            label_to_id = {k: rng.sample(v, n_samples) for k, v in label_to_id.items()}

            # generate train/test splits
            TTS = {}
            kf = KFold(n_splits=num_kfolds, shuffle=True, random_state=seed)
            for fold, (train_idx, test_idx) in enumerate(kf.split(n_samples)):

                boundary_train, full_train = cls._fetch_IDs(label_to_id, train_idx)
                boundary_test, full_test = cls._fetch_IDs(label_to_id, test_idx)

                TTS[fold] = {
                    'boundary_train': boundary_train,
                    'boundary_test': boundary_test,
                    'full_train': full_train,
                    'full_test': full_test,
                }

            # save train/test splits
            with tts_json.open('w') as f:
                json.dump(TTS, f)
        
    @classmethod       
    def train_dataloaders(cls,
                          cfg: Config
                          ) -> tuple[DataLoader, DataLoader]:
        """
        Returns the train and test dataloaders for the specified fold.

        Args
        ----
            cfg : Config
                Configuration object with the following attributes:
                - `.RANK` (int): Rank of the current process.
                - `.FOLD` (int): Fold number for K-Fold cross-validation.
                - `.WORLD_SIZE` (int): Number of GPUs used for distributed training.
                - `.BATCH_SIZE` (int): Batch size for the dataloaders.
                - `.NUM_WORKERS` (int): Number of workers for the dataloaders.
                - `.TTS_JSON` (Path): Path to the train/test splits JSON file.
                - `.TRAIN_SET` (str): Train set composition.
                - `.TEST_SET` (str): Test set composition.

        Returns
        -------
            trainLoader, testLoader : tuple
                Train and test dataloaders.
        """
        rank: int = cfg.RANK
        fold: int = cfg.DEFAULT_FOLD
        worldsize: int = cfg.WORLD_SIZE
        batch: int = cfg.BATCH_SIZE
        workers: int = cfg.NUM_WORKERS
        tts_json: Path = cfg.TTS_JSON
        train_set: str = cfg.TRAIN_SET
        test_set: str = cfg.TEST_SET

        with tts_json.open() as f:
            TTS = json.load(f)

        trainIDs = TTS[str(fold)][f'{train_set}_train']
        testIDs = TTS[str(fold)][f'{test_set}_test']
        print(f'[INFO] Total samples: {len(trainIDs) + len(testIDs)}')

        train_transform = A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.Rotate(limit=90, p=0.5),
                A.RandomBrightnessContrast(p=0.5)

            ],
            additional_targets={'mask': 'mask', 'sdm': 'mask'}
        )

        trainSet = FullDataset(imIDs=trainIDs, transforms=train_transform, cfg=cfg)
        testSet = FullDataset(imIDs=testIDs, transforms=None, cfg=cfg)

        trainSampler = DistributedSampler(
            trainSet, num_replicas=worldsize, rank=rank,
            shuffle=True, drop_last=True) if worldsize > 1 else None

        testSampler = DistributedSampler(
            testSet, num_replicas=worldsize, rank=rank,
            shuffle=False, drop_last=True) if worldsize > 1 else None

        trainLoader = DataLoader(
            trainSet,
            batch_size=batch,
            pin_memory=torch.cuda.is_available(),
            shuffle=(worldsize <= 1),
            sampler=trainSampler,
            num_workers=workers,
            persistent_workers=workers > 0
        )

        testLoader = DataLoader(
            testSet,
            batch_size=batch,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            sampler=testSampler,
            num_workers=workers,
            persistent_workers=workers > 0
        )

        return trainLoader, testLoader
    
    @classmethod
    def predict_dataloader(cls,
                           cfg: Config
                           ) -> DataLoader:
        """
        Returns the dataloader used for inference.

        Args
        ----
            cfg : Config
                Configuration object with the following attributes:
                - `.IMG_DIR` (Path): Directory containing the images for inference.
                - `.BATCH_SIZE` (int): Batch size for the dataloader.
                - `.NUM_WORKERS` (int): Number of workers for the dataloader.

        Returns
        -------
            predictLoader : DataLoader
                Dataloader for the images in the inference dataset.
        """

        img_dir: Path = cfg.IMG_DIR
        batch: int = cfg.BATCH_SIZE
        workers: int = cfg.NUM_WORKERS
        impaths: list[Path] = [impath for impath in img_dir.glob('*.png')]

        predictLoader = DataLoader(
            PredictDataset(impaths),
            batch_size=batch,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            num_workers=workers,
            persistent_workers=workers > 0
        )

        return predictLoader
    
    @classmethod
    def boundary_mask_dataloader(cls,
                                 cfg: Config
                                 ) -> DataLoader:
        """
        Returns the dataloader used to retrieve masks when generating the SDMs dataset.

        Args
        ----
            cfg : Config
                Configuration object with the following attributes:
                - `.MSK_DIR` (Path): Directory containing the masks.
                - `.LBL_JSON` (Path): Path to the class labels JSON file.
                - `.BATCH_SIZE` (int): Batch size for the dataloader.
                - `.NUM_WORKERS` (int): Number of workers for the dataloader.
                - `.SEG_CLASSES` (int): Number of segmentation classes.
        """

        msk_dir: Path = cfg.MSK_DIR
        lbl_json: Path = cfg.LBL_JSON
        batch: int = cfg.BATCH_SIZE
        workers: int = cfg.NUM_WORKERS
        seg_classes: int = cfg.SEG_CLASSES

        with lbl_json.open() as f:
            imIDs = json.load(f)['label_to_id'][str(seg_classes)]

        loader = DataLoader(
            BoundaryMasksDataset(imIDs, msk_dir, seg_classes),
            batch_size=batch,
            pin_memory=torch.cuda.is_available(),
            num_workers=workers,
            persistent_workers=(workers > 0)
        )

        return loader