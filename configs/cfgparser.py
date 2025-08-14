import yaml
from typing import Any
from pathlib import Path
from configs.validator import Validator
from configs.cli import parse_cli_args

import torch

class Config:
    """
    A configuration loader that reads key:value pairs from a YAML file
    and sets them as attributes. After initialization, attributes are
    immutable unless their names are listed in `MUTABLE_KEYS`.
    """

    INFERENCE_KEYS: list[str] = ["DATASET_DIR", "NUM_WORKERS", "CHECKPOINT", "BATCH_SIZE", "GPUs", "CLS_THRESHOLD"]
    EXPORT_KEYS: list[str] = ["MODEL", "UNET_DEPTH", "CONV_DEPTH", "INPUT_CHANNELS", "SEG_CLASSES", "CLS_CLASSES", "SEG_DROPOUT", "CLS_DROPOUT"]
    MUTABLE_KEYS: list[str] = ["RANK", "DEVICE"]

    def __init__(self, config: str, *, inference: bool = False, cli: bool = False):

        # Immutability disabled by default
        self._frozen: bool = False
        self._inference: bool = inference

        # Load the YAML into a dict
        with Path(config).open() as f:
            cfg: dict[str, dict[str, Any]] = yaml.load(f, Loader=yaml.FullLoader)

        # If inference, filter out training-specific keys
        if inference:
            cfg = {k: v for k, v in cfg.items() if k in Config.INFERENCE_KEYS}

        # If CLI is enabled, parse command line arguments and update the config
        if cli:
            cfg = parse_cli_args(cfg)

        # Validate the configuration
        Validator.validate_cfg(cfg, inference)

        # Set attributes from <cfg> dict
        for key, value in cfg.items():
            setattr(self, key, value['default'])

        # Set dependent attributes
        self._set_dependent_attributes()

        # Mark as initialized; freeze attributes
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value):
        if name.startswith('_') or name in self.MUTABLE_KEYS or not getattr(self, '_frozen', False):
            return object.__setattr__(self, name, value)
        raise AttributeError(f"Cannot modify attribute '{name}'; it's immutable.")
    
    def __getattr__(self, name: str) -> Any:
        """
        Called when an attribute lookup fails in the normal places
        (i.e. it's not found in __dict__ or via __getattribute__).
        We check our __dict__ and return it to satisfy Pylance.
        """

        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self):
        """
        String representation of the Config object, showing its attributes.
        """

        attrs = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        return f"<Config {attrs}>"
    
    def _exp_id(self) -> str:
        """
        Generate an experiment ID based on the highest existing exp_N in the results directory.

        Returns
        -------
            id : str
                Next available experiment ID (e.g., 'exp_7' if 'exp_6' exists).
        """

        exp_nums = []
        if self.RESULTS_DIR.exists():
            for d in self.RESULTS_DIR.iterdir():
                if d.is_dir() and d.name.startswith("exp_"):
                    num = int(d.name.split("_")[1])
                    exp_nums.append(num)
        id = max(exp_nums, default=-1) + 1

        return f'exp_{id}'

    def _set_dependent_attributes(self):
        """
        Set dependent attributes based on the loaded configuration.
        This includes paths, derived numerical values, and model-specific settings.
        """

        # Core paths ----------------------------------------------------------
        self.DATASET_DIR = Path(self.DATASET_DIR)
        self.CHECKPOINT = Path(self.CHECKPOINT) if self.CHECKPOINT else None

        # Derived values ------------------------------------------------------
        self.DEFAULT_DEVICE = f'cuda:{self.GPUs[0]}' if self.GPUs else 'cuda:0'

        if self._inference and self.CHECKPOINT:
            # Prediction dataset paths ----------------------------------------
            self.PREDICT_DIR = self.DATASET_DIR / 'predict'
            self.IMG_DIR = self.PREDICT_DIR / 'images'
            self.MSK_DIR = self.PREDICT_DIR / 'masks'

            # Model-specific settings -----------------------------------------
            model_cfg = torch.load(self.CHECKPOINT,
                                   map_location='meta',
                                   mmap=True,
                                   weights_only=True)['config']

            for key, value in model_cfg.items():
                setattr(self, key, value)

        else:
            # ID strings ------------------------------------------------------
            self.EXP_ID = self.EXP_ID if self.EXP_ID else self._exp_id()
            self.RUN_ID = self.RUN_ID if self.RUN_ID else f"{self.MODEL}_{self.TRAIN_SET_COMPOSITION}"

            # Training dataset paths ------------------------------------------
            self.TRAIN_DIR = self.DATASET_DIR / 'train'
            self.IMG_DIR = self.TRAIN_DIR / 'images'
            self.MSK_DIR = self.TRAIN_DIR / 'masks'
            self.SDM_DIR = self.TRAIN_DIR / 'sdms'
            self.LBL_JSON = self.TRAIN_DIR / 'labels.json'

            # Results paths ---------------------------------------------------
            self.RESULTS_DIR = Path(self.RESULTS_DIR)
            self.EXP_DIR = self.RESULTS_DIR / self.EXP_ID
            self.TTS_JSON = self.RESULTS_DIR / 'tts.json'
            self.LOG_JSON = self.EXP_DIR / f"{self.RUN_ID}-log.json"
            self.BEST_EPOCH_JSON = self.EXP_DIR / f"{self.RUN_ID}-best.json"
            self.MODEL_PTH = self.EXP_DIR / f"{self.RUN_ID}-model.pth"

            # Warn if EXP_DIR exists and contains files with current RUN_ID
            if any(self.EXP_DIR.glob(f"*{self.RUN_ID}*")):
                print(f"[WARN]: Experiment directory '{self.EXP_DIR}' "
                      f"contains files matching RUN_ID '{self.RUN_ID}'; "
                      f"they will be overwritten.")

            # Derived values --------------------------------------------------
            self.NUM_KFOLDS = int(1 / self.TEST_SPLIT)
            self.WORLD_SIZE = len(self.GPUs)
            self.BATCH_SIZE = int(self.BATCH_SIZE / self.WORLD_SIZE)

            # Model-specific settings -----------------------------------------
            model_freeze_layers = {
                'MobaNet_EDC': [],
                'MobaNet_ED': ['classifier'],
                'MobaNet_EC': ['decoder'],
                'MobaNet_C': ['encoder', 'decoder'],
                'MobaNet_D': ['encoder', 'classifier'],
                'UNet': [],
            }
            self.FREEZE_LAYERS = model_freeze_layers[self.MODEL]

    
    def export(self) -> dict[str, Any]:
        """
        Export the model-specific parameter values. 
        Later needed to load the model for inference.

        Returns
        -------
            model_cfg : dict[str, Any]
                Dictionary containing the model-specific parameters.
        """

        model_cfg: dict[str, Any] = {}
        for k in Config.EXPORT_KEYS:
            model_cfg[k] = getattr(self, k)

        return model_cfg
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration to a dictionary representation.

        Returns
        -------
            cfg_dict : dict[str, Any]
                Dictionary containing all the public `Config` attributes.
        """
        
        cfg_dict = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, Path):
                    cfg_dict[k] = str(v)
                else:
                    cfg_dict[k] = v
        return cfg_dict