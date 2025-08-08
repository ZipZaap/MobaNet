# Multi-output Boundary-Aware U-Net for Semantic Segmentation

This repository provides an **end-to-end implementation** of a multi-output U-Net that supports both segmentation and classification, along with a set of distance-based loss functions for improved boundary prediction. All utilities needed to generate distance maps and train the model are included - no extra tooling required.

## :key: Key Features

1. **Multi-output architecture**  
   The network fuses segmentation *and* classification. When a large image is tiled, the classifier decides whether a tile is likely to contain an object boundary; only those positive tiles are forwarded to the segmentation branch. This approach:

   * Avoids unnecessary computation.
   * Enables boundary-aware losses during decoder training.
   * Produces uniform masks for single-class tiles.

2. **GPU-accelerated Signed Distance Transform (SDT) approximation**  
   Metrics such as Average Symmetric Surface Distance (ASD) and Hausdorff Distance 95th percentile (HD95) rely on Signed Distance Maps (SDMs) computed at every epoch. Traditional pipelines push tensors back to the CPU and call SciPy - which is slow and communication-heavy (especially on multi-GPU setups). Our PyTorch-only implementation uses cascaded convolutions, runs on batched data entirely on the GPU, and scales seamlessly with Distributed Data Parallel (DDP).

3. **A variety of loss function**  
   Pick from pixel-level, region-level, or boundary-aware losses or combine several. You can assign fixed or *learnable* weights to each component, letting you prioritise contour accuracy, global IoU, or any balance in between.

4. **Distributed training**  
   Native Distributed Data Parallel (DDP) support enables the use of multiple GPUs for both training and Signed Distane Map (SDM) calculations.

5. **Local & cloud logging**  
   Monitor progress with detailed console output, JSON logs, and optional Weights & Biases integration for remote experiment tracking.

## :rocket: Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/ZipZaap/AuxNet.git
    ```

2. Create a virtual envirnement & install the required dependencies

    ```bash
    conda env create -f requirements.yml
    ```

## :open_file_folder: Repository Structure

```graphql
├───configs/
│   ├──config.yaml ---------------------- # File with default parameters
│   ├──config_parser.py ----------------- # Defines Config() class which stores the defaults
│   └──validator.py --------------------- # Defines validation logic
|
├───engines/
│   └──SegTrainer.py--------------------- # Defines the main training loop
|
├───model/
│   ├───AuxNet.py ----------------------- # PyTorch model architechture 
│   ├───loss.py ------------------------- # Collection of loss functions used for training
│   └───metrics.py ---------------------- # Collection of metrics used for evaluation
│
├───saved/
│   └──tts.json ------------------------- # Train-test-split dictionary of image IDs
│
├───utils/
│   ├───dataset.py ---------------------- # PyTorch DataLoader class; also handles train/test split & image augmentation
│   ├───loggers.py ---------------------- # Tools for experiment tracking
│   ├───sdf.py -------------------------- # Tools for calculating the Signed Distane Function
│   └───util.py ------------------------- # Model loading, process-to-device allocation & misc functions
|
├───main.py ----------------------------- # Main executable
├───requirements.yml -------------------- # Core dependencies
└───README.md
```

> [!NOTE]
> Folders for storing experiment tracking data are indexed and created automatically in the `saved/` folder.

## :brain: Network architechture

...
image
...

The layout of the network is similar to that of a UNet with an aditional classification path barnching off at the bottlneck layer. Since network graph in our code is constructed dynamically, the user is free to customize the number of horizontal layers in the UNet or the feature depth of the Conv2D blocks, as well as the number of input channels and output classes.

## :straight_ruler: Signed Distance Transform

This repository delivers a fully-differentiable, GPU-native approximation of Signed Distance Maps using cascaded convolutions, following the method of [Pham et al.](https://doi.org/10.1007/978-3-030-71278-5_31) (MICCAI 2021). We build on [Kornia’s](https://github.com/kornia/kornia/blob/main/kornia/contrib/distance_transform.py) `distance_transform` kernel, adding Sobel-based edge extraction and per-map normalization - all implemented in pure PyTorch. Once the training loop starts, every tensor is created on the target device and never leaves it, eliminating host-device transfer overhead. Even on a single consumer-grade GPU, our batched implementation consistently outpaces SciPy’s CPU-bound `edt_distance_transform`, with only negligible loss in numerical accuracy.

## :chart_with_downwards_trend: Loss functions

The library ships with a self-contained, GPU-friendly collection of segmentation and classification losses, all exposed through a unified `Loss` wrapper. Core segmentation options include pixel-wise criteria (standard and weighted **BCE**), region-level losses (probabilstic and discrete **DICE**, **IoU**), as well as boundary aware losses (standard and clamped **MAE**, **Boundary Loss** from [Kervadec et al. (2019)](https://doi.org/10.1016/j.media.2020.101851.)). We also provide a custom **Sign** term that penalises distance-map sign errors, and helps overcome some of the limitations of the standard **MAE**. Any subset can be fused together with `CombinedLoss`, which supports fixed or learnable weights, enabling the network to balance multiple objectives during training.

## :hammer_and_wrench: Basic Usage

### Run options

All configurable options, sensible defaults, and variable types are defined in the [config.yaml](configs/config.yaml) file. This file serves as the primary interface for users to customize and tailor the network to their specific needs. Before training begins, the configuration is loaded and validated by the `Validator` method. This method performs pred-defined checks to ensure consistency and prevent conflicts between parameters. This helps catch potential issues early, reducing the likelihood of unexpected behaviors during execution.

<details>
   <summary> Parameter breakdown </summary>

   | | Option | Type | Description |
   |---|---|---|---|
   | **Directories** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `DATASET_DIR` | `str` | Path to the dataset folder. |
   || `RESULTS_DIR` | `str` | Path to the output folder. |
   | **Dataset** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `SEED` | `int` | Random seed for dataset split. |
   || `TRAIN_SET_COMPOSITION` | `str` | Training-set composition: <br>•`full`: unfiltered dataset  <br>•`boundary`: only images containing boundaries. |
   || `TEST_SET_COMPOSITION` | `str` | Test-set composition: <br>•`full`: unfiltered dataset  <br>•`boundary`: only images containing boundaries. |
   || `TEST_SPLIT` | `float` | Fraction reserved for testing. |
   || `CROSS_VALIDATION` | `bool` | Enable K-fold cross-validation. |
   || `DEFAULT_FOLD` | `int` | Fold to use when CV is disabled. |
   || `NUM_WORKERS` | `int` | Number of subprocesses used by PyTorch `DataLoader`. If set to 0, data loading occurs in the main process. |
   | **Model** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `MODEL` | `str` | The user can either train the entire model end-to-end in a single run, or train each component separately, using dedicated datasets and loss functions for each part. Available options: <br>• `AuxNet-ED`: Multi-output UNet with trainable Encoder & Decoder. <br> • `AuxNet-EDC`: Multi-output UNet with trainable Encoder, Decoder & Classification head.<br>• `AuxNet-C`: Multi-output UNet with trainable Classification head.<br>• `AuxNet-D`: Multi-output UNet with trainable Decoder.<br>• `UNet`: Standard U-Net architecture.  |
   || `MODEL_WEIGHTS` | `str` | Pre-trained weights filename (omit `.pth`). The user can preload weights from the previous run (e.g. `MODEL = AuxNet-ED`) and then train the Classifier separately by setting `MODEL = AuxNet-C`.|
   || `INPUT_SIZE` | `int` | Input image side length (pixels). |
   || `INPUT_CHANNELS` | `int` | Number of image channels. |
   || `UNET_DEPTH` | `int` | Number of down-sampling levels in a UNet (incl. bottleneck). |
   || `CONV_DEPTH` | `int` | Base feature-map depth of the Conv2D block (doubles per level). |
   || `BATCH_SIZE` | `int` | Training batch size. |
   | **Segmentation** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `SEG_CLASSES` | `int` | Number of segmentation classes. |
   || `SEG_DROPOUT` | `float` | Dropout for encoder/decoder. |
   || `SEG_THRESHOLD` | `float` | Mask binarisation threshold. |
   | **Classification** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `CLS_CLASSES` | `int` | Number of classification classes. |
   || `CLS_DROPOUT` | `float` | Dropout in classification head. |
   || `CLS_THRESHOLD` | `float` | Positive-class probability threshold. |
   | **Optimizer** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `INIT_LR` | `float` | Initial learning rate at the beginning of warmup. |
   || `BASE_LR` | `float` | Base learning rate reached by the end of warmup. |
   || `L2_DECAY` | `float` | L2 regularization decay. |
   || `WARMUP_EPOCHS` | `int` | Number of warmup epochs (0 = no warmup) |
   || `TRAIN_EPOCHS` | `int` | Number of training epochs (excl. warmup). |
   | **SDM** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `SDM_KERNEL_SIZE` | `int` | Kernel size for SDM estimation. |
   || `SDM_DISTANCE` | `str` | Type of distance used for the SDM. Available options: `manhattan`, `chebyshev`, `euclidean`. |
   || `SDM_NORMALIZATION` | `str` | SDM normalisation mode.  Available normalization options: <br>•`minmax`: by both max and min distance values of each individual SDM. <br>•`dynamic_max`: by the max distance value of each individual SDM. <br>•`static_max`: by the global max distance value (depends on `SDM_DISTANCE`)|
   | **Loss** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `LOSS` | `str` | Loss function used to train the model. Can either be a single loss or a combination of multiple losses, separated by `_` (e.g., `softDICE_BCE`). Available options: <br>•`SoftDICE`: Soft (probabilistic) DICE loss. <br>•`HardDICE`: Hard (discrete) DICE loss. <br>•`IoU`: Intersection over Union loss. <br>•`BCE`: Binary Cross-Entropy loss. <br>•`wBCE`: SDM Weighted Binary Cross-Entropy loss. <br>•`MAE`: Mean Absolute Error loss. <br>•`cMAE`: Clamped Mean Absolute Error loss. <br>•`sMAE`: Signed Mean Absolute Error loss. <- **OUR CONTRIBUTION** <br>•`Boundary`: Boundary loss. <br>•`CE`: Cross-Entropy loss. |
   || `INCLUDE_BACKGROUND` | `bool` | Include background class in `DICE`/`IoU`. |
   || `ADAPTIVE_WEIGHTS` | `bool` | Auto-balance multi-loss components. |
   || `STATIC_WEIGHTS` | `list` | Manual loss weights (if `ADAPTIVE_WEIGHTS = False`). |
   || `CLAMP_DELTA` | `float` | Delta for `cMAE` kernel clamping. Smaller values concentrate the network’s capacity on details near the boundary. |
   || `SIGMOID_STEEPNESS` | `int` | Sigmoid steepness for `DICE`/`IoU`/`sMAE` losses. Higher values yield a steeper curve and a closer approximation of the step function. |
   | **Evaluation** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `SAVE_MODEL` | `bool` | Save best model checkpoint. |
   || `EVAL_METRIC` | `str` | Best epoch selection metric: <br>•`TTR`: True-to-test ratio. Measures classification accuracy. <br>•`DSC`: DICE score. Measures global overlap. <br>•`IoU`: Intersection over Union score. Measures global overlap, but penalizes false positives more harshly. <br>•`ASD`: Average Symmetric Distance. Measures the mean distance between the boundaries of ground-truth and predicted segmentations. <br>•`AD`: Average one-way Distance. Relaxes the impact of false-positives. <br>•`HD95`: Hausdorff Distance 95th percentile. Measures the worst-case boundary discrepancy. <br>•`D95`: One-way Distance 95th percentile. Relaxes the impact of false-positives.  <br>•`CMA`: Combined Mean Accuracy. A weighted combination of the above. |
   || `CMA_COEFFICIENTS` | `dict` | Coefficients that define the contribution of each metric to the overall `CMA`. |
   || `DISTANCE_METRICS` | `bool` | Compute `ASD`, `AD`, `HD95`, `D95` during eval. |
   | **DDP** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `GPUs` | `list` | GPU indices for DDP. |
   || `MASTER_ADDR` | `str` | Address of the master node. |
   || `MASTER_PORT` | `str` | Port for DDP communication. |
   || `NCCL_P2P_DISABLE` | `bool` | Disable NCCL peer-to-peer (troubleshooting). |
   | **Logging** |--------------------------|-------|--------------------------------------------------------------------------------------|
   || `LOG_WANDB` | `bool` | Enable Weights & Biases logging. |
   || `LOG_LOCAL` | `bool` | Save logs locally. |
   || `EXP_ID` | `str` | Custom experiment identifier. |
   || `RUN_ID` | `str` | Custom run identifier. |

</details>

### Training & evaluation

### CMD output

## :artificial_satellite: Example use-case

## :memo: License

Distributed under the MIT License. See [`LICENSE`](LICENSE.txt) for more information.

## :envelope: Contact

Martynchuk Oleksii - martyn.chuckie@gmail.com

## :handshake: Acknowledgements

This project was made possible thanks to the support and resources provided by:

* [Technische Universität Berlin (TU Berlin)](https://www.tu.berlin/)
* [German Aerospace Center (DLR) Berlin](https://www.dlr.de/de/das-dlr/standorte-und-bueros/berlin)
* [HiRISE (High Resolution Imaging Science Experiment) team at the University of Arizona](https://www.uahirise.org/)
* [HEIBRIDS School for Data Science](https://www.heibrids.berlin/)

Additional thanks to the open‑source community and all contributors who help improve this project.