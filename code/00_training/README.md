# Stage 1 Autoencoder Training and Embedding Extraction

This directory contains the training code, model definition, and data loaders used to generate the 48-dimensional embeddings for supplier identification. Trained checkpoints are excluded from Git.

## Workflow

```text
voxel shape + manufacturing metadata
                  |
                  v
        MultiModalAutoencoder
                  |
                  v
          48-dimensional latent z
                  |
                  v
       train/test embedding CSV
                  |
                  v
 Stage 1 supplier identification
```

The model jointly processes a 3D voxel representation and manufacturing metadata. It combines the outputs of the shape and metric encoders into a 48-dimensional latent vector, `z`, which is used as the input to downstream supplier identification.

## Directory contents

```text
00_training/
├── 01_train_multimodal_autoencoder.ipynb
├── 02_extract_supplier_embeddings.ipynb
├── multimodal_autoencoder.py
├── voxel_dataset.py
├── binvox_io.py
├── training_config.py
└── checkpoints/                 # Local training output; excluded from Git
```

- `01_train_multimodal_autoencoder.ipynb`: trains the multimodal autoencoder and saves checkpoints.
- `02_extract_supplier_embeddings.ipynb`: loads a checkpoint and generates train/test embeddings.
- `multimodal_autoencoder.py`: defines the shape encoder/decoder, metric encoder/decoder, and `MultiModalAutoencoder`.
- `voxel_dataset.py`: loads BINVOX geometry as PyTorch tensors during embedding extraction.
- `binvox_io.py`: provides BINVOX input/output utilities.
- `training_config.py`: defines the default epochs, learning rate, batch size, and related settings.
- `checkpoints/`: stores locally generated model weights; `*.pth` files are excluded from Git.

## Input data

The CSV files used for training and embedding extraction are located at the following repository-relative paths:

```text
data/01_supplier_identification/main_split_70_30/
├── train_dataset_without_quantity.csv
└── test_dataset_without_quantity.csv
```

The code links CSV rows to voxel files using either the `filename` or `FileName` column. The model uses the following metadata columns:

- `LogScaled_Time`
- `LogScaled_Cost`
- `LogScaled_Quantity`
- `LogScaled_Tolerance`
- `LogScaled_Density`
- `LogScaled_Service_Temperature`
- `LogScaled_Ultimate_Tensile`

The embedding-extraction CSV also requires a `Supplier` column containing the supplier label or feasible supplier set.

Both notebooks read the included BINVOX geometry from the following repository-relative directory:

```text
data/voxel_geometry/
```

The `filename` or `FileName` value in each CSV row must match a file in this directory. The loader expands each BINVOX file into a `1 x 128 x 128 x 128` float32 tensor at runtime. Cached `.pt` tensors are not used.

## 1. Train the autoencoder

Run `01_train_multimodal_autoencoder.ipynb`. Its main default settings are:

| Setting | Value |
|---|---:|
| Epochs per run | 10 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.00004 |
| Adam betas | (0.8, 0.99) |
| Latent dimension | 48 |
| Metadata scale | 10.0 |
| Checkpoint interval | 5 epochs |

The quantity input is set to zero so that quantity does not influence supplier identification. Shape reconstruction combines weighted binary cross-entropy and Dice loss. The objective also includes reconstruction mean squared errors for time, cost, tolerance, and material properties.

Checkpoints are saved using the following naming convention:

```text
checkpoints/multimodal_autoencoder_epoch_NNN.pth
```

For example, the epoch 10 checkpoint is named `multimodal_autoencoder_epoch_010.pth`. Training-loss records and TensorBoard logs are also written under `checkpoints/`.

## 2. Extract embeddings

After training a model or obtaining a checkpoint separately, run `02_extract_supplier_embeddings.ipynb`. The default configuration selects the epoch 10 checkpoint, but the checkpoint itself is not distributed in this repository.

```python
RUN_ALL_CHECKPOINTS = False
RUN_SELECTED_EPOCHS = True
SELECTED_EPOCHS = [10]
```

The available execution modes are:

- `RUN_ALL_CHECKPOINTS = True`: process every `.pth` file in `checkpoints/`.
- `RUN_SELECTED_EPOCHS = True`: process the checkpoints listed in `SELECTED_EPOCHS`.
- Both options set to `False`: process the checkpoint specified by `model_path_single`.

The notebook saves the model's latent output, `z`, for every sample. Generated files are written to:

```text
data/01_supplier_identification/main_split_70_30/
├── train_embeddings_epoch_010.csv
└── test_embeddings_epoch_010.csv
```

Each output contains three columns:

| Column | Description |
|---|---|
| `filename` | Source voxel filename |
| `supplier` | Supplier ID for a training row or feasible supplier set for a test row |
| `embedding` | 48-dimensional latent vector |

The proposed supplier-identification notebooks consume these generated embedding CSVs directly. Metadata-enriched embedding files are needed only for the disabled benchmark paths and are not part of the default pipeline.

## 3. Run supplier identification

After generating embeddings, run:

```text
../01_supplier_identification/01_evaluate_supplier_identification.ipynb
../01_supplier_identification/02_tune_neighbor_threshold.ipynb
```

`01_evaluate_supplier_identification.ipynb` runs the proposed embedding-based method. The product-type and histogram methods are retained as benchmarks, but their imports and evaluation paths are disabled by default. `02_tune_neighbor_threshold.ipynb` evaluates the proposed method across neighbor-threshold values, `K`; its histogram benchmark path is also disabled by default.

## Environment

The primary Python dependencies are:

```text
torch
pandas
numpy
tqdm
tensorboard
jupyter
matplotlib
```

Run the notebooks from either the repository root or `code/00_training`. CUDA is used when available; otherwise, the code falls back to the CPU.

## Notes

- Trained checkpoints (`*.pth`) are excluded through `.gitignore`.
- The complete BINVOX set under `data/voxel_geometry/` is required for retraining and embedding regeneration.
- A locally generated or separately supplied `.pth` file must be used with the `MultiModalAutoencoder` architecture defined in `multimodal_autoencoder.py`.
