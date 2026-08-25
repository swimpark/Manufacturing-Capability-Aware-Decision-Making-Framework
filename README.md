# JMS Stage 1: Supplier Identification

This private repository contains the Stage 1 model-training and supplier-identification code, together with the quantity-free train/test CSV metadata used by the workflow.

## Included files

```text
code/
├── 00_training/                 # Autoencoder training and embedding extraction
└── 01_supplier_identification/  # Proposed retrieval plus disabled benchmark methods
data/01_supplier_identification/main_split_70_30/
├── train_dataset_without_quantity.csv
└── test_dataset_without_quantity.csv
data/voxel_geometry/              # 2,147 BINVOX geometry files
data/Component_binvox.Zip         # Original Train/Test/Total voxel archive
requirements.txt
```

`Component_binvox.Zip` is included unchanged as a convenience copy of the supplied voxel archive. It contains 4,985 BINVOX entries organized under `Train/`, `Test/`, and `Total/`, plus the two original `__temp_filtered.csv` files.

The quantity-free CSVs retain the quantity columns for schema compatibility, but the Stage 1 model does not use quantity as a supplier-identification signal. In the test CSV, `Supplier` may contain a comma-separated set of feasible suppliers.

## Data not included

The repository intentionally excludes:

- trained checkpoints (`*.pth`);
- cached voxel tensors (`*.pt`);
- precomputed embedding CSVs;
- Stage 2/3 and robustness-analysis files.

The repository includes the BINVOX geometry referenced by the train/test CSVs. Both training and embedding extraction read `data/voxel_geometry/*.binvox` directly, so no external voxel path or cached `.pt` tensors are required. Training creates a checkpoint locally, and embedding extraction uses that checkpoint to generate the CSVs consumed directly by the supplier-identification notebooks.

## Workflow

1. Run `01_train_multimodal_autoencoder.ipynb` using `train_dataset_without_quantity.csv` and the included BINVOX files.
2. Run `02_extract_supplier_embeddings.ipynb` to generate train/test embeddings.
3. Run the notebooks under `code/01_supplier_identification/`.

See `code/00_training/README.md` for model inputs, checkpoint behavior, and detailed execution notes.

## Environment

```bash
python -m pip install -r requirements.txt
```

CUDA is recommended for model training but is not required for reading the code or CSV metadata.
