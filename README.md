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
requirements.txt
```

The quantity-free CSVs retain the quantity columns for schema compatibility, but the Stage 1 model does not use quantity as a supplier-identification signal. In the test CSV, `Supplier` may contain a comma-separated set of feasible suppliers.

## Data not included

The repository intentionally excludes:

- trained checkpoints (`*.pth`);
- cached voxel tensors (`*.pt`);
- raw voxel geometry (`*.binvox`);
- precomputed embedding CSVs;
- Stage 2/3 and robustness-analysis files.

Consequently, the included CSV files alone are not sufficient for full retraining. Provide a `pt_cache/` directory containing one cached tensor for each voxel name referenced by the CSV `filename` or `FileName` column. For example, `Bearing_10_Ball__1.binvox` maps to `pt_cache/Bearing_10_Ball__1.pt`. Both training and embedding extraction use these cached tensors, so the original BINVOX files are not required. Training creates a checkpoint locally, and embedding extraction uses that checkpoint to generate the CSVs consumed directly by the supplier-identification notebooks.

## Workflow

1. Place the external `pt_cache/` directory under an accessible voxel-data directory.
2. Update `voxel_dir` in both notebooks under `code/00_training/` if necessary.
3. Run `01_train_multimodal_autoencoder.ipynb` using `train_dataset_without_quantity.csv`.
4. Run `02_extract_supplier_embeddings.ipynb` to generate train/test embeddings.
5. Run the notebooks under `code/01_supplier_identification/`.

See `code/00_training/README.md` for model inputs, checkpoint behavior, and detailed execution notes.

## Environment

```bash
python -m pip install -r requirements.txt
```

CUDA is recommended for model training but is not required for reading the code or CSV metadata.
