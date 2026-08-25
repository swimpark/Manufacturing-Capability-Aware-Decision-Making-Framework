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

Consequently, the included CSV files alone are not sufficient for full retraining. To train the autoencoder, provide the voxel data referenced by the CSV `FileName` column. Training creates a checkpoint locally; embedding extraction then uses that checkpoint and the raw voxel files to generate the CSVs required by the supplier-identification notebooks.

## Workflow

1. Place the external voxel dataset in an accessible directory.
2. Update the voxel path in `code/00_training/01_train_multimodal_autoencoder.ipynb` if necessary.
3. Run `01_train_multimodal_autoencoder.ipynb` using `train_dataset_without_quantity.csv`.
4. Run `02_extract_supplier_embeddings.ipynb` to generate train/test embeddings.
5. Run the notebooks under `code/01_supplier_identification/`.

See `code/00_training/README.md` for model inputs, checkpoint behavior, and detailed execution notes.

## Environment

```bash
python -m pip install -r requirements.txt
```

CUDA is recommended for model training but is not required for reading the code or CSV metadata.
