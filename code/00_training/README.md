# Model training and embedding extraction

Run these notebooks in order, from the repository root or this directory:

1. `01_train_multimodal_autoencoder.ipynb`
2. `02_extract_supplier_embeddings.ipynb`

Default settings: ten epochs, batch size eight, Adam learning rate 0.00004,
48-dimensional latent representation, and metadata scale 10. Quantity is fixed
to zero. Model architecture and loss calculations are retained from the
research source.

Inputs are the two CSVs in
`data/01_supplier_identification/main_split_70_30/` and BINVOX geometry in
`data/voxel_geometry/`. Extraction uses
`data/02_allocation_inputs/train_allocation_metadata.csv` to assign each
database row to its original supplier, rather than expanding paired labels.

Training saves `checkpoints/multimodal_autoencoder_epoch_010.pth` locally.
Extraction writes `train_embeddings_epoch_010.csv` and
`test_embeddings_epoch_010.csv` beside the input metadata. Each generated row
contains `sample_id`, `filename`, `supplier`, and `embedding`.

All generated checkpoints, embeddings, logs, and results are excluded from Git.
Continue with the commands in the root README after extraction.
