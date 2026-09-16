"""Attach scenario inputs to locally generated embeddings, with identity checks."""
from pathlib import Path
import ast
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/01_supplier_identification/main_split_70_30"
METADATA = ROOT / "data/02_allocation_inputs"

def supplier_key(value):
    text = str(value).strip()
    values = ast.literal_eval(text) if text.startswith("[") else text.split(",")
    return tuple(sorted(int(x) for x in values))

def attach_metadata(embeddings, metadata):
    required = {"sample_id", "filename", "supplier", "embedding"}
    if not required.issubset(embeddings.columns):
        raise ValueError("Regenerate embeddings with the provided extraction notebook (sample_id is required).")
    if embeddings.sample_id.duplicated().any() or metadata.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id; the source-row identity must be unique.")
    if set(embeddings.sample_id) != set(metadata.sample_id):
        raise ValueError("Embedding and scenario sample IDs differ; check the input split and missing geometry.")
    joined = metadata.merge(embeddings, on="sample_id", how="left", validate="one_to_one",
                            suffixes=("", "_embedding"), sort=False)
    if not (joined.filename == joined.filename_embedding).all():
        raise ValueError("Filename mismatch between embeddings and scenario metadata.")
    if not (joined.supplier.map(supplier_key) == joined.supplier_embedding.map(supplier_key)).all():
        raise ValueError("Supplier-label mismatch between embeddings and scenario metadata.")
    if joined.embedding.isna().any():
        raise ValueError("Missing embeddings.")
    return joined.drop(columns=["filename_embedding", "supplier_embedding"])

def main():
    for split in ("train", "test"):
        embeddings = pd.read_csv(DATA / f"{split}_embeddings_epoch_010.csv")
        metadata = pd.read_csv(METADATA / f"{split}_allocation_metadata.csv")
        result = attach_metadata(embeddings, metadata)
        output = DATA / f"{split}_embeddings_epoch_010_with_metadata.csv"
        result.to_csv(output, index=False)
        print(f"Saved {len(result)} rows: {output}")

if __name__ == "__main__":
    main()
