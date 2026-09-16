"""Run the proposed supplier identification method (k=11)."""
from pathlib import Path
import json
from proposed_supplier_retriever import MultiLabelSupplierRetriever

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/01_supplier_identification/main_split_70_30"

def main():
    retriever = MultiLabelSupplierRetriever(
        DATA / "train_embeddings_epoch_010.csv",
        DATA / "test_embeddings_epoch_010.csv", k_for_threshold=11)
    _, metrics = retriever.run_retrieval(verbose=False)
    print(metrics)
    pools = retriever.get_feasible_suppliers()
    if isinstance(pools, dict):
        pools = [pools[i] for i in range(len(retriever.test_filenames))]
    output = ROOT / "results/01_supplier_identification/candidate_suppliers.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([sorted(int(x) for x in pool) for pool in pools], indent=2))
    print(f"Saved candidate sets: {output}")

if __name__ == "__main__":
    main()
