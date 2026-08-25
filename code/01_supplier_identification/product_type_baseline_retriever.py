#!/usr/bin/env python
# coding: utf-8
# %%
# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# BENCHMARK ONLY: product-type comparison method; not part of the proposed model.
"""
Baseline 1 Retrieval: Product-type based supplier selection

- Uses train/test CSVs with columns:
  ['filename', 'supplier', 'embedding', 'Demand', 'Time', 'Cost',
   'GT_Time', 'GT_Cost', 'Tolerance']
- "family" is defined as filename without the right-side index:
    e.g. 'Bearing_10_Ball__2.binvox' -> 'Bearing_10_Ball'
- For each test sample, predicts all suppliers that have ever produced
  the same family in the training data.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple

import pandas as pd
from pathlib import Path
import re


class ProductTypeOnlyRetriever:
    """
    Baseline 1: Product-type based supplier selection.

      - run_retrieval(verbose=False) -> (pred_sets, metrics_dict)
      - get_feasible_suppliers() -> List[List[int]]
      - get_test_info_dataframe() -> pd.DataFrame
    """

    def __init__(
        self,
        train_csv: str,
        test_csv: str,
        k_for_threshold: int | None = None,
        require_same_component: bool = False,
        require_same_assembly: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        train_csv : str
            Training CSV path (history of produced parts).
        test_csv : str
            Test CSV path (queries).
        k_for_threshold : int, optional
            Not used in Baseline 1. Included for interface compatibility.
        require_same_component : bool, optional
            Not used in Baseline 1.
        require_same_assembly : bool, optional
            Not used in Baseline 1.
        """
        self.train_csv = train_csv
        self.test_csv = test_csv

        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)

        if "family" not in self.train_df.columns:
            self.train_df["family"] = self.train_df["filename"].astype(str).apply(
                self._extract_family_from_filename
            )
        if "family" not in self.test_df.columns:
            self.test_df["family"] = self.test_df["filename"].astype(str).apply(
                self._extract_family_from_filename
            )


        self.train_df["_supplier_list"] = self.train_df["supplier"].apply(
            self._parse_suppliers
        )
        self.test_df["_supplier_list"] = self.test_df["supplier"].apply(
            self._parse_suppliers
        )

        self.family_to_suppliers: Dict[str, set[int]] = {}
        for _, row in self.train_df.iterrows():
            fam = row["family"]
            sup_list = row["_supplier_list"]
            if fam not in self.family_to_suppliers:
                self.family_to_suppliers[fam] = set()
            self.family_to_suppliers[fam].update(sup_list)

        self._pred_sets: List[List[int]] | None = None
        self._metrics: Dict[str, float] | None = None

    @staticmethod
    def _parse_suppliers(val: Any) -> List[int]:
        """
        """
        if pd.isna(val):
            return []
        s = str(val).strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            try:
                out.append(int(p))
            except ValueError:
                continue
        return out
    
    @staticmethod
    def _extract_family_from_filename(name: str) -> str:
        """
        '<Assembly>_<Aidx>_<Component>__<Cidx>.binvox' -> 'Assembly_Component'
        """
        base = Path(str(name)).name
        m = re.match(r"^([A-Za-z0-9]+)_(\d+)_([A-Za-z0-9]+)__(\d+)\.binvox$", base)
        if m:
            assembly = m.group(1)
            component = m.group(3)
            return f"{assembly}_{component}"

        return base.split("__")[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_retrieval(self, verbose: bool = False) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Run product-type based retrieval for all test samples.

        Returns
        -------
        pred_sets : List[List[int]]
            For each test row i, a list of predicted supplier IDs.
        metrics : Dict[str, float]
            Retrieval metrics:
              - AvgPrecision, AvgRecall, AvgF1, AvgAccuracy
              - MacroPurity@τ, MicroPurity@τ
              - ViolationRate_any@τ, WrongProduct@1@τ
        """
        if self._pred_sets is None:
            self._pred_sets = self._compute_predictions()
        if self._metrics is None:
            self._metrics = self._compute_metrics(self._pred_sets)

        if verbose:
            m = self._metrics
            print("[Baseline 1 – Product-type based retrieval]")
            print(f" - Average Precision : {m['AvgPrecision']:.4f}")
            print(f" - Average Recall    : {m['AvgRecall']:.4f}")
            print(f" - Average F1-score  : {m['AvgF1']:.4f}")
            print(f" - Average Accuracy  : {m['AvgAccuracy']:.4f}")
            print(f" - MacroPurity@τ     : {m['MacroPurity@τ']:.4f}")
            print(f" - MicroPurity@τ     : {m['MicroPurity@τ']:.4f}")
            print(f" - ViolationRate_any : {m['ViolationRate_any@τ']:.4f}")
            print(f" - WrongProduct@1@τ  : {m['WrongProduct@1@τ']:.4f}")

        return self._pred_sets, self._metrics

    def get_feasible_suppliers(self) -> List[List[int]]:
        """

        Returns
        -------
        List[List[int]]
        """
        if self._pred_sets is None:
            self._pred_sets = self._compute_predictions()
        return self._pred_sets

    def get_test_info_dataframe(self) -> pd.DataFrame:
        """

        Returns
        -------
        pd.DataFrame
        """
        cols = ["filename", "family", "supplier"]
        existing = [c for c in cols if c in self.test_df.columns]
        return self.test_df[existing].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_predictions(self) -> List[List[int]]:
        """
        For each test sample, return all suppliers that produced
        the same family in the training set.
        """
        pred_sets: List[List[int]] = []
        for _, row in self.test_df.iterrows():
            fam = row["family"]
            sup_set = self.family_to_suppliers.get(fam, set())
            preds = sorted(sup_set)
            pred_sets.append(preds)
        return pred_sets

    def _compute_metrics(self, pred_sets: List[List[int]]) -> Dict[str, float]:
        """
        Simple multilabel retrieval metrics based on supplier ID match.
        """
        y_true_list = [set(s) for s in self.test_df["_supplier_list"]]
        y_pred_list = [set(ps) for ps in pred_sets]

        assert len(y_true_list) == len(y_pred_list)
        n = len(y_true_list)

        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        accs: List[float] = []

        total_tp = 0
        total_pred = 0
        total_true = 0

        violation_any = 0
        wrong_top1 = 0

        for y_true, y_pred in zip(y_true_list, y_pred_list):
            inter = y_true & y_pred
            union = y_true | y_pred

            tp = len(inter)
            fp = len(y_pred - y_true)
            fn = len(y_true - y_pred)

            total_tp += tp
            total_pred += len(y_pred)
            total_true += len(y_true)

            # per-query precision, recall, F1, Jaccard-accuracy
            if len(y_pred) > 0:
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            else:
                p = 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            acc = tp / len(union) if len(union) > 0 else 1.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            accs.append(acc)

            # Violation: no correct supplier at all
            if len(y_pred) > 0 and tp == 0:
                violation_any += 1

            if len(y_pred) > 0:
                top1 = sorted(y_pred)[0]
                if top1 not in y_true:
                    wrong_top1 += 1

        avg_precision = float(sum(precisions) / n) if n > 0 else 0.0
        avg_recall = float(sum(recalls) / n) if n > 0 else 0.0
        avg_f1 = float(sum(f1s) / n) if n > 0 else 0.0
        avg_acc = float(sum(accs) / n) if n > 0 else 0.0

        micro_purity = float(total_tp / total_pred) if total_pred > 0 else 0.0
        macro_purity = avg_precision

        violation_rate_any = float(violation_any / n) if n > 0 else 0.0
        wrong_product_top1 = float(wrong_top1 / n) if n > 0 else 0.0

        metrics = {
            "AvgPrecision": avg_precision,
            "AvgRecall": avg_recall,
            "AvgF1": avg_f1,
            "AvgAccuracy": avg_acc,
            "MacroPurity@τ": macro_purity,
            "MicroPurity@τ": micro_purity,
            "ViolationRate_any@τ": violation_rate_any,
            "WrongProduct@1@τ": wrong_product_top1,
        }
        return metrics

    
    def debug_low_recall_cases(self, threshold: float = 1.0, max_print: int = 20):
        y_true_list = [set(s) for s in self.test_df["_supplier_list"]]
        y_pred_list = [set(ps) for ps in self._compute_predictions()]

        rows = []
        for idx, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            if len(y_true) == 0:
                continue
            tp = len(y_true & y_pred)
            fn = len(y_true - y_pred)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if recall < threshold:
                rows.append((idx, recall, y_true, y_pred))

        print(f"# of queries with recall < {threshold}: {len(rows)}")
        for idx, r, y_true, y_pred in rows[:max_print]:
            print(f"[idx={idx}] recall={r:.3f}")
            print("  filename :", self.test_df.loc[idx, "filename"])
            print("  family   :", self.test_df.loc[idx, "family"])
            print("  GT       :", sorted(y_true))
            print("  Pred     :", sorted(y_pred))
            print("  Missed   :", sorted(y_true - y_pred))
            print("-" * 50)

    

if __name__ == "__main__":
    
    BASE = Path.cwd()
    RESULT_DIR = BASE / "data" / "01_supplier_identification" / "main_split_70_30"
    train_csv = RESULT_DIR / "train_embeddings_epoch_010_with_metadata.csv"
    test_csv = RESULT_DIR / "test_embeddings_epoch_010_with_metadata.csv"

    retriever = ProductTypeOnlyRetriever(train_csv, test_csv)
    pred_sets, metrics = retriever.run_retrieval(verbose=True)
    print("Number of queries:", len(pred_sets))
    



# %%
