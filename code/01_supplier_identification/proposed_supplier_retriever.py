#!/usr/bin/env python
# coding: utf-8
# %%
# -*- coding: utf-8 -*-
"""Identify feasible suppliers from multimodal embeddings.

Use MultiLabelSupplierRetriever with training and query embedding CSVs.
Call run_retrieval() to obtain candidate supplier sets and evaluation metrics,
then get_feasible_suppliers() to pass those sets to downstream allocation."""

import os
import re
import ast
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix,
)


# ============================================================
# Helper functions (label, filename parsing)
# ============================================================

def parse_label_str(x) -> List[int]:
    """Normalize supplier labels to a list of integer IDs."""
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]

    s = str(x).strip()

    # List/tuple string
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        vals = ast.literal_eval(s)
        if isinstance(vals, (list, tuple)):
            return [int(v) for v in vals]
        return [int(vals)]

    # Single integer
    if re.fullmatch(r"[+-]?\d+", s):
        return [int(s)]

    tokens = [t for t in re.split(r"[\s,;]+", s.strip("[](){}")) if t]
    return [int(t) for t in tokens]


def parse_assembly_component(filename: str) -> Tuple[str, str]:
    """
    '<Assembly>_<Aidx>_<Component>__<Cidx>.binvox' -> (assembly, component)
    """
    base = os.path.basename(str(filename))
    m = re.match(r"^([A-Za-z0-9]+)_(\d+)_([A-Za-z0-9]+)__(\d+)\.binvox$", base)
    if m:
        return m.group(1), m.group(3)

    # Fallback
    stem = re.sub(r"\.binvox$", "", base, flags=re.IGNORECASE)
    left = stem.split("__")[0]
    parts = left.split("_")
    assembly = parts[0] if len(parts) > 0 else ""
    component = parts[2] if len(parts) > 2 else ""
    return assembly, component


def canonicalize_component(name: str) -> str:
    """Normalize a component name and combine left/right leaf variants."""
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if key in {"leafright", "leafleft"}:
        return "leaf"
    return key


def compute_threshold(train_embeddings: torch.Tensor, k: int = 3) -> float:
    """Average the k-th nearest distances within the training set.

    Self-distances are included, so k >= 2 is recommended."""
    n_train = int(train_embeddings.size(0))
    if n_train == 0:
        raise ValueError("Cannot compute a threshold from an empty training set.")
    dists = torch.cdist(train_embeddings, train_embeddings)
    k_eff = min(max(1, int(k)), n_train)
    kth = torch.kthvalue(dists, k_eff, dim=1).values
    return kth.mean().item()


def compute_binary_confusion(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    preds = np.array(preds)
    gts = np.array(gts)
    TP = np.sum((preds == 1) & (gts == 1))
    FP = np.sum((preds == 1) & (gts == 0))
    FN = np.sum((preds == 0) & (gts == 1))
    TN = np.sum((preds == 0) & (gts == 0))
    return np.array([[TP, FN], [FP, TN]])


# ============================================================
# Main class
# ============================================================

class MultiLabelSupplierRetriever:
    """

    ----------
    train_embeddings : torch.Tensor [Nt, D]
    test_embeddings  : torch.Tensor [Nv, D]
    train_labels     : List[List[int]]
    test_labels      : List[List[int]]
    train_filenames  : List[str]
    test_filenames   : List[str]
    threshold        : float
    """

    def __init__(
        self,
        train_csv: Path,
        test_csv: Path,
        k_for_threshold: int = 11,
        require_same_component: bool = False,
        require_same_assembly: bool = False,
        margin_delta: float = 0.01,
    ):
        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv)
        self.k_for_threshold = k_for_threshold
        self.require_same_component = require_same_component
        self.require_same_assembly = require_same_assembly
        self.margin_delta = margin_delta

        self.train_embeddings: torch.Tensor = None
        self.test_embeddings: torch.Tensor = None
        self.train_labels: List[List[int]] = []
        self.test_labels: List[List[int]] = []
        self.train_filenames: List[str] = []
        self.test_filenames: List[str] = []

        self.train_assemblies: List[str] = []
        self.train_components_norm: List[str] = []
        self.test_ac: List[Tuple[str, str]] = []

        self.threshold: float = None
        self.predicted_sets: List[Set[int]] = []
        self.neighbor_indices: List[List[int]] = []
        self.metrics: Dict[str, Any] = {}

        self._load_data()
        self._prepare_product_meta()
        self._compute_default_threshold()

    # ---------------------------
    # Internal helpers
    # ---------------------------

    @staticmethod
    def _load_embeddings_from_csv(path: Path):
        """Load embeddings, supplier labels, and optional filenames from a CSV."""
        df = pd.read_csv(path)

        embeddings = torch.tensor(
            df["embedding"].apply(lambda s: list(ast.literal_eval(str(s)))).tolist(),
            dtype=torch.float32,
        )

        labels = df["supplier"].apply(parse_label_str).tolist()
        filenames = df["filename"].tolist() if "filename" in df.columns else [
            f"idx_{i}" for i in range(len(df))
        ]

        return embeddings, labels, filenames

    def _load_data(self):
        print(f"[MultiLabelSupplierRetriever] Load train embeddings: {self.train_csv}")
        self.train_embeddings, self.train_labels, self.train_filenames = \
            self._load_embeddings_from_csv(self.train_csv)

        print(f"[MultiLabelSupplierRetriever] Load test  embeddings: {self.test_csv}")
        self.test_embeddings, self.test_labels, self.test_filenames = \
            self._load_embeddings_from_csv(self.test_csv)

    def _prepare_product_meta(self):
        # Parse assembly and component names in the training set.
        train_ac = [parse_assembly_component(fn) for fn in self.train_filenames]
        self.train_assemblies = [a for a, c in train_ac]
        self.train_components_norm = [
            canonicalize_component(c) for a, c in train_ac
        ]

        # Parse assembly and component names in the test set.
        self.test_ac = [parse_assembly_component(fn) for fn in self.test_filenames]

    def _compute_default_threshold(self):
        self.threshold = compute_threshold(self.train_embeddings, k=self.k_for_threshold)
        print(
            f"[MultiLabelSupplierRetriever] Computed threshold "
            f"(k={self.k_for_threshold}): {self.threshold:.4f}"
        )

    # ---------------------------
    # Core retrieval
    # ---------------------------

    def run_retrieval(
        self,
        threshold: float = None,
        log_top: int = 5,
        verbose: bool = False,
    ) -> Tuple[List[Set[int]], Dict[str, Any]]:
        """Retrieve candidate suppliers and return their sets with evaluation metrics."""
        if threshold is None:
            threshold = self.threshold

        dists = torch.cdist(self.test_embeddings, self.train_embeddings)
        all_suppliers = sorted({s for sub in self.train_labels + self.test_labels for s in sub})

        self.predicted_sets = []
        self.neighbor_indices = []

        purity_list = []
        any_violation_list = []
        wrong_top1_list = []
        margin_list = []
        total_correct = 0
        total_above = 0
        queries_with_above = 0

        for i in range(dists.size(0)):
            test_name = self.test_filenames[i]
            gt_suppliers = self.test_labels[i]

            # ----------------------------------------------------
            # ----------------------------------------------------
            close_idxs = torch.where(dists[i] <= threshold)[0].tolist()
            if not close_idxs:
                j_nn = int(torch.argmin(dists[i]).item())
                close_idxs = [j_nn]
            self.neighbor_indices.append(close_idxs)

            t_asm, t_cmp = self.test_ac[i]
            t_cmp_norm = canonicalize_component(t_cmp)

            def pass_product_rule(j):
                ok = True
                if self.require_same_component:
                    ok = ok and (self.train_components_norm[j] == t_cmp_norm)
                if self.require_same_assembly:
                    ok = ok and (self.train_assemblies[j] == t_asm)
                return ok

            # ----------------------------------------------------
            # ----------------------------------------------------
            prediction_neighbors = [j for j in close_idxs if pass_product_rule(j)]
            if not prediction_neighbors:
                j_best = int(torch.argmin(dists[i]).item())
                prediction_neighbors = [j_best]

            pred_suppliers = {
                sup for j in prediction_neighbors for sup in self.train_labels[j]
            }
            self.predicted_sets.append(pred_suppliers)

            same_comp = [j for j in close_idxs if self.train_components_norm[j] == t_cmp_norm]
            diff_comp = [j for j in close_idxs if self.train_components_norm[j] != t_cmp_norm]

            correct_cnt = len(same_comp)
            total_cnt = len(close_idxs)
            acc_ratio = correct_cnt / max(1, total_cnt)

            if verbose:
                print(f"\n[Test {i}] {test_name} | GT: {gt_suppliers} | Pred: {sorted(pred_suppliers)}")
                print(f" - Above-threshold neighbors (with fallback): {total_cnt}")
                print(f" - Correct neighbors (same-component): {correct_cnt}/{total_cnt}  (accuracy={acc_ratio:.3f})")

                def print_list(idxs, title):
                    if not idxs:
                        print(f" - {title}: None")
                        return
                    idxs_sorted = sorted(idxs, key=lambda j: dists[i, j].item())[:log_top]
                    print(f" - {title}:")
                    for j in idxs_sorted:
                        asm, cmp = parse_assembly_component(self.train_filenames[j])
                        dist = dists[i, j].item()
                        print(
                            f"    > {self.train_filenames[j]:30s} | {asm}/{cmp:10s} | "
                            f"Suppliers:{self.train_labels[j]} | Dist:{dist:.4f}"
                        )

                print_list(same_comp, "Correct neighbors (same-component)")
                print_list(diff_comp, "False Component Type Match")

            if total_cnt > 0:
                queries_with_above += 1
                purity_list.append(correct_cnt / total_cnt)
                any_violation_list.append(1.0 if len(diff_comp) > 0 else 0.0)

                j_top = min(close_idxs, key=lambda j: dists[i, j].item())
                wrong_top1_list.append(
                    1.0 if canonicalize_component(
                        parse_assembly_component(self.train_filenames[j_top])[1]
                    ) != t_cmp_norm else 0.0
                )

                valid_dists = [dists[i, j].item() for j in same_comp]
                invalid_dists = [dists[i, j].item() for j in diff_comp]
                if valid_dists and invalid_dists:
                    mv = min(valid_dists)
                    mi = min(invalid_dists)
                    margin_list.append(mi - mv)

                total_correct += correct_cnt
                total_above += total_cnt

        # Compute supplier-level metrics for multilabel predictions.
        report_data = []
        for s in all_suppliers:
            binary_gts = [1 if s in gts else 0 for gts in self.test_labels]
            binary_preds = [1 if s in pred_set else 0 for pred_set in self.predicted_sets]

            cm = compute_binary_confusion(binary_preds, binary_gts)
            TP, FN = cm[0]
            FP, TN = cm[1]

            precision = precision_score(binary_gts, binary_preds, zero_division=0)
            recall = recall_score(binary_gts, binary_preds, zero_division=0)
            f1 = f1_score(binary_gts, binary_preds, zero_division=0)
            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            report_data.append((s, precision, recall, f1, accuracy))

        avg_p = np.mean([p for _, p, _, _, _ in report_data]) if report_data else float("nan")
        avg_r = np.mean([r for _, _, r, _, _ in report_data]) if report_data else float("nan")
        avg_f1 = np.mean([f1v for _, _, _, f1v, _ in report_data]) if report_data else float("nan")
        avg_acc = np.mean([acc for _, _, _, _, acc in report_data]) if report_data else float("nan")

        if verbose:
            print("\nSupplier-wise Classification Report")
            print("Supplier | Precision | Recall | F1     | Accuracy")
            for s, p, r, f1v, acc in report_data:
                print(f"{s:^8} | {p:.4f}    | {r:.4f} | {f1v:.4f} | {acc:.4f}")

            print("\nAverage Metrics Across All Suppliers")
            print(f"Precision: {avg_p:.4f} | Recall: {avg_r:.4f} | F1-score: {avg_f1:.4f} | Accuracy: {avg_acc:.4f}")

        # Evaluate product-matching quality at the retrieval threshold.
        if queries_with_above == 0 or total_above == 0:
            if verbose:
                print("\nProduct-matching Quality @ threshold τ")
                print(" - No above-threshold neighbors across queries. Metrics undefined.")
            metrics = {
                "MacroPurity@τ": float("nan"),
                "MicroPurity@τ": float("nan"),
                "ViolationRate_any@τ": float("nan"),
                "WrongProduct@1@τ": float("nan"),
                "Margin_mean": float("nan"),
                "Margin_p5": float("nan"),
                "Margin_p50": float("nan"),
                "Margin_p95": float("nan"),
                "StrictViolationRate_margin@δ": float("nan"),
                "Coverage@τ": 0.0,
                "AvgPrecision": avg_p,
                "AvgRecall": avg_r,
                "AvgF1": avg_f1,
                "AvgAccuracy": avg_acc,
            }
        else:
            macro_purity = float(np.mean(purity_list)) if purity_list else float("nan")
            micro_purity = float(total_correct / total_above) if total_above > 0 else float("nan")
            viol_any = float(np.mean(any_violation_list)) if any_violation_list else float("nan")
            wrong_top1 = float(np.mean(wrong_top1_list)) if wrong_top1_list else float("nan")

            if margin_list:
                margin_mean = float(np.mean(margin_list))
                margin_p5 = float(np.percentile(margin_list, 5))
                margin_p50 = float(np.percentile(margin_list, 50))
                margin_p95 = float(np.percentile(margin_list, 95))
                strict_rate = float(
                    np.mean([1.0 if m <= self.margin_delta else 0.0 for m in margin_list])
                )
            else:
                margin_mean = margin_p5 = margin_p50 = margin_p95 = strict_rate = float("nan")

            coverage = float(queries_with_above / dists.size(0))

            if verbose:
                print("\nProduct-matching Quality @ threshold τ")
                print(f" - MacroPurity@τ: {macro_purity:.4f}")
                print(f" - MicroPurity@τ: {micro_purity:.4f}")
                print(f" - Violation-rate(any)@τ: {viol_any:.4f}")
                print(f" - Wrong-product@1@τ: {wrong_top1:.4f}")
                print(f" - Margin_mean: {margin_mean:.4f}")
                print(
                    f" - Margin_p5/p50/p95: {margin_p5:.4f} / "
                    f"{margin_p50:.4f} / {margin_p95:.4f}"
                )
                print(f" - StrictViolationRate(margin, δ={self.margin_delta}): {strict_rate:.4f}")
                print(f" - Coverage@τ: {coverage:.4f}")

            metrics = {
                "MacroPurity@τ": macro_purity,
                "MicroPurity@τ": micro_purity,
                "ViolationRate_any@τ": viol_any,
                "WrongProduct@1@τ": wrong_top1,
                "Margin_mean": margin_mean,
                "Margin_p5": margin_p5,
                "Margin_p50": margin_p50,
                "Margin_p95": margin_p95,
                "StrictViolationRate_margin@δ": strict_rate,
                "Coverage@τ": coverage,
                "AvgPrecision": avg_p,
                "AvgRecall": avg_r,
                "AvgF1": avg_f1,
                "AvgAccuracy": avg_acc,
            }

        self.metrics = metrics
        self.threshold = threshold
        return self.predicted_sets, metrics

    # ---------------------------
    # Access retrieval results for downstream allocation.
    # ---------------------------

    def get_feasible_suppliers(self) -> List[Set[int]]:
        """Return the candidate supplier sets from the most recent retrieval run."""
        if not self.predicted_sets:
            raise RuntimeError("Operation failed or required precondition is missing.")
        return self.predicted_sets

    def get_test_info_dataframe(self) -> pd.DataFrame:
        """Summarize query identities, reference suppliers, and predicted suppliers."""
        if not self.predicted_sets:
            raise RuntimeError("Operation failed or required precondition is missing.")

        rows = []
        for fn, (asm, cmp), gt, pred in zip(
            self.test_filenames, self.test_ac, self.test_labels, self.predicted_sets
        ):
            rows.append(
                {
                    "filename": fn,
                    "assembly": asm,
                    "component": cmp,
                    "gt_suppliers": sorted(gt),
                    "pred_suppliers": sorted(pred),
                }
            )
        return pd.DataFrame(rows)




# %%
