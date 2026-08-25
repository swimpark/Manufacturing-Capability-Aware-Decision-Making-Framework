#!/usr/bin/env python
# coding: utf-8
# %%
# histogram_baseline_retriever.py
# -*- coding: utf-8 -*-

import os
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


METRIC_COLS_DEFAULT = [
    "LogScaled_Tolerance",
    "LogScaled_Time",
    "LogScaled_Cost",
    # "LogScaled_Quantity",
    "LogScaled_Density",
    "LogScaled_Service_Temperature",
    "LogScaled_Ultimate_Tensile",
]


# ------------------------------------------------------------
# ------------------------------------------------------------

def parse_supplier_label(x):
    """
      '1,19' -> [1, 19]
      '3'    -> [3]
    """
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]

    s = str(x).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        vals = ast.literal_eval(s)
        if isinstance(vals, (list, tuple)):
            return [int(v) for v in vals]
        return [int(vals)]

    if re.fullmatch(r"[+-]?\d+", s):
        return [int(s)]

    tokens = [t for t in re.split(r"[\s,;]+", s.strip("[](){}")) if t]
    try:
        return [int(t) for t in tokens]
    except Exception:
        raise ValueError(f"Invalid Supplier format: {x}")


def parse_shape_embedding(s):
    """
    """
    if pd.isna(s):
        raise ValueError("shape_embedding is NaN")
    vec = ast.literal_eval(str(s))
    return np.array(vec, dtype=np.float32)


def load_joint_embeddings_from_cirp_csv(path, metric_cols=None, shape_scale=10.0):
    """
    """
    df = pd.read_csv(path)

    if "filename" in df.columns:
        filenames = df["filename"].astype(str).tolist()
    elif "FileName" in df.columns:
        filenames = df["FileName"].astype(str).tolist()
    else:
        filenames = [f"idx_{i}" for i in range(len(df))]

    labels = df["Supplier"].apply(parse_supplier_label).tolist()

    shape_list = df["shape_embedding"].apply(parse_shape_embedding).tolist()
    shape_arr = np.stack(shape_list, axis=0).astype(np.float32)   # [N, 64]
    shape_arr = shape_arr * float(shape_scale)

    if metric_cols is None:
        metric_cols = [c for c in METRIC_COLS_DEFAULT if c in df.columns]
    metrics_arr = df[metric_cols].to_numpy(dtype=np.float32)      # [N, M]

    joint_arr = np.concatenate([shape_arr, metrics_arr], axis=1)  # [N, 64+M]
    embeddings = torch.tensor(joint_arr, dtype=torch.float32)

    return df, embeddings, labels, filenames, metric_cols


# ------------------------------------------------------------
# ------------------------------------------------------------

def parse_assembly_component(filename: str):
    """
    '<Assembly>_<Aidx>_<Component>__<Cidx>.(binvox|stl)' -> (assembly, component)
    """
    base = os.path.basename(str(filename))
    stem, _ext = os.path.splitext(base)

    m = re.match(r"^([A-Za-z0-9]+)_(\d+)_([A-Za-z0-9]+)__(\d+)$", stem)
    if m:
        return m.group(1), m.group(3)

    left = stem.split("__")[0]
    parts = left.split("_")
    assembly = parts[0] if len(parts) > 0 else ""
    component = parts[2] if len(parts) > 2 else ""
    return assembly, component


def canonicalize_component(name: str) -> str:
    """
    Normalize component names for comparison
    - lowercase
    - keep alphanumeric characters only
    - leafright, leafleft -> leaf
    """
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if key in {"leafright", "leafleft"}:
        return "leaf"
    return key


# ------------------------------------------------------------
# ------------------------------------------------------------

def compute_threshold(train_embeddings, k=3):
    """
    Because self-distance is included, k >= 2 is recommended.
    """
    dists = torch.cdist(train_embeddings, train_embeddings)
    k_eff = max(2, int(k))
    kth = torch.kthvalue(dists, k_eff, dim=1).values
    return kth.mean().item()


def compute_binary_confusion(preds, gts):
    preds = np.array(preds)
    gts = np.array(gts)
    TP = np.sum((preds == 1) & (gts == 1))
    FP = np.sum((preds == 1) & (gts == 0))
    FN = np.sum((preds == 0) & (gts == 1))
    TN = np.sum((preds == 0) & (gts == 0))
    return np.array([[TP, FN], [FP, TN]])


# ------------------------------------------------------------
# kNN + product-matching metrics
# ------------------------------------------------------------

def knn_hit_predict_with_product_metrics(
    val_embeds,
    train_embeds,
    train_labels,
    true_labels,
    train_filenames,
    test_filenames,
    threshold,
    require_same_component=False,
    require_same_assembly=False,
    log_top=5,
    margin_delta=0.01,
    verbose=True,
    debug_every=100,
    max_debug=100,
):
    """
    """
    dists = torch.cdist(val_embeds, train_embeds)  # [Nv, Nt]
    all_suppliers = sorted({s for sub in train_labels + true_labels for s in sub})

    train_ac = [parse_assembly_component(fn) for fn in train_filenames]
    test_ac = [parse_assembly_component(fn) for fn in test_filenames]
    train_assemblies = [a for a, c in train_ac]
    train_components_norm = [canonicalize_component(c) for a, c in train_ac]

    predicted_sets = []
    neighbor_indices = []

    purity_list = []
    any_violation_list = []
    wrong_top1_list = []
    margin_list = []
    total_correct = 0
    total_above = 0
    queries_with_above = 0

    debug_count = 0

    for i in range(dists.size(0)):
        test_name = test_filenames[i]
        gt_suppliers = true_labels[i]

        close_idxs = torch.where(dists[i] <= threshold)[0].tolist()
        if len(close_idxs) == 0:
            j_star = torch.argmin(dists[i]).item()
            close_idxs = [j_star]

        neighbor_indices.append(close_idxs)

        t_asm, t_cmp = test_ac[i]
        t_cmp_norm = canonicalize_component(t_cmp)

        def pass_product_rule(j):
            ok = True
            if require_same_component:
                ok = ok and (train_components_norm[j] == t_cmp_norm)
            if require_same_assembly:
                ok = ok and (train_assemblies[j] == t_asm)
            return ok

        prediction_neighbors = [j for j in close_idxs if pass_product_rule(j)]
        pred_suppliers = {sup for j in prediction_neighbors for sup in train_labels[j]}
        predicted_sets.append(pred_suppliers)

        same_comp = [j for j in close_idxs if train_components_norm[j] == t_cmp_norm]
        diff_comp = [j for j in close_idxs if train_components_norm[j] != t_cmp_norm]

        correct_cnt = len(same_comp)
        total_cnt = len(close_idxs)
        acc_ratio = correct_cnt / max(1, total_cnt)

        if (
            verbose
            and debug_count < max_debug
            and (i < 5 or i % debug_every == 0)
        ):
            debug_count += 1
            print(f"\n[Test {i}] {test_name} | GT: {gt_suppliers} | Pred: {sorted(pred_suppliers)}")
            print(f" - Above-threshold neighbors: {total_cnt}")
            print(f" - Correct neighbors (same-component): {correct_cnt}/{total_cnt}  (accuracy={acc_ratio:.3f})")

        if total_cnt > 0:
            queries_with_above += 1
            purity_list.append(correct_cnt / total_cnt)
            any_violation_list.append(1.0 if len(diff_comp) > 0 else 0.0)

            j_top = min(close_idxs, key=lambda j: dists[i, j].item())
            wrong_top1_list.append(
                1.0 if canonicalize_component(train_ac[j_top][1]) != t_cmp_norm else 0.0
            )

            valid_dists = [dists[i, j].item() for j in same_comp]
            invalid_dists = [dists[i, j].item() for j in diff_comp]
            if valid_dists and invalid_dists:
                mv = min(valid_dists)
                mi = min(invalid_dists)
                margin_list.append(mi - mv)

            total_correct += correct_cnt
            total_above += total_cnt

    report_data = []
    for s in all_suppliers:
        binary_gts = [1 if s in gts else 0 for gts in true_labels]
        binary_preds = [1 if s in pred_set else 0 for pred_set in predicted_sets]

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
        print("\nProduct-matching Quality @ threshold τ")

    if queries_with_above == 0 or total_above == 0:
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
                np.mean([1.0 if m <= margin_delta else 0.0 for m in margin_list])
            )
        else:
            margin_mean = margin_p5 = margin_p50 = margin_p95 = strict_rate = float("nan")

        coverage = float(queries_with_above / dists.size(0))

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

    return predicted_sets, metrics


# ------------------------------------------------------------
#   - run_retrieval()
#   - get_feasible_suppliers()
#   - get_test_info_dataframe()
# ------------------------------------------------------------

class Baseline2HistogramRetriever:
    """
    Baseline 2
    - D2 shape histogram + raw metrics joint embedding
    """

    def __init__(
        self,
        train_csv,
        test_csv,
        k_for_thr=6,
        metric_cols=None,
        shape_scale=10.0,
    ):
        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv)
        self.k_for_thr = k_for_thr
        self.metric_cols = metric_cols
        self.shape_scale = shape_scale

        self.train_df = None
        self.test_df = None
        self.train_embeddings = None
        self.test_embeddings = None
        self.train_labels = None
        self.test_labels = None
        self.train_filenames = None
        self.test_filenames = None

        self.threshold = None
        self.predicted_sets = None
        self.retrieval_metrics = None

    def _load_embeddings(self):
        if self.train_embeddings is not None:
            return

        self.train_df, self.train_embeddings, self.train_labels, self.train_filenames, used_cols = \
            load_joint_embeddings_from_cirp_csv(
                self.train_csv,
                metric_cols=self.metric_cols,
                shape_scale=self.shape_scale,
            )
        self.test_df, self.test_embeddings, self.test_labels, self.test_filenames, _ = \
            load_joint_embeddings_from_cirp_csv(
                self.test_csv,
                metric_cols=self.metric_cols,
                shape_scale=self.shape_scale,
            )
        self.metric_cols = used_cols

        print("[Baseline2] Loaded embeddings")
        print(f" - train: N={self.train_embeddings.shape[0]}, dim={self.train_embeddings.shape[1]}")
        print(f" - test : N={self.test_embeddings.shape[0]}, dim={self.test_embeddings.shape[1]}")

    def _build_test_info_df(self):
        """
        - query_idx
        - filename
        - assembly
        - component
        - family (assembly_component)
        - Supplier_GT
        """
        records = []
        for i, fn in enumerate(self.test_filenames):
            asm, comp = parse_assembly_component(fn)
            family = f"{asm}_{comp}"
            records.append(
                {
                    "query_idx": i,
                    "filename": fn,
                    "assembly": asm,
                    "component": comp,
                    "family": family,
                    "Supplier_GT": self.test_labels[i],
                }
            )
        self.test_df = pd.DataFrame(records)

    def run_retrieval(
        self,
        verbose=True,
        require_same_component=False,
        require_same_assembly=False,
        log_top=5,
        margin_delta=0.01,
        debug_every=100,
        max_debug=100,
    ):
        """
        """
        self._load_embeddings()

        if self.threshold is None:
            self.threshold = compute_threshold(self.train_embeddings, k=self.k_for_thr)
            print(f"[Baseline2] threshold (k={self.k_for_thr}): {self.threshold:.4f}")

        pred_sets, metrics = knn_hit_predict_with_product_metrics(
            self.test_embeddings,
            self.train_embeddings,
            self.train_labels,
            self.test_labels,
            self.train_filenames,
            self.test_filenames,
            threshold=self.threshold,
            require_same_component=require_same_component,
            require_same_assembly=require_same_assembly,
            log_top=log_top,
            margin_delta=margin_delta,
            verbose=verbose,
            debug_every=debug_every,
            max_debug=max_debug,
        )

        self.predicted_sets = pred_sets
        self.retrieval_metrics = metrics

        if self.test_df is None:
            self._build_test_info_df()

        return pred_sets, metrics

    def get_feasible_suppliers(self):
        """
        """
        if self.predicted_sets is None:
            raise RuntimeError("Operation failed or required precondition is missing.")

        return {i: sorted(list(sset)) for i, sset in enumerate(self.predicted_sets)}

    def get_test_info_dataframe(self):
        """
        """
        if self.test_df is None:
            self._build_test_info_df()
        return self.test_df.copy()

    def get_retrieval_metrics(self):
        return self.retrieval_metrics


# %%
