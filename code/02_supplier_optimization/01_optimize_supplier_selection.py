"""Stage 2 allocation using the original cost-minimizing solver and scoring."""
from pathlib import Path
import os
import sys
import re
import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/01_supplier_identification/main_split_70_30"
ALPHA_COST = 10.0
BETA_TIME = 10.0
ALPHA_UNMET = 10.0
TOL_EPS = 1e-9

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Columns not found in df: {candidates}")

def parse_assembly_component(filename: str):
    base = os.path.basename(str(filename))
    m = re.match(r"^([A-Za-z0-9]+)_(\d+)_([A-Za-z0-9]+)__(\d+)\.binvox$", base)
    if m:
        return m.group(1), m.group(3)

    stem = re.sub(r"\.(binvox|stl|csv)$", "", base, flags=re.IGNORECASE)
    left = stem.split("__")[0]
    parts = left.split("_")
    assembly = parts[0] if len(parts) > 0 else ""
    component = parts[2] if len(parts) > 2 else ""
    return assembly, component

def canonicalize_component(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if key in {"leafright", "leafleft"}:
        return "leaf"
    return key

def feasible_to_list(feasible, nQ: int):
    if isinstance(feasible, dict):
        return [feasible[q] for q in range(nQ)]
    return feasible

def solve_single_query_milp(
    query_idx: int,
    feasible_suppliers,
    supplier_comp_params: dict,
    comp_key: str,
    demand: float,
    big_lambda_unmet: float,
    objective: str = "cost",  # "cost" or "time"
):
    active = [s for s in feasible_suppliers if (s, comp_key) in supplier_comp_params]
    if (len(active) == 0) or (demand <= 0):
        return "Infeasible", np.nan, np.nan, np.nan, float(max(demand, 0.0)), {}

    prob = pulp.LpProblem(f"Scenario2_q{query_idx}_{objective}", pulp.LpMinimize)
    x = {s: pulp.LpVariable(f"x_q{query_idx}_s{s}", lowBound=0) for s in active}
    unmet = pulp.LpVariable(f"unmet_q{query_idx}", lowBound=0)

    coeff_key = "cost" if objective == "cost" else "time"

    prob += pulp.lpSum(supplier_comp_params[(s, comp_key)][coeff_key] * x[s] for s in active) + big_lambda_unmet * unmet
    prob += pulp.lpSum(x[s] for s in active) + unmet == demand

    for s in active:
        prob += x[s] <= supplier_comp_params[(s, comp_key)]["capacity"]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus[status]

    x_sol = {s: float(x[s].value() or 0.0) for s in active}
    unmet_qty = float(unmet.value() or 0.0)

    pred_cost = sum(supplier_comp_params[(s, comp_key)]["cost"] * x_sol[s] for s in active)
    pred_time = sum(supplier_comp_params[(s, comp_key)]["time"] * x_sol[s] for s in active)

    return status_str, float(pulp.value(prob.objective)), float(pred_time), float(pred_cost), float(unmet_qty), x_sol

def compute_tol_reject_qty_exact(
    x_sol: dict,
    req_tol: float,
    family_key: str,
    supplier_comp_tol_cap: dict,
) -> float:
    reject_qty = 0.0
    for sup, qty in x_sol.items():
        if qty <= 1e-9:
            continue
        tau_cap = supplier_comp_tol_cap.get((sup, family_key), None)
        if tau_cap is None:
            reject_qty += qty
            continue
        if abs(float(tau_cap) - float(req_tol)) > TOL_EPS:
            reject_qty += qty
    return float(reject_qty)

def run_stage2_for_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feasible_suppliers_per_query):
    # --- train columns
    col_sup_train  = find_col(train_df, ["supplier", "Supplier"])
    col_file_train = find_col(train_df, ["filename", "Filename", "FileName"])
    col_cap_train  = find_col(train_df, ["Capacity", "capacity", "Quantity (Ea)", "quantity"])

    col_cost_train = find_col(train_df, ["Cost", "unit_cost", "GT_cost", "gt_cost", "Mfg_Cost", "mfg_cost", "cost"])
    col_time_train = find_col(train_df, ["Time", "unit_time", "GT_time", "gt_time", "Mfg_Time", "mfg_time", "time"])
    col_tol_train  = find_col(train_df, ["Tolerance", "tolerance", "Tol", "tol"])

    # --- test columns
    col_demand        = find_col(test_df, ["Demand", "demand"])
    col_filename_test = find_col(test_df, ["filename", "Filename", "FileName"])
    col_gt_cost       = find_col(test_df, ["GT_Cost", "GT_cost", "gt_cost"])
    col_gt_time       = find_col(test_df, ["GT_Time", "GT_time", "gt_time"])
    col_req_tol       = find_col(test_df, ["Tolerance", "tolerance", "Req_Tolerance", "req_tolerance", "GT_Tolerance", "gt_tolerance", "Tol", "tol"])

    # --- supplier_comp_params: capacity=max, cost/time keep first
    supplier_comp_params = {}
    for _, row in train_df.iterrows():
        sup = int(row[col_sup_train])
        fn  = row[col_file_train]
        cap  = float(row[col_cap_train])
        cost = float(row[col_cost_train])
        t    = float(row[col_time_train])

        _, comp_raw = parse_assembly_component(fn)
        comp_key = canonicalize_component(comp_raw)

        key = (sup, comp_key)
        if key not in supplier_comp_params:
            supplier_comp_params[key] = {"capacity": cap, "cost": cost, "time": t}
        else:
            supplier_comp_params[key]["capacity"] = max(supplier_comp_params[key]["capacity"], cap)

    # --- supplier_comp_tol_cap: min achieved tolerance per (supplier, component_key)
    supplier_comp_tol_cap = (
        train_df
        .assign(component_key=lambda df: df[col_file_train].apply(lambda fn: canonicalize_component(parse_assembly_component(fn)[1])))
        .groupby([col_sup_train, "component_key"])[col_tol_train]
        .min()
        .to_dict()
    )

    # --- feasible normalize
    nQ = len(test_df)
    feasible_list = feasible_to_list(feasible_suppliers_per_query, nQ=nQ)

    rows = []

    for q in range(nQ):
        r = test_df.iloc[q]
        filename = r[col_filename_test]
        demand = float(r[col_demand])
        gt_cost = float(r[col_gt_cost])
        gt_time = float(r[col_gt_time])
        req_tol = float(r[col_req_tol])

        _, comp_raw = parse_assembly_component(filename)
        comp_key = canonicalize_component(comp_raw)

        feas = feasible_list[q]

        # MILP unmet penalty: C_max among active suppliers
        active_costs = [
            supplier_comp_params[(s, comp_key)]["cost"]
            for s in feas
            if (s, comp_key) in supplier_comp_params
        ]
        lambda_unmet_cost = ALPHA_UNMET * max(active_costs) if active_costs else 0.0

        status, obj_val, pred_time, pred_cost, unmet_qty, x_sol = solve_single_query_milp(
            query_idx=q,
            feasible_suppliers=feas,
            supplier_comp_params=supplier_comp_params,
            comp_key=comp_key,
            demand=demand,
            big_lambda_unmet=lambda_unmet_cost,
            objective="cost",
        )

        unmet_qty = max(0.0, float(unmet_qty)) if np.isfinite(unmet_qty) else np.nan

        reject_qty = 0.0
        if (demand > 0) and isinstance(x_sol, dict) and (len(x_sol) > 0):
            reject_qty = compute_tol_reject_qty_exact(
                x_sol=x_sol,
                req_tol=req_tol,
                family_key=comp_key,
                supplier_comp_tol_cap=supplier_comp_tol_cap,
            )

        # Per-unit GT cost (for reference, but we use ratio-based penalty)
        gt_cost_u = (gt_cost / demand) if (demand > 0) else 0.0
        gt_time_u = (gt_time / demand) if (demand > 0) else 0.0

        # Calculate capacity shortage and quality deficiency ratios
        capa_shortage_ratio = (float(unmet_qty) / demand) if (demand > 0 and np.isfinite(unmet_qty)) else 0.0
        capa_shortage_ratio = float(min(1.0, max(0.0, capa_shortage_ratio)))

        qual_deficiency_ratio = (float(reject_qty) / demand) if (demand > 0) else 0.0
        qual_deficiency_ratio = float(min(1.0, max(0.0, qual_deficiency_ratio)))

        # Infeasible qty ratio = (unmet + reject) / demand (combined infeasibility, accurate sum)
        total_infeasible_qty = float(unmet_qty) + float(reject_qty) if np.isfinite(unmet_qty) else float(reject_qty)
        infeasible_qty_ratio = (total_infeasible_qty / demand) if (demand > 0) else 0.0
        infeasible_qty_ratio = float(min(1.0, max(0.0, infeasible_qty_ratio)))

        # Unified penalty using infeasible_qty_ratio: ALPHA_COST * infeasible_ratio * gt_cost
        penalty_infeasible_cost = ALPHA_COST * infeasible_qty_ratio * gt_cost
        penalty_infeasible_time = BETA_TIME * infeasible_qty_ratio * gt_time

        # Effective cost = predicted cost + penalty from infeasibility
        effective_cost = (float(pred_cost) + float(penalty_infeasible_cost)) if np.isfinite(pred_cost) else np.nan
        effective_time = (float(pred_time) + float(penalty_infeasible_time)) if np.isfinite(pred_time) else np.nan

        ratio_pred_cost = (pred_cost / gt_cost) if (gt_cost > 0 and np.isfinite(pred_cost)) else np.nan
        ratio_eff_cost  = (effective_cost / gt_cost) if (gt_cost > 0 and np.isfinite(effective_cost)) else np.nan
        ratio_pred_time = (pred_time / gt_time) if (gt_time > 0 and np.isfinite(pred_time)) else np.nan
        ratio_eff_time  = (effective_time / gt_time) if (gt_time > 0 and np.isfinite(effective_time)) else np.nan

        rows.append({
            "model": model_name,
            "query_idx": q,
            "family": comp_key,
            "infeasible_qty_ratio": infeasible_qty_ratio,
            "capa_shortage_ratio": capa_shortage_ratio,
            "qual_deficiency_ratio": qual_deficiency_ratio,

            "Predicted_Cost": float(pred_cost) if np.isfinite(pred_cost) else np.nan,
            "Effective_Cost": float(effective_cost) if np.isfinite(effective_cost) else np.nan,
            "Predicted_Time": float(pred_time) if np.isfinite(pred_time) else np.nan,
            "Effective_Time": float(effective_time) if np.isfinite(effective_time) else np.nan,

            "ratio_pred_cost": float(ratio_pred_cost) if np.isfinite(ratio_pred_cost) else np.nan,
            "ratio_eff_cost":  float(ratio_eff_cost)  if np.isfinite(ratio_eff_cost)  else np.nan,
            "ratio_pred_time": float(ratio_pred_time) if np.isfinite(ratio_pred_time) else np.nan,
            "ratio_eff_time":  float(ratio_eff_time)  if np.isfinite(ratio_eff_time)  else np.nan,

            "_status": status,
            "_demand": demand,
            "_gt_cost": gt_cost,
            "_gt_time": gt_time,
            "_req_tol": req_tol,
            "_unmet_qty": float(unmet_qty) if np.isfinite(unmet_qty) else np.nan,
            "_reject_qty": float(reject_qty),
        })

    df = pd.DataFrame(rows)

    compact_cols = [
        "query_idx","family","infeasible_qty_ratio",
        "Predicted_Cost","Effective_Cost","Predicted_Time","Effective_Time",
        "ratio_pred_cost","ratio_eff_cost","ratio_pred_time","ratio_eff_time",
        "capa_shortage_ratio","qual_deficiency_ratio",
    ]
    compact = df[compact_cols].copy()

    summary = {
        "Model": model_name,
        "Predicted cost ratio": float(np.nanmean(compact["ratio_pred_cost"].values)),
        "Effective cost ratio": float(np.nanmean(compact["ratio_eff_cost"].values)),
        "Predicted time ratio": float(np.nanmean(compact["ratio_pred_time"].values)),
        "Effective time ratio": float(np.nanmean(compact["ratio_eff_time"].values)),
        "Infeasible qty. ratio": float(np.nanmean(compact["infeasible_qty_ratio"].values)),
        "Capacity shortage (η_dem)": float(np.nanmean(compact["capa_shortage_ratio"].values)),
        "Quality deficiency (η_qual)": float(np.nanmean(compact["qual_deficiency_ratio"].values)),
    }

    return df, compact, summary

def main():
    sys.path.insert(0, str(ROOT / "code/01_supplier_identification"))
    from proposed_supplier_retriever import MultiLabelSupplierRetriever
    train = DATA / "train_embeddings_epoch_010_with_metadata.csv"
    test = DATA / "test_embeddings_epoch_010_with_metadata.csv"
    retriever = MultiLabelSupplierRetriever(train, test, k_for_threshold=11)
    retriever.run_retrieval(verbose=False)
    rows, _, summary = run_stage2_for_model(
        "Proposed", pd.read_csv(train), pd.read_csv(test), retriever.get_feasible_suppliers())
    output = ROOT / "results/02_supplier_optimization"
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "allocation.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)
    print(pd.Series(summary).to_string())

if __name__ == "__main__":
    main()
