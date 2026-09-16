"""Stage 3 reconfiguration; calculation functions retained from the research source."""
import os
import re
import copy
import time
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import pulp

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "code/01_supplier_identification"))
DATA = BASE_DIR / "data/01_supplier_identification/main_split_70_30"
FILES = {"mcss": {"train": DATA / "train_embeddings_epoch_010_with_metadata.csv",
                  "test": DATA / "test_embeddings_epoch_010_with_metadata.csv"}}
NUM_SEEDS = 20
TOL_EPS = 1e-9
ALPHA_LIST = [1.0, 5.0, 10.0, 20.0]
DISRUPTION_RATES = [0.0, 0.2, 0.4, 0.6, 0.8]


def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Columns not found: {candidates}")

def parse_assembly_component(filename: str):
    base = os.path.basename(str(filename))
    stem = re.sub(r"\.(binvox|stl|csv)$", "", base, flags=re.IGNORECASE)
    parts = stem.split("__")[0].split("_")
    return (parts[0], parts[2]) if len(parts) >= 3 else ("Unknown", "Unknown")

def canonicalize_component(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return "leaf" if key in {"leafright", "leafleft"} else key

def build_params_and_capability(train_df: pd.DataFrame):
    col_sup  = find_col(train_df, ["supplier", "Supplier"])
    col_file = find_col(train_df, ["filename", "Filename", "FileName"])
    col_cap  = find_col(train_df, ["Capacity", "capacity", "Quantity (Ea)"])
    col_cost = find_col(train_df, ["Cost", "unit_cost", "GT_cost", "gt_cost"])
    col_time = find_col(train_df, ["Time", "unit_time", "GT_time", "gt_time"])
    col_tol  = find_col(train_df, ["Tolerance", "tolerance"])

    params, tol_caps = {}, {}
    for _, row in train_df.iterrows():
        try:
            sup_id = int(str(row[col_sup]).replace("[", "").replace("]", "").split(",")[0])
            fn = str(row[col_file])
            _, comp_raw = parse_assembly_component(fn)
            key = (sup_id, canonicalize_component(comp_raw))

            if key not in params:
                params[key] = {
                    "capacity": float(row[col_cap]),
                    "cost": float(row[col_cost]),
                    "time": float(row[col_time]),
                }
            else:
                params[key]["capacity"] = max(params[key]["capacity"], float(row[col_cap]))

            tol_val = float(row[col_tol])
            tol_caps[key] = min(tol_caps.get(key, 999.0), tol_val)
        except Exception:
            continue

    return params, tol_caps

def process_single_query(args):
    (i, row_data, feasible_sups, disrupted_params, tol_caps, alpha_val) = args
    demand  = row_data["demand"]
    gt_c    = row_data["gt_c"]
    gt_t    = row_data["gt_t"]
    req_t   = row_data["req_t"]
    comp_key = row_data["comp_key"]

    active = [s for s in feasible_sups if (s, comp_key) in disrupted_params]

    if (not active) or (demand <= 0):
        return float(alpha_val), float(alpha_val)

    def solve_mip(obj_type: str):
        prob = pulp.LpProblem(f"Q{i}_{obj_type}", pulp.LpMinimize)
        x = {s: pulp.LpVariable(f"x_{s}", lowBound=0) for s in active}
        unmet = pulp.LpVariable("unmet", lowBound=0)

        coeffs = {s: disrupted_params[(s, comp_key)][obj_type] for s in active}
        lambda_val = float(alpha_val) * float(max(coeffs.values()))

        prob += pulp.lpSum(coeffs[s] * x[s] for s in active) + lambda_val * unmet
        prob += pulp.lpSum(x[s] for s in active) + unmet == demand
        for s in active:
            prob += x[s] <= disrupted_params[(s, comp_key)]["capacity"]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        x_sol = {s: x[s].value() for s in active if x[s].value() is not None and x[s].value() > 1e-6}
        u_cap = unmet.value() if unmet.value() is not None else demand
        pred_val = sum(coeffs[s] * v for s, v in x_sol.items())

        def tol_exact_match(supplier_id: int) -> bool:
            tol_cap = tol_caps.get((supplier_id, comp_key), None)
            if tol_cap is None:
                return False
            return abs(float(tol_cap) - float(req_t)) <= TOL_EPS

        u_qual = sum(qty for s, qty in x_sol.items() if not tol_exact_match(s))
        return float(pred_val), float(u_cap), float(u_qual)

    p_c, uc_c, uq_c = solve_mip("cost")
    p_t, uc_t, uq_t = solve_mip("time")

    u_gt_c = (gt_c / demand) if demand > 0 else 0.0
    u_gt_t = (gt_t / demand) if demand > 0 else 0.0

    # Effective = (pred + penalty) / GT
    eff_c = (p_c + (uc_c + uq_c) * float(alpha_val) * u_gt_c) / gt_c if gt_c > 0 else float(alpha_val)
    eff_t = (p_t + (uc_t + uq_t) * float(alpha_val) * u_gt_t) / gt_t if gt_t > 0 else float(alpha_val)

    return float(eff_c), float(eff_t)

def run_full_sensitivity_analysis():
    models = ['mcss']
    final_results = []
    try:
        from proposed_supplier_retriever import MultiLabelSupplierRetriever
    except ImportError:
        print('[Error] Retriever files not found.')
        return
    num_cores = cpu_count()
    start_total = time.time()
    for mode in models:
        print(f"\n{'=' * 20}\n[Model: {mode.upper()}] Processing Stage 2 (Retrieval)...\n{'=' * 20}")
        train_df = pd.read_csv(FILES[mode]['train'])
        test_df = pd.read_csv(FILES[mode]['test'])
        retriever = MultiLabelSupplierRetriever(train_csv=FILES[mode]['train'], test_csv=FILES[mode]['test'], k_for_threshold=11)
        retriever.run_retrieval(verbose=False)
        feasible_map = retriever.get_feasible_suppliers()
        base_params, tol_caps = build_params_and_capability(train_df)
        col_dem = find_col(test_df, ['Demand', 'demand'])
        col_gt_c = find_col(test_df, ['GT_Cost', 'gt_cost'])
        col_gt_t = find_col(test_df, ['GT_Time', 'gt_time'])
        col_tol = find_col(test_df, ['Tolerance', 'tolerance'])
        col_fn = find_col(test_df, ['filename', 'Filename', 'FileName'])
        for alpha in ALPHA_LIST:
            print(f'  > Testing ALPHA = {alpha}')
            for rate in DISRUPTION_RATES:
                print(f'    -> Rate {int(rate * 100):2d}%: ', end='', flush=True)
                seeds_c, seeds_t = ([], [])
                for seed_idx in range(NUM_SEEDS):
                    current_seed = 42 + int(rate * 100) + seed_idx
                    disrupted_params = copy.deepcopy(base_params)
                    all_sups = sorted(list(set([k[0] for k in base_params.keys()])))
                    num_failed = int(len(all_sups) * rate)
                    if num_failed > 0:
                        np.random.seed(current_seed)
                        hit_sups = set(np.random.choice(all_sups, num_failed, replace=False))
                        for k in disrupted_params:
                            if k[0] in hit_sups:
                                disrupted_params[k]['capacity'] = 0.0
                    tasks = []
                    for i in range(len(test_df)):
                        row = test_df.iloc[i]
                        _, comp_raw = parse_assembly_component(row[col_fn])
                        row_data = {'demand': float(row[col_dem]), 'gt_c': float(row[col_gt_c]), 'gt_t': float(row[col_gt_t]), 'req_t': float(row[col_tol]), 'comp_key': canonicalize_component(comp_raw)}
                        feas = feasible_map[i]
                        tasks.append((i, row_data, feas, disrupted_params, tol_caps, alpha))
                    with Pool(processes=num_cores) as pool:
                        results = pool.map(process_single_query, tasks)
                    valid_c = [r[0] for r in results if not np.isnan(r[0])]
                    valid_t = [r[1] for r in results if not np.isnan(r[1])]
                    if valid_c:
                        seeds_c.append(float(np.mean(valid_c)))
                    if valid_t:
                        seeds_t.append(float(np.mean(valid_t)))
                    print('.', end='', flush=True)
                final_results.append({'Model': mode, 'Alpha': alpha, 'Disruption_Rate': rate, 'Eff_Cost_Ratio_Mean': float(np.mean(seeds_c)) if seeds_c else np.nan, 'Eff_Cost_Ratio_Std': float(np.std(seeds_c)) if seeds_c else np.nan, 'Eff_Time_Ratio_Mean': float(np.mean(seeds_t)) if seeds_t else np.nan, 'Eff_Time_Ratio_Std': float(np.std(seeds_t)) if seeds_t else np.nan})
                print(f' Done. (Mean Cost: {np.mean(seeds_c):.3f})')
    res_df = pd.DataFrame(final_results)
    out_path = BASE_DIR / 'results' / '03_resilience_analysis' / 'disruption_rate_sensitivity_exact_tolerance.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}\n [Final Analysis Report] \n{'=' * 60}")
    print(f'Total Time: {(time.time() - start_total) / 60:.2f} mins')
    print(f'Results saved to: {out_path}')
    print('\n[Summary: Effective Cost Ratio at ALPHA=10]')
    summary = res_df[res_df['Alpha'] == 10.0].pivot(index='Disruption_Rate', columns='Model', values='Eff_Cost_Ratio_Mean')
    print(summary.round(4))

if __name__ == "__main__":
    run_full_sensitivity_analysis()
