# Manufacturing Capability-Aware Decision-Making Framework

Code and input data accompanying the study:

Suyoung Park and Shreyes N. Melkote,
[A manufacturing capability-aware supplier decision framework for resilient production allocation and disruption response](https://doi.org/10.1016/j.jmsy.2026.09.009),
*Journal of Manufacturing Systems* (2026).

The workflow connects multimodal representation learning, supplier
identification, production allocation, and disruption-responsive reconfiguration.

## Repository contents

```text
code/
  00_training/                   Model, training, and embedding extraction
  01_supplier_identification/    Proposed supplier retrieval
  02_supplier_optimization/      Production allocation
  03_resilience_analysis/        Capacity loss and degradation scenarios
  common/                       Validated joins of embeddings and scenario inputs
data/
  01_supplier_identification/main_split_70_30/
                                Training and query input metadata
  02_allocation_inputs/          Supplier capacities, demand, costs, and targets
  voxel_geometry/                2,147 input BINVOX geometries
```

This repository provides the proposed method's core workflow and input data.
The checkpoint used for the paper and saved experimental results are not
included. Running the workflow generates checkpoints, embeddings, and results
locally.

## Setup

```bash
python -m pip install -r requirements.txt
```

CUDA is recommended for training the 128-cubed voxel model. The allocation and
reconfiguration scripts use PuLP's CBC solver; verify that
`pulp.PULP_CBC_CMD().available()` returns a solver path in your environment.

## Run the workflow

Run commands from the repository root. Start the two training notebooks with
Jupyter, in the order shown below.

1. `code/00_training/01_train_multimodal_autoencoder.ipynb`
2. `code/00_training/02_extract_supplier_embeddings.ipynb`

The first notebook trains for ten epochs and writes local checkpoints. The
second uses the epoch-10 checkpoint and writes 48-dimensional embeddings.
The quantity input is fixed at zero during training and embedding extraction.

Then run:

```bash
python code/01_supplier_identification/01_identify_suppliers.py
python code/common/prepare_allocation_inputs.py
python code/02_supplier_optimization/01_optimize_supplier_selection.py
python code/03_resilience_analysis/02_analyze_disruption_rate.py
python code/03_resilience_analysis/03_analyze_capacity_degradation.py
```

The default Stage 1 threshold parameter is `k=11`. Stage 3 uses 20 random seeds
per scenario and runs with multiprocessing.

## Input data and sample identity

The main split contains 8,976 training/database rows and 3,906 test/query rows.
Each row references a BINVOX file; geometries can appear in several rows with
different manufacturing attributes.

The input CSV's `Supplier` field can contain paired feasible supplier labels.
`train_allocation_metadata.csv` records the assigned supplier for each database
row, and `test_allocation_metadata.csv` records query demand and ground-truth
reference targets. These metadata tables are shared by Stage 2 and Stage 3.

Embeddings and allocation metadata are joined by `sample_id`. Preserve the
supplied row order and IDs when adapting the dataset.

### Voxel geometry data

The voxel geometry files are included in `data/voxel_geometry/` and are also
available from the [Georgia Tech Digital Repository](https://repository.gatech.edu/entities/publication/ebb00f26-5572-41c5-b26c-388a33043a52).

## Calculation behavior

Stage 1 assesses feasibility for each supplier by comparing the query embedding
with that supplier's historical production embeddings. The same global
Euclidean-distance threshold, computed from the training embeddings, is
applied across suppliers to identify candidates. If no training record falls
within this threshold, retrieval uses the nearest record as a fallback.

Stage 2 minimizes cost and reports the cost and time of the resulting
allocation. Stage 3 solves cost and time objectives separately under the
specified capacity disruptions.

Allocation optimization penalizes unmet demand using the configured weight
times the largest active supplier unit cost or time. Post-allocation evaluation
uses its configured weight times the ground-truth value per unit demand to
penalize unmet and quality-deficient quantities. Effective ratios combine the
nominal allocation cost or time with these evaluation penalties and divide by
the corresponding ground-truth total.

## License

The combined software is distributed under GPL-3.0-or-later because it includes
`binvox_io.py` from binvox-rw-py. Original project code also retains its MIT
permission. See `LICENSE`, `LICENSES/MIT.txt`, and `THIRD_PARTY_NOTICES.md` for
license terms, attribution, and scope.
