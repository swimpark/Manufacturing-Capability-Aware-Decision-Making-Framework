# Manufacturing Capability-Aware Decision-Making Framework

Core implementation of the proposed method: multimodal representation learning,
supplier identification, production allocation, and disruption-responsive
reconfiguration.

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

This release contains the proposed method's core workflow. Benchmark methods,
parameter sweeps, robustness experiments, and figure-generation code are outside
this release. Trained weights, cached tensors, generated embeddings, and saved
experimental results are excluded. Running the code creates local outputs that
are ignored by Git.

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
The model architecture, training loss, and quantity-fixed-at-zero input are
retained from the research implementation.

Then run:

```bash
python code/01_supplier_identification/01_identify_suppliers.py
python code/common/prepare_allocation_inputs.py
python code/02_supplier_optimization/01_optimize_supplier_selection.py
python code/03_resilience_analysis/02_analyze_disruption_rate.py
python code/03_resilience_analysis/03_analyze_capacity_degradation.py
```

The default proposed-method threshold is `k=11`. Stage 3 uses 20 random seeds
for each scenario. Its scripts use multiprocessing and should be run as Python
scripts, on a compute node with sufficient allocated CPU resources.

## Input data and sample identity

The main split contains 8,976 training/database rows and 3,906 test/query rows.
Each row references a BINVOX file; geometries can appear in several rows with
different manufacturing attributes.

The input CSV's `Supplier` field can contain paired feasible supplier labels.
`train_allocation_metadata.csv` records the assigned supplier for each database
row, and `test_allocation_metadata.csv` records query demand and ground-truth
reference targets. These are scenario inputs, extracted without embeddings or
predictions from the original experiment input tables. The training and test
tables are shared by Stage 2 and Stage 3.

Embedding extraction retains a `sample_id` for every input row. The preparation
script joins by this ID and checks filenames and supplier labels. It fails if
rows are missing or identities disagree. Keep the supplied input row order and
IDs consistent when adapting the dataset.

## Calculation behavior

Stage 2 minimizes cost and reports the cost and time of that allocation. Its
effective scores include capacity-shortage and quality-deficiency penalties;
cost/time means omit undefined values in the original implementation. Stage 3
solves cost and time objectives separately. The solver and scoring functions
are preserved from the research source.

Re-training produces new weights and embeddings; this repository does not
distribute the checkpoint used to obtain the reported paper results.
