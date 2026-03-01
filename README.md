# Graph-Based Imitation and Reinforcement Learning for Efficient Benders Decomposition

This repository contains the implementation for:

**"Graph-Based Imitation and Reinforcement Learning for Efficient Benders Decomposition"**  
Authors: Bernard T. Agyeman, Zhe Li, Ilias Mitrai, Prodromos Daoutidis

The framework combines:
- Graph representation of the evolving Benders master problem.
- Graph neural policy pretraining with imitation learning (IL).
- PPO fine-tuning with reinforcement learning (RL).
- Verification/confidence logic for robust master-problem assignments.


## Repository Structure

```text
.
|-- src/graph_il_rl_for_benders/     # Package layer + CLI
|-- tests/                           # Automated checks
|-- scripts/                         # Utility scripts
|-- configs/                         # JSON configs for workflows
|-- data/                            # Standardized data entrypoints
|-- artifacts/                       # Standardized model/agent entrypoints
|-- case_study_1/                    # Case study 1 (MINLP)
|-- case_study_2/                    # Case study 2 (irrigation scheduling)
|-- pyproject.toml                   # Packaging + tooling config
`-- README.md
```

## Setup

Python requirement: `>=3.8,<3.10`.

```bash
python3.8 -m venv .venv38
source .venv38/bin/activate
pip install -e ".[dev]"
```

For full research dependencies:

```bash
pip install -e ".[dev,research]"
```

Create/refresh standardized symlink layout:

```bash
make layout
```

## Case Study 1 Dataset Prerequisite

Case Study 1 IL training expects a real pickle dataset file. The checked-in
`graph_gbd_dataset_3k_improved.pkl` may be a Git LFS pointer on fresh clones.

Use one of these options before `cs1_train_il`:

```bash
# Option A: generate locally (recommended if LFS data is unavailable)
graph-bd run --workflow cs1_generate_training_data
```

```bash
# Option B: pull LFS-backed dataset (if using git-lfs)
git lfs pull --include="case_study_1/data_generation_and_il_stage/datasets/graph_gbd_dataset_3k_improved.pkl"
```

## Run Workflows

Use absolute paths for `--config` and `--dataset-path` when possible. Workflow
scripts execute from case-study subdirectories, so relative paths are resolved
from those working directories.

List workflows:

```bash
graph-bd list
```

Run standard training workflows:

```bash
graph-bd run --workflow cs1_train_il
graph-bd run --workflow cs1_train_rl
graph-bd run --workflow cs2_train_il
graph-bd run --workflow cs2_train_rl
```

Run evaluation workflows:

```bash
graph-bd run --workflow cs1_evaluate_comparative
graph-bd run --workflow cs2_generate_evaluation_data
```

Run with config files:

```bash
graph-bd run --workflow cs1_train_il -- --config /abs/path/to/configs/case_study_1/train_il.json --dataset-path /abs/path/to/case_study_1/data_generation_and_il_stage/datasets/graph_gbd_dataset_3k.pkl
graph-bd run --workflow cs1_train_rl -- --config /abs/path/to/configs/case_study_1/train_rl.json --episodes 1000 --no-live-plot
graph-bd run --workflow cs2_train_il -- --config /abs/path/to/configs/case_study_2/train_il.json --years 2021,2022
graph-bd run --workflow cs2_train_rl -- --config /abs/path/to/configs/case_study_2/train_rl.json --no-live-plot
```

Generate yearly Case Study 2 IL data:

```bash
graph-bd generate-cs2-data --year 2018
```

Dry run:

```bash
graph-bd run --workflow cs1_train_il --dry-run
```

Quick smoke test:

```bash
graph-bd run --workflow cs1_train_rl -- --config /abs/path/to/configs/case_study_1/train_rl.json --episodes 10 --save-every 10 --no-live-plot --output-dir ./results_smoke
graph-bd run --workflow cs1_evaluate_comparative
```

Canonical case-study layout and migration notes:
[docs/modernization_plan.md](/home/btagyeman/Desktop/graph_il_rl_for_benders/docs/modernization_plan.md)

## Data and Artifacts

- Data layout details: [data/README.md](/home/btagyeman/Desktop/graph_il_rl_for_benders/data/README.md)
- Artifacts layout details: [artifacts/README.md](/home/btagyeman/Desktop/graph_il_rl_for_benders/artifacts/README.md)

## Development

```bash
pre-commit install
make lint
make test
```

Modernization/refactor log: [docs/modernization_plan.md](/home/btagyeman/Desktop/graph_il_rl_for_benders/docs/modernization_plan.md)
