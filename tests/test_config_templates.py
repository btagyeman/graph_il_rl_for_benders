from __future__ import annotations

import json
from pathlib import Path

from graph_il_rl_for_benders.config_schemas import (
    validate_cs1_train_il_config,
    validate_cs1_train_rl_config,
    validate_cs2_train_il_config,
    validate_cs2_train_rl_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    assert isinstance(payload, dict)
    return payload


def test_case_study_1_train_il_template() -> None:
    payload = _load(ROOT / "configs" / "case_study_1" / "train_il.json")
    required = {
        "dataset_path",
        "output_dir",
        "batch_size",
        "epochs",
        "train_split",
        "threshold_high",
        "threshold_low",
        "seed",
        "no_plots",
    }
    assert required.issubset(payload.keys())
    validate_cs1_train_il_config(payload)


def test_case_study_1_train_rl_template() -> None:
    payload = _load(ROOT / "configs" / "case_study_1" / "train_rl.json")
    required = {
        "parameter_file",
        "output_dir",
        "episodes",
        "batch_size",
        "max_steps_per_episode",
        "save_every",
        "seed",
        "no_live_plot",
    }
    assert required.issubset(payload.keys())
    validate_cs1_train_rl_config(payload)


def test_case_study_2_train_il_template() -> None:
    payload = _load(ROOT / "configs" / "case_study_2" / "train_il.json")
    required = {
        "dataset_dir",
        "output_dir",
        "years",
        "batch_size",
        "epochs",
        "train_split",
        "threshold_high",
        "threshold_low",
        "seed",
        "no_plots",
    }
    assert required.issubset(payload.keys())
    validate_cs2_train_il_config(payload)


def test_case_study_2_train_rl_template() -> None:
    payload = _load(ROOT / "configs" / "case_study_2" / "train_rl.json")
    required = {
        "common_dir",
        "model_weights_dir",
        "output_dir",
        "episodes",
        "batch_size",
        "max_steps_per_episode",
        "epsilon",
        "save_every",
        "seed",
        "no_live_plot",
    }
    assert required.issubset(payload.keys())
    validate_cs2_train_rl_config(payload)
