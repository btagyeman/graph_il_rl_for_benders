from __future__ import annotations

import pytest

from graph_il_rl_for_benders.config_schemas import (
    validate_cs1_train_il_config,
    validate_cs1_train_rl_config,
    validate_cs2_train_il_config,
    validate_cs2_train_rl_config,
)


def test_cs1_train_il_rejects_unknown_key() -> None:
    payload = {
        "dataset_path": "./datasets/graph_gbd_dataset_3k_improved.pkl",
        "output_dir": "saved_models",
        "batch_size": 128,
        "epochs": 100,
        "train_split": 0.95,
        "threshold_high": 0.95,
        "threshold_low": 0.05,
        "seed": 42,
        "no_plots": False,
        "extra": 123,
    }
    with pytest.raises(ValueError, match="unknown config keys"):
        validate_cs1_train_il_config(payload)


def test_cs1_train_il_rejects_inverted_thresholds() -> None:
    payload = {
        "dataset_path": "./datasets/graph_gbd_dataset_3k_improved.pkl",
        "output_dir": "saved_models",
        "batch_size": 128,
        "epochs": 100,
        "train_split": 0.95,
        "threshold_high": 0.2,
        "threshold_low": 0.8,
        "seed": 42,
        "no_plots": False,
    }
    with pytest.raises(ValueError, match="threshold_low must be <= threshold_high"):
        validate_cs1_train_il_config(payload)


def test_cs1_train_rl_allows_null_episodes() -> None:
    payload = {
        "parameter_file": "./parameter_verification/generated_parameters.txt",
        "output_dir": "./results",
        "episodes": None,
        "batch_size": 32,
        "max_steps_per_episode": 1000,
        "save_every": 500,
        "seed": 42,
        "no_live_plot": False,
    }
    validate_cs1_train_rl_config(payload)


def test_cs2_train_il_rejects_invalid_years() -> None:
    payload = {
        "dataset_dir": "./gbd_datasets",
        "output_dir": "./results",
        "years": "1999,2000",
        "batch_size": 128,
        "epochs": 100,
        "train_split": 0.85,
        "threshold_high": 0.8,
        "threshold_low": 0.2,
        "seed": 42,
        "no_plots": False,
    }
    with pytest.raises(ValueError, match="invalid type/value"):
        validate_cs2_train_il_config(payload)


def test_cs2_train_rl_rejects_negative_epsilon() -> None:
    payload = {
        "common_dir": "./common",
        "model_weights_dir": "./trained_lstm_models/model_weights",
        "output_dir": "./results",
        "episodes": 5000,
        "batch_size": 32,
        "max_steps_per_episode": 50,
        "epsilon": -0.1,
        "save_every": 100,
        "seed": 42,
        "no_live_plot": False,
    }
    with pytest.raises(ValueError, match="must be >="):
        validate_cs2_train_rl_config(payload)
