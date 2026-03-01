from __future__ import annotations

from typing import Any, Callable, Dict

SchemaField = Dict[str, Any]
Schema = Dict[str, SchemaField]


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float_like(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_path_string(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0


def _is_years_spec(value: Any) -> bool:
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return True
        try:
            years = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError:
            return False
    elif isinstance(value, list):
        if not all(_is_int(item) for item in value):
            return False
        years = value
    else:
        return False

    if not years:
        return False
    return all(2009 <= year <= 2023 for year in years)


def _type_name(checker: Callable[[Any], bool]) -> str:
    return checker.__name__.replace("_is_", "")


def validate_config_payload(payload: dict[str, Any], schema: Schema, schema_name: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{schema_name}: config payload must be a JSON object.")

    unknown_keys = set(payload) - set(schema)
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"{schema_name}: unknown config keys: {keys}")

    for key, spec in schema.items():
        required = bool(spec.get("required", False))
        allow_none = bool(spec.get("allow_none", False))
        checker = spec.get("checker")
        min_value = spec.get("min")
        max_value = spec.get("max")

        if required and key not in payload:
            raise ValueError(f"{schema_name}: missing required key '{key}'")

        if key not in payload:
            continue

        value = payload[key]
        if value is None:
            if allow_none:
                continue
            raise ValueError(f"{schema_name}: key '{key}' cannot be null")

        if checker is not None and not checker(value):
            expected = _type_name(checker)
            raise ValueError(f"{schema_name}: key '{key}' has invalid type/value (expected {expected})")

        if min_value is not None and _is_float_like(value) and value < min_value:
            raise ValueError(f"{schema_name}: key '{key}' must be >= {min_value}")

        if max_value is not None and _is_float_like(value) and value > max_value:
            raise ValueError(f"{schema_name}: key '{key}' must be <= {max_value}")


CS1_TRAIN_IL_SCHEMA: Schema = {
    "dataset_path": {"required": True, "checker": _is_path_string},
    "output_dir": {"required": True, "checker": _is_path_string},
    "batch_size": {"required": True, "checker": _is_int, "min": 1},
    "epochs": {"required": True, "checker": _is_int, "min": 1},
    "train_split": {"required": True, "checker": _is_float_like, "min": 0.01, "max": 0.99},
    "threshold_high": {"required": True, "checker": _is_float_like, "min": 0.0, "max": 1.0},
    "threshold_low": {"required": True, "checker": _is_float_like, "min": 0.0, "max": 1.0},
    "seed": {"required": True, "checker": _is_int, "min": 0},
    "no_plots": {"required": True, "checker": _is_bool},
}

CS1_TRAIN_RL_SCHEMA: Schema = {
    "parameter_file": {"required": True, "checker": _is_path_string},
    "output_dir": {"required": True, "checker": _is_path_string},
    "episodes": {"required": True, "checker": _is_int, "allow_none": True, "min": 1},
    "batch_size": {"required": True, "checker": _is_int, "min": 1},
    "max_steps_per_episode": {"required": True, "checker": _is_int, "min": 1},
    "save_every": {"required": True, "checker": _is_int, "min": 1},
    "seed": {"required": True, "checker": _is_int, "min": 0},
    "no_live_plot": {"required": True, "checker": _is_bool},
}

CS2_TRAIN_IL_SCHEMA: Schema = {
    "dataset_dir": {"required": True, "checker": _is_path_string},
    "output_dir": {"required": True, "checker": _is_path_string},
    "years": {"required": True, "checker": _is_years_spec},
    "batch_size": {"required": True, "checker": _is_int, "min": 1},
    "epochs": {"required": True, "checker": _is_int, "min": 1},
    "train_split": {"required": True, "checker": _is_float_like, "min": 0.01, "max": 0.99},
    "threshold_high": {"required": True, "checker": _is_float_like, "min": 0.0, "max": 1.0},
    "threshold_low": {"required": True, "checker": _is_float_like, "min": 0.0, "max": 1.0},
    "seed": {"required": True, "checker": _is_int, "min": 0},
    "no_plots": {"required": True, "checker": _is_bool},
}

CS2_TRAIN_RL_SCHEMA: Schema = {
    "common_dir": {"required": True, "checker": _is_path_string},
    "model_weights_dir": {"required": True, "checker": _is_path_string},
    "output_dir": {"required": True, "checker": _is_path_string},
    "episodes": {"required": True, "checker": _is_int, "min": 1},
    "batch_size": {"required": True, "checker": _is_int, "min": 1},
    "max_steps_per_episode": {"required": True, "checker": _is_int, "min": 1},
    "epsilon": {"required": True, "checker": _is_float_like, "min": 0.0},
    "save_every": {"required": True, "checker": _is_int, "min": 1},
    "seed": {"required": True, "checker": _is_int, "min": 0},
    "no_live_plot": {"required": True, "checker": _is_bool},
}


def validate_cs1_train_il_config(payload: dict[str, Any]) -> None:
    validate_config_payload(payload, CS1_TRAIN_IL_SCHEMA, "case_study_1.train_il")
    if payload["threshold_low"] > payload["threshold_high"]:
        raise ValueError("case_study_1.train_il: threshold_low must be <= threshold_high")


def validate_cs1_train_rl_config(payload: dict[str, Any]) -> None:
    validate_config_payload(payload, CS1_TRAIN_RL_SCHEMA, "case_study_1.train_rl")


def validate_cs2_train_il_config(payload: dict[str, Any]) -> None:
    validate_config_payload(payload, CS2_TRAIN_IL_SCHEMA, "case_study_2.train_il")
    if payload["threshold_low"] > payload["threshold_high"]:
        raise ValueError("case_study_2.train_il: threshold_low must be <= threshold_high")


def validate_cs2_train_rl_config(payload: dict[str, Any]) -> None:
    validate_config_payload(payload, CS2_TRAIN_RL_SCHEMA, "case_study_2.train_rl")
