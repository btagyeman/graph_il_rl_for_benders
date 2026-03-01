from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
            return
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"Refusing to overwrite directory: {link_path}")
        link_path.unlink()
    resolved_target = target_path.resolve()
    if not resolved_target.exists():
        raise FileNotFoundError(f"Target does not exist: {resolved_target}")
    relative_link_target = Path(os.path.relpath(resolved_target, start=link_path.parent))
    link_path.symlink_to(relative_link_target)


def main() -> int:
    links = {
        "data/case_study_1/datasets": "case_study_1/data_generation_and_il_stage/datasets",
        "data/case_study_1/scaling_data": "case_study_1/comparative_study/scaling_data",
        "data/case_study_1/test_parameters": "case_study_1/comparative_study/test_parameters",
        "data/case_study_2/il_stage_gbd_datasets": "case_study_2/il_stage/gbd_datasets",
        "data/case_study_2/comparative_gbd_datasets": "case_study_2/comparative_study/gbd/gbd_datasets",
        "data/case_study_2/weather_generation": "case_study_2/weather_data_generation",
        "data/case_study_2/rl_common_weather_data": "case_study_2/rl_stage/common/weather_data",
        "data/case_study_2/rl_common_initial_states": "case_study_2/rl_stage/common/initial_states",
        "artifacts/trained_agents/case_study_1/comparative_gbd_agents": "case_study_1/comparative_study/gbd_agents",
        "artifacts/trained_agents/case_study_1/rl_il_agent": "case_study_1/rl_stage/proposed_approach/il_agent",
        "artifacts/trained_agents/case_study_2/rl_il_agent": "case_study_2/rl_stage/il_agent",
        "artifacts/trained_agents/case_study_2/comparative_proposed_trained_agent": "case_study_2/comparative_study/proposed/trained_agent",
        "artifacts/model_weights/case_study_2_lstm": "case_study_2/trained_lstm_models/model_weights",
        "case_study_2/il_stage/graph_datasets": "case_study_2/il_stage/gbd_datasets",
        "case_study_2/comparative_study/gbd/graph_datasets": "case_study_2/comparative_study/gbd/gbd_datasets",
        "case_study_1/data_generation_and_il_stage/utils": "case_study_1/data_generation_and_il_stage/helper_functions",
        "case_study_1/comparative_study/utils": "case_study_1/comparative_study/helper_functions",
        "case_study_1/rl_stage/proposed_approach/utils": "case_study_1/rl_stage/proposed_approach/helper_functions",
        "case_study_1/rl_stage/proposed_approach/agent": "case_study_1/rl_stage/proposed_approach/core_agent",
        "case_study_1/rl_stage/random_actor/utils": "case_study_1/rl_stage/random_actor/helper_functions",
        "case_study_1/rl_stage/random_actor/agent": "case_study_1/rl_stage/random_actor/core_agent",
        "case_study_2/rl_stage/utils": "case_study_2/rl_stage/helper_functions",
        "case_study_2/rl_stage/agent": "case_study_2/rl_stage/core_agent",
        "case_study_2/comparative_study/proposed/utils": "case_study_2/comparative_study/proposed/helper_functions",
        "case_study_2/comparative_study/gbd/utils": "case_study_2/comparative_study/gbd/helper_functions",
        "case_study_2/data_generation_il_stage/shared/utils": "case_study_2/data_generation_il_stage/shared/helper_functions",
        "case_study_2/pipelines/il": "case_study_2/il_stage",
        "case_study_2/pipelines/rl": "case_study_2/rl_stage",
        "case_study_2/pipelines/comparative": "case_study_2/comparative_study",
        "case_study_2/pipelines/il_data_generation": "case_study_2/data_generation_il_stage",
        "case_study_2/shared/environments": "case_study_2/data_generation_il_stage/shared/common/environments",
        "case_study_2/shared/estimators": "case_study_2/data_generation_il_stage/shared/common/estimator_design",
        "case_study_2/shared/simulation": "case_study_2/data_generation_il_stage/shared/common/actual_field",
        "case_study_2/shared/optimization": "case_study_2/data_generation_il_stage/shared/helper_functions",
        "case_study_2/data/graph_datasets/il": "case_study_2/il_stage/gbd_datasets",
        "case_study_2/data/graph_datasets/comparative": "case_study_2/comparative_study/gbd/gbd_datasets",
        "case_study_2/data/weather": "case_study_2/rl_stage/common/weather_data",
        "case_study_2/data/initial_states": "case_study_2/rl_stage/common/initial_states",
        "case_study_2/models/il_agent": "case_study_2/rl_stage/il_agent",
        "case_study_2/models/lstm_weights": "case_study_2/trained_lstm_models/model_weights",
        "case_study_1/pipelines/il": "case_study_1/data_generation_and_il_stage",
        "case_study_1/pipelines/rl": "case_study_1/rl_stage",
        "case_study_1/pipelines/comparative": "case_study_1/comparative_study",
        "case_study_1/shared/optimization": "case_study_1/data_generation_and_il_stage/helper_functions",
        "case_study_1/shared/agent": "case_study_1/rl_stage/proposed_approach/core_agent",
        "case_study_1/data/datasets": "case_study_1/data_generation_and_il_stage/datasets",
        "case_study_1/data/scaling_data": "case_study_1/comparative_study/scaling_data",
        "case_study_1/data/test_parameters": "case_study_1/comparative_study/test_parameters",
        "case_study_1/models/il_agent": "case_study_1/rl_stage/proposed_approach/il_agent",
        "case_study_1/models/comparative_gbd_agents": "case_study_1/comparative_study/gbd_agents",
    }

    for link_rel, target_rel in links.items():
        ensure_symlink(REPO_ROOT / link_rel, REPO_ROOT / target_rel)

    print(f"Created/verified {len(links)} layout symlinks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
