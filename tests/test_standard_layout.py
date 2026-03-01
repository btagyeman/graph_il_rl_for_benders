from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_LINKS = {
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


def test_standard_layout_script_builds_expected_links() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "setup_standard_layout.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    for link_rel, target_rel in EXPECTED_LINKS.items():
        link_path = ROOT / link_rel
        target_path = ROOT / target_rel
        assert link_path.is_symlink(), f"Expected symlink: {link_path}"
        assert target_path.exists(), f"Expected target missing: {target_path}"
        assert link_path.resolve() == target_path.resolve(), f"Incorrect target for {link_path}"
