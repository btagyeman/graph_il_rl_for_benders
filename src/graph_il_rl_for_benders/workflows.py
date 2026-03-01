from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .paths import resolve_repo_path


@dataclass(frozen=True)
class Workflow:
    name: str
    script: Path
    description: str


def _prefer_path(preferred_rel: str, fallback_rel: str) -> Path:
    preferred = resolve_repo_path(preferred_rel)
    if preferred.exists():
        return preferred
    return resolve_repo_path(fallback_rel)


WORKFLOWS: dict[str, Workflow] = {
    "cs1_generate_training_data": Workflow(
        name="cs1_generate_training_data",
        script=_prefer_path(
            "case_study_1/pipelines/il/generate_training_data.py",
            "case_study_1/data_generation_and_il_stage/generate_training_data.py",
        ),
        description="Generate graph dataset for Case Study 1.",
    ),
    "cs1_train_il": Workflow(
        name="cs1_train_il",
        script=_prefer_path(
            "case_study_1/pipelines/il/train_il_agent.py",
            "case_study_1/data_generation_and_il_stage/train_il_agent.py",
        ),
        description="Train imitation-learning agent for Case Study 1.",
    ),
    "cs1_train_rl": Workflow(
        name="cs1_train_rl",
        script=_prefer_path(
            "case_study_1/pipelines/rl/proposed_approach/train_rl_agent.py",
            "case_study_1/rl_stage/proposed_approach/train_rl_agent.py",
        ),
        description="Train PPO agent for Case Study 1.",
    ),
    "cs1_evaluate_comparative": Workflow(
        name="cs1_evaluate_comparative",
        script=_prefer_path(
            "case_study_1/pipelines/comparative/comparative_study.py",
            "case_study_1/comparative_study/comparative_study.py",
        ),
        description="Run comparative evaluation for Case Study 1.",
    ),
    "cs2_train_il": Workflow(
        name="cs2_train_il",
        script=_prefer_path(
            "case_study_2/pipelines/il/train_il_agent.py",
            "case_study_2/il_stage/train_il_agent.py",
        ),
        description="Train imitation-learning agent for Case Study 2.",
    ),
    "cs2_train_rl": Workflow(
        name="cs2_train_rl",
        script=_prefer_path(
            "case_study_2/pipelines/rl/train_rl_agent.py",
            "case_study_2/rl_stage/train_rl_agent.py",
        ),
        description="Train PPO agent for Case Study 2.",
    ),
    "cs2_generate_evaluation_data": Workflow(
        name="cs2_generate_evaluation_data",
        script=_prefer_path(
            "case_study_2/pipelines/comparative/proposed/generate_graph_data.py",
            "case_study_2/comparative_study/proposed/generate_graph_data.py",
        ),
        description="Generate graph data used for Case Study 2 comparative evaluation.",
    ),
}

WORKFLOW_ALIASES: dict[str, str] = {
    "cs1_comparative_study": "cs1_evaluate_comparative",
    "cs2_prepare_evaluation_data": "cs2_generate_evaluation_data",
    "cs2_comparative_generate_graph_data": "cs2_generate_evaluation_data",
}


SUPPORTED_CASE_STUDY_2_YEARS = tuple(range(2009, 2024))


def list_workflows() -> list[Workflow]:
    return [WORKFLOWS[key] for key in sorted(WORKFLOWS)]


def get_workflow(name: str) -> Workflow:
    resolved_name = WORKFLOW_ALIASES.get(name, name)
    if resolved_name not in WORKFLOWS:
        available = ", ".join(sorted(WORKFLOWS))
        aliases = ", ".join(
            f"{legacy}->{canonical}" for legacy, canonical in sorted(WORKFLOW_ALIASES.items())
        )
        raise KeyError(
            f"Unknown workflow '{name}'. Available workflows: {available}. "
            f"Legacy aliases: {aliases}"
        )
    return WORKFLOWS[resolved_name]


def get_case_study_2_yearly_script(year: int) -> Path:
    if year not in SUPPORTED_CASE_STUDY_2_YEARS:
        raise ValueError(
            f"Year {year} is not supported. Expected one of {SUPPORTED_CASE_STUDY_2_YEARS}."
        )
    return _prefer_path(
        f"case_study_2/pipelines/il_data_generation/{year}/generate_graph_data.py",
        f"case_study_2/data_generation_il_stage/{year}/generate_graph_data.py",
    )


def _run_script(
    script_path: Path, extra_args: Optional[Sequence[str]] = None, dry_run: bool = False
) -> int:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    args = [sys.executable, script_path.name]
    if extra_args:
        args.extend(extra_args)

    command_text = " ".join(args)
    print(f"Working directory: {script_path.parent}")
    print(f"Command: {command_text}")

    if dry_run:
        return 0

    child_env = os.environ.copy()
    child_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    child_env.setdefault("ABSL_LOGGING_VERBOSITY", "error")
    child_env.setdefault(
        "PYTHONWARNINGS",
        "ignore:Custom mask layers require a config and must override get_config:UserWarning",
    )

    completed = subprocess.run(args=args, cwd=script_path.parent, check=False, env=child_env)
    return completed.returncode


def run_workflow(
    name: str, extra_args: Optional[Sequence[str]] = None, dry_run: bool = False
) -> int:
    workflow = get_workflow(name)
    return _run_script(workflow.script, extra_args=extra_args, dry_run=dry_run)


def run_case_study_2_data_generation(
    year: int,
    extra_args: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> int:
    script = get_case_study_2_yearly_script(year)
    return _run_script(script, extra_args=extra_args, dry_run=dry_run)
