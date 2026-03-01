from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CS2_DG_ROOT = ROOT / "case_study_2" / "data_generation_il_stage"


def _run_help(script_path: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_shared_generate_script_help() -> None:
    script = CS2_DG_ROOT / "shared" / "generate_graph_data_shared.py"
    output = _run_help(script)
    assert "--year" in output
    assert "--year-dir" in output
    assert "--output-path" in output


def test_year_generate_wrapper_help_2010() -> None:
    script = CS2_DG_ROOT / "2010" / "generate_graph_data.py"
    output = _run_help(script)
    assert "year 2010" in output.lower()
    assert "--output-path" in output
    assert "--no-plot" in output


def test_year_generate_wrapper_help_2023() -> None:
    script = CS2_DG_ROOT / "2023" / "generate_graph_data.py"
    output = _run_help(script)
    assert "year 2023" in output.lower()
    assert "--output-path" in output
    assert "--no-plot" in output
