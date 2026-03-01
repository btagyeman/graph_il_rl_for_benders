from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CS2_DG_ROOT = ROOT / "case_study_2" / "data_generation_il_stage"
YEARS = list(range(2009, 2024))


def _python_files_under(directory: Path) -> set[str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/")
        for path in directory.rglob("*.py")
        if path.is_file()
    }


def test_shared_module_file_set_matches_each_year() -> None:
    shared_set = _python_files_under(CS2_DG_ROOT / "shared")
    expected_set = {
        item
        for item in shared_set
        if item.startswith("common/") or item.startswith("helper_functions/")
    }

    for year in YEARS:
        year_dir = CS2_DG_ROOT / str(year)
        year_set = _python_files_under(year_dir)
        actual_set = {
            item
            for item in year_set
            if item.startswith("common/") or item.startswith("helper_functions/")
        }
        assert actual_set == expected_set, f"Mismatch in forwarded files for year {year}"


def test_year_module_wrappers_forward_to_shared_paths() -> None:
    shared_set = _python_files_under(CS2_DG_ROOT / "shared")
    target_set = [
        item
        for item in shared_set
        if item.startswith("common/") or item.startswith("helper_functions/")
    ]

    for year in YEARS:
        year_dir = CS2_DG_ROOT / str(year)
        for rel_path in target_set:
            wrapper = year_dir / rel_path
            assert wrapper.exists(), f"Missing wrapper file: {wrapper}"
            content = wrapper.read_text(encoding="utf-8")
            expected_fragment = f'_SHARED_FILE = _YEAR_DIR.parent / "shared" / "{rel_path}"'
            assert "exec(compile(" in content, f"Missing forwarding exec in {wrapper}"
            assert expected_fragment in content, f"Shared target mismatch in {wrapper}"


def test_generate_graph_data_wrapper_year_binding() -> None:
    for year in YEARS:
        wrapper = CS2_DG_ROOT / str(year) / "generate_graph_data.py"
        content = wrapper.read_text(encoding="utf-8")
        assert "from generate_graph_data_shared import main_for_year" in content
        assert f"default_year={year}" in content
