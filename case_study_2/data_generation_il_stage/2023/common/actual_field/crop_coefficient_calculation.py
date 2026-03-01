from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
for _parent in _CURRENT_FILE.parents:
    if _parent.parent.name == "data_generation_il_stage":
        _YEAR_DIR = _parent
        break
else:
    raise RuntimeError("Could not resolve year directory for shared-module forwarding")

_SHARED_FILE = _YEAR_DIR.parent / "shared" / "common/actual_field/crop_coefficient_calculation.py"
exec(compile(_SHARED_FILE.read_text(encoding="utf-8"), str(_SHARED_FILE), "exec"))
