from pathlib import Path
import sys

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from generate_graph_data_shared import main_for_year


if __name__ == "__main__":
    raise SystemExit(main_for_year(default_year=2016, year_dir=Path(__file__).resolve().parent))
