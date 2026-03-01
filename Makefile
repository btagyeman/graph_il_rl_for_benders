.PHONY: install install-research lint format test workflows layout

install:
	pip install -e ".[dev]"

install-research:
	pip install -e ".[dev,research]"

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

test:
	pytest

workflows:
	python -m graph_il_rl_for_benders.cli list

layout:
	python scripts/setup_standard_layout.py
