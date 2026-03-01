from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .workflows import list_workflows, run_case_study_2_data_generation, run_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph-bd",
        description="Standardized command runner for repository workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available workflows.")

    run_parser = subparsers.add_parser("run", help="Run a named workflow script.")
    run_parser.add_argument("--workflow", required=True, help="Workflow name from `graph-bd list`.")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing."
    )
    run_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying script. Prefix with `--`.",
    )

    cs2_data_parser = subparsers.add_parser(
        "generate-cs2-data",
        help="Generate Case Study 2 graph data for a specific year.",
    )
    cs2_data_parser.add_argument("--year", type=int, required=True)
    cs2_data_parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing."
    )
    cs2_data_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying script. Prefix with `--`.",
    )

    return parser


def _strip_remainder_prefix(values: Sequence[str]) -> list[str]:
    if values and values[0] == "--":
        return list(values[1:])
    return list(values)


def _handle_list() -> int:
    for workflow in list_workflows():
        print(f"{workflow.name:35} {workflow.script} | {workflow.description}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _handle_list()

    if args.command == "run":
        forwarded_args = _strip_remainder_prefix(args.script_args)
        return run_workflow(args.workflow, extra_args=forwarded_args, dry_run=args.dry_run)

    if args.command == "generate-cs2-data":
        forwarded_args = _strip_remainder_prefix(args.script_args)
        return run_case_study_2_data_generation(
            year=args.year,
            extra_args=forwarded_args,
            dry_run=args.dry_run,
        )

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
