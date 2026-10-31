"""Command-line entry point for LfE study preparation and analyses."""

from __future__ import annotations

import argparse

import preprocessing


def _prepare_if_needed(config_name: str) -> None:
    if preprocessing.prepared_is_current(config_name):
        return
    print("Prepared data are missing or do not match the current configuration; preparing them now.")
    preprocessing.prepare(config_name)


def _run_analysis(name: str, config_name: str) -> None:
    _prepare_if_needed(config_name)
    if name == "adas":
        import adas
        adas.analyze(config_name)
    else:
        import eyetracking
        eyetracking.analyze(config_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lfecli", description="LfE study preparation and analysis")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="Create the compact row-preserving Parquet snapshot")
    prepare.add_argument("--config", default="default.toml", help="TOML file in configs/ (default: default.toml)")
    analyze = subcommands.add_parser("analyze", help="Run analyses, preparing data first when needed")
    analyze.add_argument("target", choices=("adas", "eyetracking", "all"), nargs="?", default="all")
    analyze.add_argument("--config", default="default.toml", help="TOML file in configs/ (default: default.toml)")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        print(f"Prepared data written to {preprocessing.prepare(args.config)}")
        return

    targets = ("adas", "eyetracking") if args.target == "all" else (args.target,)
    for target in targets:
        _run_analysis(target, args.config)


if __name__ == "__main__":
    main()
