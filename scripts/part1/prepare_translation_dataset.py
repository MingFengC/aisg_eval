"""CLI entry point for the staged Part 1 translation dataset pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import apply_runtime_overrides, load_pipeline_config
from .pipeline import TranslationDatasetPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, judge, and finalize the English->Indonesian evaluation dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/part1_practical_api_flow.json"),
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "cometkiwi", "judge",
                 "finalize", "remediate", "all"],
        default="generate",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing JSONL checkpoints.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without model/API calls.")
    parser.add_argument(
        "--limit", type=int, help="Temporarily override sample size for smoke tests.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_pipeline_config(args.config)
    config.log_level = args.log_level
    apply_runtime_overrides(
        config,
        dry_run=args.dry_run,
        resume=args.resume,
        limit=args.limit,
    )

    TranslationDatasetPipeline(config).run(args.stage)


if __name__ == "__main__":
    main()
