"""CLI entry point for Part 2 translation model evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import Part2Config
from .pipeline import Part2EvaluationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face model on the Part 1 Indonesian translation dataset."
    )
    parser.add_argument("--model-id", required=True, help="Hugging Face model id to evaluate.")
    parser.add_argument("--input", dest="input_path", type=Path, default=Part2Config.input_path)
    parser.add_argument("--output-dir", type=Path, default=Part2Config.output_dir)
    parser.add_argument(
        "--stage",
        choices=["all", "predict", "score", "judge", "aggregate"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--skip-metricx", action="store_true")
    parser.add_argument("--enable-judge", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=10)
    parser.add_argument("--api-concurrency", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Part2Config:
    return Part2Config(
        model_id=args.model_id,
        input_path=args.input_path,
        output_dir=args.output_dir,
        stage=args.stage,
        limit=args.limit,
        resume=args.resume,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        enable_metricx=not args.skip_metricx,
        enable_judge=args.enable_judge or args.stage == "judge",
        requests_per_minute=args.requests_per_minute,
        api_concurrency=args.api_concurrency,
        log_level=args.log_level,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    Part2EvaluationPipeline(config).run()


if __name__ == "__main__":
    main()
