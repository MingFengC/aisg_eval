"""Configuration loading for the staged translation dataset pipeline."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from ..config.data_object import PipelineConfig


PATH_FIELDS = {
    "output",
    "qa_output",
    "manifest_output",
    "cometkiwi_output",
    "judge_output",
}


SECTION_KEYS = {
    "paths",
    "sampling",
    "generation",
    "api",
    "qa",
    "judge",
    "qwen3vl",
}


def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if key in SECTION_KEYS and isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _coerce_paths(values: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(values)
    for key in PATH_FIELDS:
        if key in coerced and coerced[key] is not None:
            coerced[key] = Path(coerced[key])
    return coerced


def load_pipeline_config(path: Path | str) -> PipelineConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(PipelineConfig)}
    flattened = _coerce_paths(_flatten_config(raw))
    if "judge_min_adequacy" in flattened and "judge_min_accuracy" not in flattened:
        flattened["judge_min_accuracy"] = flattened["judge_min_adequacy"]
    values = {
        key: value
        for key, value in flattened.items()
        if key in allowed
    }
    return PipelineConfig(**values)


def apply_runtime_overrides(
    config: PipelineConfig,
    *,
    dry_run: bool | None = None,
    resume: bool | None = None,
    limit: int | None = None,
) -> PipelineConfig:
    if dry_run is not None:
        config.dry_run = dry_run
    if resume is not None:
        config.resume = resume
    if limit is not None:
        config.sample_size = limit
    return config
