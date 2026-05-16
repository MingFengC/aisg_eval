"""Typed configuration and records for the Part 2 evaluation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/translation_eval_ind.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/part2")
DEFAULT_METRICX_MODEL = "google/metricx-24-hybrid-large-v2p6"
DEFAULT_METRICX_TOKENIZER = "google/mt5-xl"
DEFAULT_JUDGE_MODEL = "aisingapore/Qwen-SEA-LION-v4-32B-IT"
DEFAULT_API_BASE_URL = "https://api.sea-lion.ai/v1"


@dataclass
class Part2Config:
    model_id: str
    input_path: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    stage: str = "all"
    limit: int | None = None
    resume: bool = False
    batch_size: int = 1
    max_new_tokens: int = 3072
    torch_dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str | None = None
    enable_metricx: bool = True
    metricx_model: str = DEFAULT_METRICX_MODEL
    metricx_tokenizer: str = DEFAULT_METRICX_TOKENIZER
    metricx_max_input_length: int = 1536
    metricx_batch_size: int = 1
    enable_judge: bool = False
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_mode: str = "all"
    judge_min_overall: float = 4.0
    judge_min_accuracy: float = 4.0
    api_base_url: str = DEFAULT_API_BASE_URL
    api_key_env: str = "SEA_LION_API_KEY"
    requests_per_minute: int = 10
    api_concurrency: int = 3
    timeout_seconds: int = 120
    log_level: str = "INFO"

    def model_output_dir(self) -> Path:
        safe_model_id = self.model_id.replace("/", "__")
        return self.output_dir / safe_model_id


@dataclass
class EvaluationInput:
    id: str
    source: str
    reference: str
    source_language: str
    target_language: str
    source_word_count: int
    source_char_count: int
    source_length_bucket: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationInput":
        return cls(
            id=str(data["id"]),
            source=str(data["source"]),
            reference=str(data["reference"]),
            source_language=str(data.get("source_language", "en")),
            target_language=str(data.get("target_language", "id")),
            source_word_count=int(data.get("source_word_count", 0)),
            source_char_count=int(data.get("source_char_count", len(str(data["source"])))),
            source_length_bucket=str(data.get("source_length_bucket", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRecord:
    id: str
    model_id: str
    source: str
    reference: str
    hypothesis: str
    source_language: str
    target_language: str
    source_word_count: int
    source_char_count: int
    source_length_bucket: str
    hypothesis_word_count: int
    inference_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricRecord:
    id: str
    model_id: str
    chrfpp: float | None = None
    metricx_error: float | None = None
    metricx_quality_0_100: float | None = None
    metricx_input_tokens: int | None = None
    metricx_truncated: bool | None = None
    judge_overall: float | None = None
    judge_accuracy: float | None = None
    judge_flagged: bool | None = None
    judge_score: dict[str, Any] | None = None
    judge_raw_response: str | None = None
    judge_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
