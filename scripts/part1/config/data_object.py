"""Typed data objects for the translation dataset pipeline.

The project intentionally uses dataclasses instead of Pydantic so the Kaggle
runtime stays lightweight and does not need an extra validation dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DATASET_NAME = "HuggingFaceTB/cosmopedia_stanford_openstax_wiki_1k"

DEFAULT_OUTPUT = Path("data/translation_eval_ind.jsonl")
DEFAULT_QA_OUTPUT = Path("data/translation_eval_ind_qa.jsonl")
DEFAULT_MANIFEST_OUTPUT = Path("data/translation_eval_ind_manifest.json")
DEFAULT_COMETKIWI_OUTPUT = Path("data/translation_eval_ind_cometkiwi.jsonl")
DEFAULT_JUDGE_OUTPUT = Path("data/translation_eval_ind_judge_sealion_qwen.jsonl")

DEFAULT_GENERATOR_MODEL = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
DEFAULT_JUDGE_MODEL = "aisingapore/Qwen-SEA-LION-v4-32B-IT"
DEFAULT_QWEN3VL_JUDGE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_QWEN3VL_INT4_JUDGE_MODEL = "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"
DEFAULT_API_BASE_URL = "https://api.sea-lion.ai/v1"


@dataclass
class PipelineConfig:
    dataset_name: str = DATASET_NAME
    output: Path = DEFAULT_OUTPUT
    qa_output: Path = DEFAULT_QA_OUTPUT
    manifest_output: Path = DEFAULT_MANIFEST_OUTPUT
    cometkiwi_output: Path = DEFAULT_COMETKIWI_OUTPUT
    judge_output: Path = DEFAULT_JUDGE_OUTPUT
    sample_size: int = 300
    min_words: int = 100
    seed: int = 42
    generator_provider: str = "sealion_api"
    generator_model: str = DEFAULT_GENERATOR_MODEL
    judge_provider: str = "sealion_api"
    judge_model: str = DEFAULT_JUDGE_MODEL
    qwen3vl_model: str = DEFAULT_QWEN3VL_JUDGE_MODEL
    qwen3vl_dtype: str = "bfloat16"
    qwen3vl_device_map: str = "auto"
    qwen3vl_attn_implementation: str | None = None
    qwen3vl_load_in_4bit: bool = False
    qwen3vl_bnb_4bit_quant_type: str = "nf4"
    qwen3vl_bnb_4bit_use_double_quant: bool = True
    qwen3vl_max_new_tokens: int = 1024
    api_base_url: str = DEFAULT_API_BASE_URL
    api_key_env: str = "SEA_LION_API_KEY"
    requests_per_minute: int = 10
    api_concurrency: int = 3
    timeout_seconds: int = 120
    max_retries: int = 2
    max_replacements: int = 5
    max_generation_tokens: int = 4096
    length_ratio_min: float = 0.40
    length_ratio_max: float = 2.20
    copy_rate_max: float = 0.65
    enable_cometkiwi: bool = False
    cometkiwi_model: str = "Unbabel/wmt22-cometkiwi-da"
    cometkiwi_batch_size: int = 8
    cometkiwi_gpus: int | None = None
    enable_judge: bool = False
    judge_mode: str = "sample"
    judge_random_audit_rate: float = 0.05
    judge_min_overall: float = 4.0
    judge_min_accuracy: float = 4.0
    dry_run: bool = False
    resume: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_namespace(cls, namespace: Any) -> "PipelineConfig":
        return cls(**vars(namespace))


@dataclass
class QAResult:
    id: str
    source_id: str
    status: str
    flags: list[str] = field(default_factory=list)
    hard_gate_passed: bool = False
    retry_count: int = 0
    replacement_count: int = 0
    source_length_bucket: str | None = None
    source_word_count: int | None = None
    reference_word_count: int | None = None
    source_reference_word_ratio: float | None = None
    copy_rate: float | None = None
    indonesian_marker_count: int | None = None
    indonesian_marker_ratio: float | None = None
    source_chunk_count: int | None = None
    reference_chunk_count: int | None = None
    heading_numbers_expected: list[str] = field(default_factory=list)
    heading_numbers_found: list[str] = field(default_factory=list)
    cometkiwi_score: float | None = None
    cometkiwi_flagged: bool | None = None
    cometkiwi_threshold: float | None = None
    judge_selected: bool = False
    judge_skip_reason: str | None = None
    judge_score: dict[str, Any] | None = None
    judge_flagged: bool | None = None
    judge_raw_response: str | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAResult":
        known_fields = cls.__dataclass_fields__.keys()
        values = {key: data[key] for key in known_fields if key in data}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
