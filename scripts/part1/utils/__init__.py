"""Utility functions shared by the Part 1 pipeline."""

from .io import load_dotenv, load_jsonl_by_id, utc_now, write_jsonl, write_manifest
from .text import (
    clean_model_output,
    compute_copy_rate,
    has_llm_artifact_text,
    has_repeated_text,
    heading_numbers,
    indonesian_marker_stats,
    list_marker_count,
    normalize_source_text,
    parse_json_from_model_output,
    stable_source_id,
    tokenize_words,
    word_count,
)

__all__ = [
    "clean_model_output",
    "compute_copy_rate",
    "has_llm_artifact_text",
    "has_repeated_text",
    "heading_numbers",
    "indonesian_marker_stats",
    "list_marker_count",
    "load_dotenv",
    "load_jsonl_by_id",
    "normalize_source_text",
    "parse_json_from_model_output",
    "stable_source_id",
    "tokenize_words",
    "utc_now",
    "word_count",
    "write_jsonl",
    "write_manifest",
]
