"""Dataset loading and length-stratified sampling."""

from __future__ import annotations
from ..utils.text import normalize_source_text, stable_source_id, word_count
from datasets import load_dataset
import pandas as pd

import logging
import os
import random
from typing import Any, Iterable

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")


LOGGER = logging.getLogger("prepare_translation_dataset")


class SourceSampler:
    def __init__(self, dataset_name: str, min_words: int, seed: int) -> None:
        self.dataset_name = dataset_name
        self.min_words = min_words
        self.seed = seed

    def load_source_dataset(self) -> tuple[pd.DataFrame, str]:
        try:
            LOGGER.info("Loading dataset from Hugging Face: %s",
                        self.dataset_name)
            dataset = load_dataset(self.dataset_name)
            return dataset["train"].to_pandas(), f"huggingface:{self.dataset_name}"
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {self.dataset_name} from Hugging Face") from exc

    def prepare_source_pool(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            raise ValueError("Expected dataset to contain a 'text' column")

        pool = pd.DataFrame({"source": df["text"].map(normalize_source_text)})
        pool["source_id"] = pool["source"].map(stable_source_id)
        pool = pool.drop_duplicates(subset=["source_id"]).copy()
        pool["source_word_count"] = pool["source"].map(word_count)
        pool["source_char_count"] = pool["source"].str.len()
        pool = pool.loc[pool["source_word_count"] >= self.min_words].copy()
        if len(pool) < 3:
            raise ValueError("Not enough eligible rows after filtering")

        pool["source_length_bucket"] = pd.qcut(
            pool["source_word_count"],
            q=3,
            labels=["short", "medium", "long"],
            duplicates="drop",
        ).astype(str)
        return pool.reset_index(drop=True)

    @staticmethod
    def allocate_sample_counts(sample_size: int, buckets: Iterable[str]) -> dict[str, int]:
        ordered_buckets = list(buckets)
        base = sample_size // len(ordered_buckets)
        remainder = sample_size % len(ordered_buckets)
        counts = {bucket: base for bucket in ordered_buckets}
        for bucket in ordered_buckets[:remainder]:
            counts[bucket] += 1
        return counts

    def sample_evaluation_sources(self, pool: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        buckets = ["short", "medium", "long"]
        counts = self.allocate_sample_counts(sample_size, buckets)
        sampled_parts = []
        for offset, bucket in enumerate(buckets):
            bucket_df = pool.loc[pool["source_length_bucket"] == bucket]
            n = counts[bucket]
            if len(bucket_df) < n:
                raise ValueError(
                    f"Bucket {bucket} has {len(bucket_df)} rows, cannot sample {n}")
            sampled_parts.append(bucket_df.sample(
                n=n, random_state=self.seed + offset))

        sampled = pd.concat(sampled_parts).sort_values(
            ["source_length_bucket", "source_word_count", "source_id"]
        )
        sampled = sampled.reset_index(drop=True)
        sampled["id"] = [f"eval_{idx:05d}" for idx in range(len(sampled))]
        return sampled

    def build_replacement_pools(
        self, pool: pd.DataFrame, sampled: pd.DataFrame
    ) -> dict[str, list[dict[str, Any]]]:
        used_source_ids = set(sampled["source_id"])
        replacement_pools: dict[str, list[dict[str, Any]]] = {}
        rng = random.Random(self.seed)
        for bucket in ["short", "medium", "long"]:
            candidates = (
                pool.loc[
                    (pool["source_length_bucket"] == bucket)
                    & (~pool["source_id"].isin(used_source_ids))
                ]
                .copy()
                .to_dict(orient="records")
            )
            rng.shuffle(candidates)
            replacement_pools[bucket] = candidates
        return replacement_pools
