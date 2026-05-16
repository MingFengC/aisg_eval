"""MetricX-24 reference-based scoring."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Part2Config, PredictionRecord
from ..utils import metricx_quality_0_100


LOGGER = logging.getLogger("part2_evaluation")


@dataclass
class MetricXScore:
    error: float
    quality_0_100: float
    input_tokens: int
    truncated: bool


class MetricXScorer:
    """Runs MetricX-24 with the Part 1 Indonesian reference.

    MetricX-24 returns an error score in the range 0-25, where lower is better.
    The companion quality score is a display-friendly linear transform where
    higher is better: 100 means no predicted error, 0 means maximum error.
    """

    def __init__(self, config: Part2Config) -> None:
        self.config = config
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        repo_path = os.getenv("METRICX_REPO_PATH")
        if repo_path and Path(repo_path).exists() and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from metricx24 import models
        except Exception as exc:
            raise RuntimeError(
                "MetricX-24 requires the google-research/metricx package. "
                "Install it before running the MetricX stage."
            ) from exc

        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading MetricX model: %s", self.config.metricx_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.metricx_tokenizer)
        self.model = models.MT5ForRegression.from_pretrained(
            self.config.metricx_model,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def build_reference_based_input(prediction: PredictionRecord) -> str:
        return (
            f"source: {prediction.source} "
            f"candidate: {prediction.hypothesis} "
            f"reference: {prediction.reference}"
        )

    def _tokenize_batch(self, texts: list[str]) -> tuple[dict[str, Any], list[int], list[bool]]:
        assert self.tokenizer is not None
        assert self.device is not None
        features = []
        token_counts = []
        truncated = []
        for text in texts:
            full = self.tokenizer(text, padding=False, truncation=False)
            full_length = max(0, len(full["input_ids"]) - 1)
            encoded = self.tokenizer(
                text,
                max_length=self.config.metricx_max_input_length,
                truncation=True,
                padding=False,
            )
            encoded["input_ids"] = encoded["input_ids"][:-1]
            encoded["attention_mask"] = encoded["attention_mask"][:-1]
            features.append(encoded)
            token_counts.append(full_length)
            truncated.append(full_length > self.config.metricx_max_input_length - 1)

        batch = self.tokenizer.pad(features, return_tensors="pt")
        return (
            {key: value.to(self.device) for key, value in batch.items()},
            token_counts,
            truncated,
        )

    def score(self, predictions: list[PredictionRecord]) -> dict[str, MetricXScore]:
        self.load()
        assert self.model is not None
        assert self.torch is not None

        scores: dict[str, MetricXScore] = {}
        valid = [
            prediction for prediction in predictions
            if not prediction.inference_error and prediction.hypothesis.strip()
        ]
        for start in range(0, len(valid), self.config.metricx_batch_size):
            batch_predictions = valid[start:start + self.config.metricx_batch_size]
            texts = [
                self.build_reference_based_input(prediction)
                for prediction in batch_predictions
            ]
            encoded, token_counts, truncated = self._tokenize_batch(texts)
            with self.torch.inference_mode():
                output = self.model(**encoded)
            raw_scores = output.predictions.detach().float().cpu().tolist()
            for prediction, raw_score, token_count, was_truncated in zip(
                batch_predictions,
                raw_scores,
                token_counts,
                truncated,
            ):
                error = round(float(raw_score), 4)
                scores[prediction.id] = MetricXScore(
                    error=error,
                    quality_0_100=metricx_quality_0_100(error),
                    input_tokens=token_count,
                    truncated=was_truncated,
                )
            LOGGER.info("MetricX scored %s/%s rows", min(start + len(batch_predictions), len(valid)), len(valid))

        return scores
