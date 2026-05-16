"""Overlap-based translation metrics."""

from __future__ import annotations

from ..config import PredictionRecord


class ChrfScorer:
    """chrF++ scorer using sacreBLEU's CHRF implementation."""

    def __init__(self) -> None:
        from sacrebleu.metrics import CHRF

        self.metric = CHRF(word_order=2)

    def score(self, prediction: PredictionRecord) -> float | None:
        if prediction.inference_error or not prediction.hypothesis.strip():
            return None
        score = self.metric.sentence_score(
            prediction.hypothesis,
            [prediction.reference],
        )
        return round(float(score.score), 4)

