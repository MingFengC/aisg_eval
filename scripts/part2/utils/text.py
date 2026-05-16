"""Text helpers for Part 2."""

from __future__ import annotations

import re


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?|\d+(?:\.\d+)?")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(str(text)))


def clean_translation_output(text: str) -> str:
    """Keep post-processing intentionally minimal for fair model comparison."""

    text = str(text).strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def metricx_quality_0_100(error_score: float) -> float:
    """Convert MetricX's 0-25 error score into a higher-is-better 0-100 score."""

    clipped = min(25.0, max(0.0, float(error_score)))
    return round(100.0 * (25.0 - clipped) / 25.0, 4)

