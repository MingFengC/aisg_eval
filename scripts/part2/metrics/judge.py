"""LLM-as-a-judge scoring for candidate model translations."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from scripts.part1.clients.sea_lion_client import SeaLionClient
from scripts.part1.evaluation.qa import (
    ADAPTED_MQM_NABABAN_BAKER_RUBRIC,
    JUDGE_CALIBRATION_GUIDANCE,
    RUBRIC_SCORE_KEYS,
)
from scripts.part1.utils.text import parse_json_from_model_output

from ..config import MetricRecord, Part2Config, PredictionRecord


class TranslationQualityJudge:
    def __init__(self, config: Part2Config) -> None:
        self.config = config

    @staticmethod
    def _score_value(score: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(score.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    @staticmethod
    def build_messages(prediction: PredictionRecord) -> list[dict[str, str]]:
        schema = {
            "accuracy": "integer 1-5",
            "acceptability": "integer 1-5",
            "readability": "integer 1-5",
            "lexical_equivalence": "integer 1-5",
            "grammatical_equivalence": "integer 1-5",
            "cohesion_coherence": "integer 1-5",
            "academic_fluency": "integer 1-5",
            "overall": "number 1-5",
            "critical_error": "boolean",
            "major_issues": ["short English strings"],
            "minor_issues": ["short English strings"],
            "rationale": "one short English sentence",
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are an impartial English-to-Indonesian translation quality judge. "
                    "Evaluate a candidate model translation for educational material. "
                    "Use Indonesian language standards and educational translation expectations, "
                    "but write all issue strings and the rationale in English. "
                    "Return valid JSON only. Do not add markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Evaluate the Indonesian candidate translation against the English source. "
                    "A Part 1 Indonesian reference translation is provided as supporting evidence. "
                    "Use it to help identify omissions, additions, terminology drift, or structure loss, "
                    "but do not penalize valid Indonesian alternatives just because they differ from the reference. "
                    "Do not reward literal translation if it creates unnatural Indonesian, and do not "
                    "penalize accepted Indonesian alternatives or standard retained international terms.\n\n"
                    f"{ADAPTED_MQM_NABABAN_BAKER_RUBRIC}\n\n"
                    f"{JUDGE_CALIBRATION_GUIDANCE}\n\n"
                    "Set critical_error=true for wrong language, major omission, hallucination, meaning reversal, "
                    "severe grammar or terminology failure, incoherent Indonesian, or severe structure loss.\n\n"
                    f"Return JSON with this schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
                    f"English source:\n<source>\n{prediction.source}\n</source>\n\n"
                    f"Part 1 Indonesian reference:\n<reference>\n{prediction.reference}\n</reference>\n\n"
                    f"Candidate Indonesian translation:\n<candidate>\n{prediction.hypothesis}\n</candidate>"
                ),
            },
        ]

    def _flagged(self, score: dict[str, Any]) -> bool:
        overall = self._score_value(score, "overall")
        accuracy = self._score_value(score, "accuracy")
        critical = self._bool_value(score.get("critical_error", False))
        severe_dimension = any(
            self._score_value(score, key, default=5.0) <= 2.0 for key in RUBRIC_SCORE_KEYS
        )
        return (
            critical
            or overall < self.config.judge_min_overall
            or accuracy < self.config.judge_min_accuracy
            or severe_dimension
        )

    async def judge_one(
        self,
        client: SeaLionClient,
        prediction: PredictionRecord,
    ) -> MetricRecord:
        response = await client.chat(
            model=self.config.judge_model,
            messages=self.build_messages(prediction),
            temperature=0.0,
            max_tokens=1024,
        )
        score = parse_json_from_model_output(response)
        return MetricRecord(
            id=prediction.id,
            model_id=prediction.model_id,
            judge_overall=self._score_value(score, "overall"),
            judge_accuracy=self._score_value(score, "accuracy"),
            judge_flagged=self._flagged(score),
            judge_score=score,
            judge_raw_response=response,
        )

    async def judge(self, predictions: list[PredictionRecord]) -> dict[str, MetricRecord]:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.config.api_key_env}")

        records: dict[str, MetricRecord] = {}
        semaphore = asyncio.Semaphore(max(1, self.config.api_concurrency))
        async with SeaLionClient(
            api_key=api_key,
            base_url=self.config.api_base_url,
            requests_per_minute=self.config.requests_per_minute,
            timeout_seconds=self.config.timeout_seconds,
            max_connections=self.config.api_concurrency,
        ) as client:

            async def run_one(prediction: PredictionRecord) -> tuple[str, MetricRecord]:
                async with semaphore:
                    try:
                        return prediction.id, await self.judge_one(client, prediction)
                    except Exception as exc:
                        return prediction.id, MetricRecord(
                            id=prediction.id,
                            model_id=prediction.model_id,
                            judge_flagged=True,
                            judge_error=str(exc),
                        )

            tasks = [
                asyncio.create_task(run_one(prediction))
                for prediction in predictions
                if not prediction.inference_error and prediction.hypothesis.strip()
            ]
            for task in asyncio.as_completed(tasks):
                record_id, record = await task
                records[record_id] = record

        return records
