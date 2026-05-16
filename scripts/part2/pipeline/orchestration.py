"""Orchestration for Part 2 model inference and metric scoring."""

from __future__ import annotations

import asyncio
import logging
from statistics import mean
from typing import Any

from ..config import EvaluationInput, MetricRecord, Part2Config, PredictionRecord
from ..inference import PredictionRunner
from ..metrics import ChrfScorer, MetricXScorer, TranslationQualityJudge
from ..utils import load_dotenv, read_jsonl, read_jsonl_by_id, utc_now, write_json, write_jsonl


LOGGER = logging.getLogger("part2_evaluation")


class Part2Paths:
    def __init__(self, config: Part2Config) -> None:
        self.output_dir = config.model_output_dir()
        self.predictions = self.output_dir / "predictions.jsonl"
        self.row_metrics = self.output_dir / "row_metrics.jsonl"
        self.judge = self.output_dir / "judge.jsonl"
        self.metrics = self.output_dir / "metrics.json"
        self.manifest = self.output_dir / "manifest.json"


class Part2EvaluationPipeline:
    def __init__(self, config: Part2Config) -> None:
        self.config = config
        self.paths = Part2Paths(config)

    def load_inputs(self) -> list[EvaluationInput]:
        records = [EvaluationInput.from_dict(record) for record in read_jsonl(self.config.input_path)]
        if self.config.limit is not None:
            records = records[:self.config.limit]
        return records

    def load_predictions(self) -> list[PredictionRecord]:
        return [
            PredictionRecord(**record)
            for record in read_jsonl(self.paths.predictions)
        ]

    def load_metric_records(self) -> dict[str, MetricRecord]:
        raw = read_jsonl_by_id(self.paths.row_metrics)
        return {record_id: MetricRecord(**record) for record_id, record in raw.items()}

    def write_manifest(self, stage: str, extra: dict[str, Any] | None = None) -> None:
        manifest = {
            "created_at_utc": utc_now(),
            "stage": stage,
            "model_id": self.config.model_id,
            "input": str(self.config.input_path),
            "outputs": {
                "predictions": str(self.paths.predictions),
                "row_metrics": str(self.paths.row_metrics),
                "judge": str(self.paths.judge),
                "metrics": str(self.paths.metrics),
                "manifest": str(self.paths.manifest),
            },
            "inference": {
                "batch_size": self.config.batch_size,
                "max_new_tokens": self.config.max_new_tokens,
                "torch_dtype": self.config.torch_dtype,
                "device_map": self.config.device_map,
                "attn_implementation": self.config.attn_implementation,
            },
            "metrics": {
                "overlap": "chrF++",
                "metricx_enabled": self.config.enable_metricx,
                "metricx_model": self.config.metricx_model,
                "metricx_mode": "reference-based with Part 1 Indonesian reference",
                "metricx_raw_score": "0-25 error score; lower is better",
                "metricx_quality_0_100": "linear transform of raw error; higher is better",
                "judge_enabled": self.config.enable_judge,
                "judge_model": self.config.judge_model,
            },
        }
        if extra:
            manifest.update(extra)
        write_json(self.paths.manifest, manifest)

    def predict(self) -> None:
        rows = self.load_inputs()
        existing = read_jsonl_by_id(self.paths.predictions) if self.config.resume else {}
        LOGGER.info("Running inference for %s rows with %s", len(rows), self.config.model_id)
        predictions = PredictionRunner(self.config).run(rows, existing=existing)
        write_jsonl(self.paths.predictions, [prediction.to_dict() for prediction in predictions])
        self.write_manifest("predict", {"prediction_count": len(predictions)})

    def score(self) -> None:
        predictions = self.load_predictions()
        existing_metrics = self.load_metric_records()
        scorer = ChrfScorer()
        metric_records = {
            prediction.id: existing_metrics.get(
                prediction.id,
                MetricRecord(id=prediction.id, model_id=prediction.model_id),
            )
            for prediction in predictions
        }

        for prediction in predictions:
            metric_records[prediction.id].chrfpp = scorer.score(prediction)

        if self.config.enable_metricx:
            metricx_scores = MetricXScorer(self.config).score(predictions)
            for record_id, score in metricx_scores.items():
                metric_record = metric_records[record_id]
                metric_record.metricx_error = score.error
                metric_record.metricx_quality_0_100 = score.quality_0_100
                metric_record.metricx_input_tokens = score.input_tokens
                metric_record.metricx_truncated = score.truncated

        ordered = [metric_records[key].to_dict() for key in sorted(metric_records)]
        write_jsonl(self.paths.row_metrics, ordered)
        self.write_manifest("score", {"row_metric_count": len(ordered)})
        self.aggregate()

    async def judge_async(self) -> None:
        load_dotenv()
        predictions = self.load_predictions()
        existing_metrics = self.load_metric_records()
        judge_records = await TranslationQualityJudge(self.config).judge(predictions)
        merged = {
            prediction.id: existing_metrics.get(
                prediction.id,
                MetricRecord(id=prediction.id, model_id=prediction.model_id),
            )
            for prediction in predictions
        }

        for record_id, judge_record in judge_records.items():
            metric_record = merged[record_id]
            metric_record.judge_overall = judge_record.judge_overall
            metric_record.judge_accuracy = judge_record.judge_accuracy
            metric_record.judge_flagged = judge_record.judge_flagged
            metric_record.judge_score = judge_record.judge_score
            metric_record.judge_raw_response = judge_record.judge_raw_response
            metric_record.judge_error = judge_record.judge_error

        write_jsonl(self.paths.judge, [record.to_dict() for record in judge_records.values()])
        write_jsonl(self.paths.row_metrics, [merged[key].to_dict() for key in sorted(merged)])
        self.write_manifest("judge", {"judge_count": len(judge_records)})
        self.aggregate()

    def judge(self) -> None:
        asyncio.run(self.judge_async())

    @staticmethod
    def _mean(values: list[float | None]) -> float | None:
        valid = [value for value in values if value is not None]
        if not valid:
            return None
        return round(float(mean(valid)), 4)

    def aggregate(self) -> None:
        predictions = self.load_predictions()
        metrics = [MetricRecord(**record) for record in read_jsonl(self.paths.row_metrics)]
        metric_by_id = {record.id: record for record in metrics}
        aligned_metrics = [metric_by_id[prediction.id] for prediction in predictions if prediction.id in metric_by_id]
        inference_error_count = sum(1 for prediction in predictions if prediction.inference_error)
        judged = [record for record in aligned_metrics if record.judge_overall is not None or record.judge_error]
        judge_flagged = [record for record in judged if record.judge_flagged]

        payload = {
            "model_id": self.config.model_id,
            "input": str(self.config.input_path),
            "n_predictions": len(predictions),
            "n_scored_rows": len(aligned_metrics),
            "inference_error_count": inference_error_count,
            "chrfpp_mean": self._mean([record.chrfpp for record in aligned_metrics]),
            "metricx_error_mean": self._mean([record.metricx_error for record in aligned_metrics]),
            "metricx_quality_0_100_mean": self._mean(
                [record.metricx_quality_0_100 for record in aligned_metrics]
            ),
            "metricx_truncated_count": sum(1 for record in aligned_metrics if record.metricx_truncated),
            "judge_overall_mean": self._mean([record.judge_overall for record in aligned_metrics]),
            "judge_accuracy_mean": self._mean([record.judge_accuracy for record in aligned_metrics]),
            "judge_reviewed_count": len(judged),
            "judge_flagged_count": len(judge_flagged),
            "judge_flag_rate": round(len(judge_flagged) / len(judged), 4) if judged else None,
            "notes": [
                "chrF++ and MetricX quality are higher-is-better.",
                "MetricX raw error is lower-is-better and is the canonical MetricX score.",
                "MetricX quality 0-100 is a linear display transform: 100 * (25 - error) / 25.",
            ],
            "created_at_utc": utc_now(),
        }
        write_json(self.paths.metrics, payload)
        LOGGER.info("Wrote aggregate metrics to %s", self.paths.metrics)

    def run(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.stage == "predict":
            self.predict()
        elif self.config.stage == "score":
            self.score()
        elif self.config.stage == "judge":
            self.judge()
        elif self.config.stage == "aggregate":
            self.aggregate()
        elif self.config.stage == "all":
            self.predict()
            self.score()
            if self.config.enable_judge:
                self.judge()
        else:
            raise ValueError(f"Unsupported stage: {self.config.stage}")
