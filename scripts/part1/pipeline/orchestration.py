"""Staged orchestration for Part 1 dataset generation and QA."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from ..clients import open_generator_client, open_judge_client
from ..config import PipelineConfig, QAResult
from ..dataset import SourceSampler
from ..evaluation import CometKiwiScorer, HardQAChecker, TranslationJudge
from ..translation import TranslationGenerator
from ..utils import load_dotenv, load_jsonl_by_id, utc_now, write_jsonl, write_manifest


LOGGER = logging.getLogger("prepare_translation_dataset")


class ManifestBuilder:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def build(self, data_source: str, pool: pd.DataFrame, sampled: pd.DataFrame) -> dict[str, Any]:
        bucket_counts = sampled["source_length_bucket"].value_counts(
        ).sort_index().to_dict()
        return {
            "created_at_utc": utc_now(),
            "dataset_source": data_source,
            "source_language": "en",
            "target_language": "id",
            "sampling": {
                "sample_size": self.config.sample_size,
                "min_words": self.config.min_words,
                "seed": self.config.seed,
                "eligible_rows": int(len(pool)),
                "bucket_counts": {str(key): int(value) for key, value in bucket_counts.items()},
                "strategy": "equal allocation over source word-count terciles",
            },
            "generation": {
                "provider": self.config.generator_provider,
                "model": self.config.generator_model,
                "max_retries": self.config.max_retries,
                "max_replacements": self.config.max_replacements,
                "chunking": "disabled",
                "requests_per_minute": self.config.requests_per_minute,
                "concurrency": max(1, self.config.api_concurrency),
            },
            "qa": {
                "sanity_checks": "enabled",
                "length_ratio_min": self.config.length_ratio_min,
                "length_ratio_max": self.config.length_ratio_max,
                "copy_rate_max": self.config.copy_rate_max,
                "cometkiwi_enabled": self.config.enable_cometkiwi,
                "judge": {
                    "enabled": self.config.enable_judge,
                    "provider": self.config.judge_provider,
                    "model": self.config.qwen3vl_model
                    if self.config.judge_provider == "qwen3vl_local"
                    else self.config.judge_model,
                    "mode": self.config.judge_mode,
                },
            },
            "outputs": {
                "dataset": str(self.config.output),
                "qa": str(self.config.qa_output),
                "cometkiwi": str(self.config.cometkiwi_output),
                "judge": str(self.config.judge_output),
                "manifest": str(self.config.manifest_output),
            },
        }


class SourceDatasetService:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.sampler = SourceSampler(
            dataset_name=config.dataset_name,
            min_words=config.min_words,
            seed=config.seed,
        )

    def load_sample(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        df, data_source = self.sampler.load_source_dataset()
        pool = self.sampler.prepare_source_pool(df)
        sampled = self.sampler.sample_evaluation_sources(
            pool, self.config.sample_size)
        self.validate_sample(sampled)
        return df, pool, sampled, data_source

    def validate_sample(self, sampled: pd.DataFrame) -> None:
        if len(sampled) != self.config.sample_size:
            raise AssertionError(
                f"Expected {self.config.sample_size} sampled rows, got {len(sampled)}")
        if (sampled["source_word_count"] < self.config.min_words).any():
            raise AssertionError("Sample contains rows below min_words")
        bucket_counts = sampled["source_length_bucket"].value_counts(
        ).to_dict()
        expected_counts = self.sampler.allocate_sample_counts(
            self.config.sample_size, ["short", "medium", "long"]
        )
        if bucket_counts != expected_counts:
            raise AssertionError(
                f"Bucket counts mismatch: {bucket_counts} != {expected_counts}")


class GenerationStage:
    def __init__(self, config: PipelineConfig, source_service: SourceDatasetService) -> None:
        self.config = config
        self.source_service = source_service
        self.hard_qa = HardQAChecker(config)

    @staticmethod
    def output_record(row: dict[str, Any], reference: str, source_chunks: list[str]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "source": row["source"],
            "reference": reference,
            "source_language": "en",
            "target_language": "id",
            "source_word_count": int(row["source_word_count"]),
            "source_char_count": int(row["source_char_count"]),
            "source_length_bucket": row["source_length_bucket"],
            "source_chunk_count": len(source_chunks),
        }

    @staticmethod
    def pop_replacement(
        replacement_pools: dict[str, list[dict[str, Any]]],
        bucket: str,
        used_source_ids: set[str],
    ) -> dict[str, Any] | None:
        pool = replacement_pools.get(bucket, [])
        while pool:
            candidate = pool.pop(0)
            if candidate["source_id"] not in used_source_ids:
                used_source_ids.add(candidate["source_id"])
                return candidate
        return None

    @staticmethod
    def mark_pending_judge(qa_by_id: dict[str, QAResult], config: PipelineConfig) -> None:
        if not config.enable_judge:
            return
        for qa in qa_by_id.values():
            if qa.judge_score is None and qa.judge_flagged is None:
                qa.judge_selected = False
                qa.judge_skip_reason = "pending_external_judge_stage"

    async def generate_with_hard_qa(
        self,
        row: dict[str, Any],
        generator: TranslationGenerator,
        replacement_count: int = 0,
        strict_first: bool = False,
    ) -> tuple[dict[str, Any] | None, QAResult]:
        latest_qa: QAResult | None = None
        for retry_count in range(self.config.max_retries + 1):
            stricter = strict_first or retry_count > 0
            try:
                reference, source_chunks = await generator.translate(
                    source=row["source"],
                    stricter=stricter,
                )
            except Exception as exc:
                latest_qa = QAResult(
                    id=row["id"],
                    source_id=row["source_id"],
                    status="generation_error",
                    flags=["generation_error"],
                    retry_count=retry_count,
                    replacement_count=replacement_count,
                    source_length_bucket=row["source_length_bucket"],
                    source_word_count=int(row["source_word_count"]),
                    notes=[str(exc)],
                )
                LOGGER.error("Generation error for %s: %s", row["id"], exc)
                continue

            qa = self.hard_qa.run(
                record_id=row["id"],
                row=row,
                reference=reference,
                source_chunks=source_chunks,
                retry_count=retry_count,
                replacement_count=replacement_count,
            )
            latest_qa = qa
            if qa.hard_gate_passed:
                return self.output_record(row, reference, source_chunks), qa
            LOGGER.warning("Hard QA failed for %s retry %s: %s",
                           row["id"], retry_count, qa.flags)

        assert latest_qa is not None
        return None, latest_qa

    async def process_row(
        self,
        row: dict[str, Any],
        generator: TranslationGenerator,
        replacement_pools: dict[str, list[dict[str, Any]]],
        used_source_ids: set[str],
        replacement_lock: asyncio.Lock,
    ) -> tuple[str, dict[str, Any], QAResult]:
        record_id = row["id"]
        active_row = dict(row)
        for replacement_count in range(self.config.max_replacements + 1):
            active_row["id"] = record_id
            record, qa = await self.generate_with_hard_qa(active_row, generator, replacement_count)
            if record is not None and qa.hard_gate_passed:
                return record_id, record, qa

            async with replacement_lock:
                replacement = self.pop_replacement(
                    replacement_pools,
                    row["source_length_bucket"],
                    used_source_ids,
                )
            if replacement is None:
                break
            active_row = dict(replacement)
            active_row["id"] = record_id
            LOGGER.warning("Replacing failed source for %s with %s",
                           record_id, active_row["source_id"])

        raise RuntimeError(
            f"Could not produce a hard-QA-passing record for {record_id}")

    async def run(self) -> None:
        _, pool, sampled, data_source = self.source_service.load_sample()
        manifest = ManifestBuilder(self.config).build(
            data_source, pool, sampled)
        manifest["stages"] = {"generate": {"started_at_utc": utc_now()}}
        write_manifest(self.config.manifest_output, manifest)

        if self.config.dry_run:
            LOGGER.info("Dry run complete; no generation API calls made")
            return

        existing_records = load_jsonl_by_id(
            self.config.output) if self.config.resume else {}
        existing_qa_raw = load_jsonl_by_id(
            self.config.qa_output) if self.config.resume else {}
        records_by_id: dict[str, dict[str, Any]] = dict(existing_records)
        qa_by_id: dict[str, QAResult] = {
            record_id: QAResult.from_dict(qa)
            for record_id, qa in existing_qa_raw.items()
        }

        rows_to_generate = [
            row.to_dict()
            for _, row in sampled.iterrows()
            if row["id"] not in records_by_id
        ]

        if rows_to_generate:
            used_source_ids = set(sampled["source_id"])
            replacement_pools = self.source_service.sampler.build_replacement_pools(
                pool, sampled)
            replacement_lock = asyncio.Lock()
            concurrency = max(1, self.config.api_concurrency)
            semaphore = asyncio.Semaphore(concurrency)
            LOGGER.info(
                "Generating %s records with %s async tasks at %s requests/minute",
                len(rows_to_generate),
                concurrency,
                self.config.requests_per_minute,
            )
            async with open_generator_client(self.config) as client:
                generator = TranslationGenerator(
                    client=client,
                    model=self.config.generator_model,
                    max_generation_tokens=self.config.max_generation_tokens,
                )

                async def run_one(row: dict[str, Any]) -> tuple[str, dict[str, Any], QAResult]:
                    async with semaphore:
                        return await self.process_row(
                            row,
                            generator,
                            replacement_pools,
                            used_source_ids,
                            replacement_lock,
                        )

                tasks = [asyncio.create_task(run_one(row))
                         for row in rows_to_generate]
                for task in asyncio.as_completed(tasks):
                    record_id, record, qa = await task
                    records_by_id[record_id] = record
                    qa_by_id[record_id] = qa
                    ordered_records = [records_by_id[key]
                                       for key in sorted(records_by_id)]
                    self.mark_pending_judge(qa_by_id, self.config)
                    ordered_qa = [qa_by_id[key].to_dict()
                                  for key in sorted(qa_by_id)]
                    write_jsonl(self.config.output, ordered_records)
                    write_jsonl(self.config.qa_output, ordered_qa)
                    LOGGER.info("Accepted %s (%s/%s)", record_id,
                                len(records_by_id), len(sampled))

        records = [records_by_id[key] for key in sorted(records_by_id)]
        qa_records = [qa_by_id[key].to_dict() for key in sorted(qa_by_id)]
        write_jsonl(self.config.output, records)
        write_jsonl(self.config.qa_output, qa_records)

        manifest["stages"]["generate"]["completed_at_utc"] = utc_now()
        manifest["stages"]["generate"]["records"] = len(records)
        write_manifest(self.config.manifest_output, manifest)


class JudgeStage:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def _load_qa(self, records: list[dict[str, Any]]) -> dict[str, QAResult]:
        qa_raw = load_jsonl_by_id(self.config.qa_output)
        qa_by_id = {}
        for record in records:
            raw = qa_raw.get(record["id"])
            qa_by_id[record["id"]] = (
                QAResult.from_dict(raw)
                if raw
                else QAResult(
                    id=record["id"],
                    source_id=record.get("source_id", record["id"]),
                    status="accepted",
                    hard_gate_passed=True,
                    source_length_bucket=record.get("source_length_bucket"),
                    source_word_count=record.get("source_word_count"),
                )
            )
        return qa_by_id

    @staticmethod
    def _judge_model_id(config: PipelineConfig) -> str:
        return config.qwen3vl_model if config.judge_provider == "qwen3vl_local" else config.judge_model

    async def run(self) -> None:
        records = list(load_jsonl_by_id(self.config.output).values())
        if not records:
            raise RuntimeError(
                f"No dataset records found at {self.config.output}")
        records = sorted(records, key=lambda item: item["id"])
        if self.config.sample_size and len(records) > self.config.sample_size:
            records = records[: self.config.sample_size]
        qa_by_id = self._load_qa(records)
        existing = load_jsonl_by_id(
            self.config.judge_output) if self.config.resume else {}

        judge = TranslationJudge(self.config, None)
        selected = []
        for record in records:
            if record["id"] in existing:
                continue
            if judge.should_judge_record(record["id"], qa_by_id[record["id"]]):
                selected.append(record)

        if self.config.dry_run:
            LOGGER.info("Dry run: %s records would be judged", len(selected))
            return

        if not selected:
            LOGGER.info("No new records selected for judging")
            return

        concurrency = 1 if self.config.judge_provider == "qwen3vl_local" else max(
            1, self.config.api_concurrency)
        semaphore = asyncio.Semaphore(concurrency)
        judge_results = dict(existing)

        async with open_judge_client(self.config) as client:
            judge = TranslationJudge(self.config, client)

            async def run_one(record: dict[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    try:
                        result = await judge.judge_record(record)
                        return {
                            "id": record["id"],
                            "judge_provider": self.config.judge_provider,
                            "judge_model": self._judge_model_id(self.config),
                            "judge_score": result["score"],
                            "judge_flagged": result["flagged"],
                            "judge_raw_response": result["raw_response"],
                            "judge_error": None,
                            "created_at_utc": utc_now(),
                        }
                    except Exception as exc:
                        LOGGER.warning("Judge failed for %s: %s",
                                       record["id"], exc)
                        return {
                            "id": record["id"],
                            "judge_provider": self.config.judge_provider,
                            "judge_model": self._judge_model_id(self.config),
                            "judge_score": None,
                            "judge_flagged": True,
                            "judge_raw_response": None,
                            "judge_error": str(exc),
                            "created_at_utc": utc_now(),
                        }

            LOGGER.info("Judging %s records with provider=%s",
                        len(selected), self.config.judge_provider)
            tasks = [asyncio.create_task(run_one(record))
                     for record in selected]
            for task in asyncio.as_completed(tasks):
                result = await task
                judge_results[result["id"]] = result
                write_jsonl(
                    self.config.judge_output,
                    [judge_results[key] for key in sorted(judge_results)],
                )
                LOGGER.info("Judged %s (%s/%s)",
                            result["id"], len(judge_results), len(records))


class CometKiwiStage:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    async def run(self) -> None:
        records_by_id = load_jsonl_by_id(self.config.output)
        if self.config.dry_run and not records_by_id:
            LOGGER.info(
                "Dry run: COMETKiwi stage requires generated records at %s", self.config.output)
            return
        records = list(records_by_id.values())
        if not records:
            raise RuntimeError(
                f"No dataset records found at {self.config.output}")
        records = sorted(records, key=lambda item: item["id"])
        if self.config.sample_size and len(records) > self.config.sample_size:
            records = records[: self.config.sample_size]

        existing = load_jsonl_by_id(
            self.config.cometkiwi_output) if self.config.resume else {}
        selected = [
            record for record in records if record["id"] not in existing]

        if self.config.dry_run:
            LOGGER.info(
                "Dry run: %s records would be scored with COMETKiwi", len(selected))
            return
        if not selected:
            LOGGER.info("No new records selected for COMETKiwi scoring")
            return

        scorer = CometKiwiScorer(self.config)
        model, error = scorer.load_model()
        if model is None:
            raise RuntimeError(f"COMETKiwi unavailable: {error}")

        LOGGER.info(
            "Scoring %s records with COMETKiwi model=%s batch_size=%s gpus=%s",
            len(selected),
            self.config.cometkiwi_model,
            self.config.cometkiwi_batch_size,
            scorer.resolve_gpus(),
        )
        scores = scorer.score(model, selected)
        results = dict(existing)
        for record, score in zip(selected, scores):
            results[record["id"]] = {
                "id": record["id"],
                "cometkiwi_model": self.config.cometkiwi_model,
                "cometkiwi_score": round(float(score), 4),
                "cometkiwi_error": None,
                "created_at_utc": utc_now(),
            }

        write_jsonl(
            self.config.cometkiwi_output,
            [results[key] for key in sorted(results)],
        )
        LOGGER.info("Wrote COMETKiwi scores to %s",
                    self.config.cometkiwi_output)


class FinalizeStage:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @staticmethod
    def soft_failed_ids(qa_by_id: dict[str, QAResult]) -> list[str]:
        return sorted(
            record_id
            for record_id, qa in qa_by_id.items()
            if qa.cometkiwi_flagged or qa.judge_flagged
        )

    def merge_judge_results(self, qa_by_id: dict[str, QAResult]) -> dict[str, Any]:
        judge_raw = load_jsonl_by_id(self.config.judge_output)
        merged = 0
        flagged = 0
        errors = 0
        for record_id, result in judge_raw.items():
            qa = qa_by_id.get(record_id)
            if qa is None:
                continue
            qa.judge_selected = True
            qa.judge_skip_reason = None
            qa.judge_score = result.get("judge_score")
            qa.judge_flagged = bool(result.get("judge_flagged"))
            qa.judge_raw_response = result.get("judge_raw_response")
            qa.flags = [
                flag
                for flag in qa.flags
                if flag not in {"judge_low_score", "judge_error"}
            ]
            if result.get("judge_error"):
                errors += 1
                qa.flags.append("judge_error")
                qa.notes.append(f"judge_error: {result['judge_error']}")
            elif qa.judge_flagged:
                flagged += 1
                qa.flags.append("judge_low_score")
            merged += 1
        return {"merged": merged, "flagged": flagged, "errors": errors}

    def merge_cometkiwi_results(self, qa_by_id: dict[str, QAResult]) -> dict[str, Any]:
        comet_raw = load_jsonl_by_id(self.config.cometkiwi_output)
        scored = [
            result
            for result in comet_raw.values()
            if result.get("cometkiwi_score") is not None and not result.get("cometkiwi_error")
        ]
        if not scored:
            return {"merged": 0, "flagged": 0, "errors": len(comet_raw)}

        scores = [float(result["cometkiwi_score"]) for result in scored]
        threshold = CometKiwiScorer.threshold_for_scores(scores)
        merged = 0
        flagged = 0
        errors = 0
        for record_id, result in comet_raw.items():
            qa = qa_by_id.get(record_id)
            if qa is None:
                continue
            qa.flags = [flag for flag in qa.flags if flag not in {
                "cometkiwi_low_score", "cometkiwi_error"}]
            if result.get("cometkiwi_error"):
                errors += 1
                qa.cometkiwi_score = None
                qa.cometkiwi_threshold = round(threshold, 4)
                qa.cometkiwi_flagged = True
                qa.flags.append("cometkiwi_error")
                qa.notes.append(
                    f"cometkiwi_error: {result['cometkiwi_error']}")
            else:
                score = float(result["cometkiwi_score"])
                qa.cometkiwi_score = round(score, 4)
                qa.cometkiwi_threshold = round(threshold, 4)
                qa.cometkiwi_flagged = score <= threshold
                if qa.cometkiwi_flagged:
                    flagged += 1
                    qa.flags.append("cometkiwi_low_score")
            merged += 1
        return {
            "merged": merged,
            "flagged": flagged,
            "errors": errors,
            "threshold": round(threshold, 4),
        }

    def run(self) -> None:
        records = list(load_jsonl_by_id(self.config.output).values())
        qa_by_id = {
            record_id: QAResult.from_dict(raw)
            for record_id, raw in load_jsonl_by_id(self.config.qa_output).items()
        }
        if not records or not qa_by_id:
            raise RuntimeError(
                "Finalize requires generated dataset and QA JSONL files")

        cometkiwi_stats = (
            self.merge_cometkiwi_results(
                qa_by_id) if self.config.cometkiwi_output.exists() else {}
        )
        judge_stats = self.merge_judge_results(
            qa_by_id) if self.config.judge_output.exists() else {}
        for qa in qa_by_id.values():
            if qa.hard_gate_passed and not qa.cometkiwi_flagged and not qa.judge_flagged:
                qa.status = "accepted"
            elif qa.hard_gate_passed:
                qa.status = "accepted_with_soft_flags"

        write_jsonl(self.config.qa_output, [
                    qa_by_id[key].to_dict() for key in sorted(qa_by_id)])
        manifest = self._load_or_minimal_manifest()
        manifest["completed_at_utc"] = utc_now()
        manifest["stages"] = manifest.get("stages", {})
        manifest["stages"]["finalize"] = {
            "completed_at_utc": utc_now(),
            "cometkiwi_stats": cometkiwi_stats,
            "judge_stats": judge_stats,
        }
        manifest["final_counts"] = {
            "records": len(records),
            "qa_records": len(qa_by_id),
            "accepted": sum(1 for qa in qa_by_id.values() if qa.status == "accepted"),
            "accepted_with_soft_flags": sum(
                1 for qa in qa_by_id.values() if qa.status == "accepted_with_soft_flags"
            ),
            "hard_gate_failures": sum(1 for qa in qa_by_id.values() if not qa.hard_gate_passed),
            "soft_qa_flagged": len(self.soft_failed_ids(qa_by_id)),
        }
        write_manifest(self.config.manifest_output, manifest)

    def _load_or_minimal_manifest(self) -> dict[str, Any]:
        if self.config.manifest_output.exists():
            import json

            return json.loads(self.config.manifest_output.read_text(encoding="utf-8"))
        return {"created_at_utc": utc_now(), "outputs": {"manifest": str(self.config.manifest_output)}}


class RemediationStage:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.hard_qa = HardQAChecker(config)

    async def run(self) -> None:
        records_by_id = load_jsonl_by_id(self.config.output)
        qa_by_id = {
            record_id: QAResult.from_dict(raw)
            for record_id, raw in load_jsonl_by_id(self.config.qa_output).items()
        }
        failed_ids = [
            record_id
            for record_id, qa in qa_by_id.items()
            if qa.hard_gate_passed and (qa.cometkiwi_flagged or qa.judge_flagged)
        ]
        if self.config.dry_run:
            LOGGER.info("Dry run: %s rows would be regenerated",
                        len(failed_ids))
            return
        if not failed_ids:
            LOGGER.info("No soft-QA flagged rows to remediate")
            return

        async with open_generator_client(self.config) as client:
            generator = TranslationGenerator(
                client=client,
                model=self.config.generator_model,
                max_generation_tokens=self.config.max_generation_tokens,
            )
            for record_id in failed_ids:
                record = records_by_id[record_id]
                qa = qa_by_id[record_id]
                row = {
                    "id": record_id,
                    "source_id": qa.source_id,
                    "source": record["source"],
                    "source_word_count": record["source_word_count"],
                    "source_char_count": len(record["source"]),
                    "source_length_bucket": record["source_length_bucket"],
                }
                reference, chunks = await generator.translate(
                    source=row["source"],
                    stricter=True,
                )
                new_qa = self.hard_qa.run(
                    record_id=record_id,
                    row=row,
                    reference=reference,
                    source_chunks=chunks,
                    retry_count=qa.retry_count + 1,
                    replacement_count=qa.replacement_count,
                )
                if not new_qa.hard_gate_passed:
                    LOGGER.warning(
                        "Remediated row %s failed hard QA: %s", record_id, new_qa.flags)
                    continue
                record["reference"] = reference
                record["source_chunk_count"] = len(chunks)
                records_by_id[record_id] = record
                qa_by_id[record_id] = new_qa
                LOGGER.info("Remediated %s", record_id)

        write_jsonl(self.config.output, [records_by_id[key]
                    for key in sorted(records_by_id)])
        write_jsonl(self.config.qa_output, [
                    qa_by_id[key].to_dict() for key in sorted(qa_by_id)])


class TranslationDatasetPipeline:
    VALID_STAGES = {"generate", "cometkiwi",
                    "judge", "finalize", "remediate", "all"}

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.source_service = SourceDatasetService(config)

    def run(self, stage: str) -> None:
        if stage not in self.VALID_STAGES:
            raise ValueError(
                f"Unsupported stage {stage!r}. Choose from {sorted(self.VALID_STAGES)}")
        asyncio.run(self.run_async(stage))

    async def run_async(self, stage: str) -> None:
        load_dotenv()
        if stage in {"generate", "all"}:
            await GenerationStage(self.config, self.source_service).run()
        if stage in {"cometkiwi", "all"}:
            if not self.config.enable_cometkiwi:
                LOGGER.info(
                    "COMETKiwi stage skipped because enable_cometkiwi=false")
            else:
                await CometKiwiStage(self.config).run()
        if stage in {"judge", "all"}:
            if not self.config.enable_judge:
                LOGGER.info("Judge stage skipped because enable_judge=false")
            else:
                await JudgeStage(self.config).run()
        if stage in {"remediate"}:
            await RemediationStage(self.config).run()
        if stage in {"finalize", "all"}:
            FinalizeStage(self.config).run()
