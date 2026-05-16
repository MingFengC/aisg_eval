"""Deterministic, COMETKiwi, and judge-LLM QA components."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from typing import Any

import numpy as np

from ..config.data_object import PipelineConfig, QAResult
from ..utils.text import (
    compute_copy_rate,
    has_llm_artifact_text,
    has_repeated_text,
    heading_numbers,
    indonesian_marker_stats,
    list_marker_count,
    looks_incomplete_text,
    parse_json_from_model_output,
    word_count,
)


LOGGER = logging.getLogger("prepare_translation_dataset")


ADAPTED_MQM_NABABAN_BAKER_RUBRIC = (
    "Use this adapted MQM-Nababan-Baker rubric for English-to-Indonesian educational translation. "
    "Score each dimension from 1 to 5:\n"
    "1. accuracy: source meaning is preserved completely, with no omissions, hallucinations, unsupported "
    "additions, or meaning reversals.\n"
    "2. acceptability: the translation conforms to standard Indonesian norms, including EYD Edisi V, "
    "KBBI-compatible usage, appropriate spelling, capitalization, punctuation, absorption words, and "
    "di-/ke- affixes versus di/ke prepositions.\n"
    "3. readability: the Indonesian is clear and easy for the intended educational audience to understand.\n"
    "4. lexical_equivalence: words and terms are translated with precise Indonesian equivalents or accepted "
    "international terms; false friends and literal lexical calques are penalized.\n"
    "5. grammatical_equivalence: Indonesian syntax, word order, passive/active choices, and morphology are "
    "natural, including affixes such as meN-, di-, ber-, ter-, -kan, and -i.\n"
    "6. cohesion_coherence: logical relations, references, comparisons, cause-effect links, paragraph flow, "
    "and discourse connectors remain coherent.\n"
    "7. academic_fluency: tone is formal, neutral, and textbook-like; slang, casual pronouns, informal "
    "particles, and Malay-specific forms are penalized when standard Indonesian has a better equivalent.\n"
    "Also consider paragraph, heading, numbering, list, equation, citation, unit, and code-token preservation "
    "when assigning accuracy, cohesion_coherence, academic_fluency, overall, and critical_error."
)


JUDGE_CALIBRATION_GUIDANCE = (
    "Judge in error-audit mode, not praise mode. First look for omissions, mistranslations, "
    "unsupported additions, awkward literal calques, terminology drift, Indonesian morphology errors, "
    "cohesion breaks, and structure loss. Then assign scores.\n"
    "Calibration anchors:\n"
    "- 5 means publishable as a reference translation with no meaningful edits needed. This should be rare.\n"
    "- 4 means good but at least one minor edit would improve accuracy, terminology, readability, or Indonesian naturalness.\n"
    "- 3 means usable for gist but not reliable as an evaluation reference without revision.\n"
    "- 2 means serious adequacy, fluency, terminology, or structure problems.\n"
    "- 1 means wrong language, largely untranslated, incoherent, or unusable.\n"
    "Do not give all 5s unless you can honestly find no material or minor issue after comparing the source "
    "and translation. Avoid generic praise; cite specific evidence in English."
)


RUBRIC_SCORE_KEYS = (
    "accuracy",
    "acceptability",
    "readability",
    "lexical_equivalence",
    "grammatical_equivalence",
    "cohesion_coherence",
    "academic_fluency",
)


def _score_value(score: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(score.get(key, default))
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


class HardQAChecker:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        record_id: str,
        row: dict[str, Any],
        reference: str,
        source_chunks: list[str],
        retry_count: int,
        replacement_count: int,
    ) -> QAResult:
        source = row["source"]
        source_words = int(row["source_word_count"])
        reference_words = word_count(reference)
        ratio = reference_words / max(source_words, 1)
        copy_rate = compute_copy_rate(source, reference)
        marker_count, marker_ratio = indonesian_marker_stats(reference)
        source_headings = heading_numbers(source)
        reference_headings = heading_numbers(reference)

        flags: list[str] = []
        if not reference.strip():
            flags.append("empty_reference")
        if marker_count < 3 and marker_ratio < 0.012:
            flags.append("indonesian_language_id_low_confidence")
        if ratio < self.config.length_ratio_min or ratio > self.config.length_ratio_max:
            flags.append("source_reference_length_ratio_out_of_range")
        if copy_rate > self.config.copy_rate_max:
            flags.append("copy_rate_too_high")
        if has_repeated_text(reference):
            flags.append("repeated_text_detected")
        if looks_incomplete_text(source):
            flags.append("source_looks_incomplete")
        if looks_incomplete_text(reference):
            flags.append("reference_looks_incomplete")
        if has_llm_artifact_text(source):
            flags.append("source_llm_artifact")
        if has_llm_artifact_text(reference):
            flags.append("reference_llm_artifact")
        if source_headings and not set(source_headings) & set(reference_headings):
            flags.append("heading_numbering_not_preserved")

        source_markers = list_marker_count(source)
        reference_markers = list_marker_count(reference)
        if source_markers >= 3 and reference_markers < max(1, math.floor(source_markers * 0.4)):
            flags.append("list_structure_not_preserved")

        return QAResult(
            id=record_id,
            source_id=str(row["source_id"]),
            status="accepted" if not flags else "failed_hard_gates",
            flags=flags,
            hard_gate_passed=not flags,
            retry_count=retry_count,
            replacement_count=replacement_count,
            source_length_bucket=str(row["source_length_bucket"]),
            source_word_count=source_words,
            reference_word_count=reference_words,
            source_reference_word_ratio=round(ratio, 3),
            copy_rate=round(copy_rate, 3),
            indonesian_marker_count=marker_count,
            indonesian_marker_ratio=round(marker_ratio, 4),
            source_chunk_count=len(source_chunks),
            reference_chunk_count=len(
                [chunk for chunk in reference.split("\n\n") if chunk.strip()]
            ),
            heading_numbers_expected=source_headings,
            heading_numbers_found=reference_headings,
        )


class CometKiwiScorer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def resolve_gpus(self) -> int:
        if self.config.cometkiwi_gpus is not None:
            return max(0, int(self.config.cometkiwi_gpus))
        try:
            import torch  # type: ignore
        except Exception:
            return 0
        return 1 if torch.cuda.is_available() else 0

    def load_model(self) -> tuple[Any | None, str | None]:
        try:
            from comet import download_model, load_from_checkpoint  # type: ignore
        except Exception as exc:
            return None, f"COMET package unavailable: {exc}"
        try:
            model_path = download_model(self.config.cometkiwi_model)
            return load_from_checkpoint(model_path), None
        except Exception as exc:
            return None, f"COMETKiwi model load failed: {exc}"

    def score(self, model: Any, records: list[dict[str, Any]]) -> list[float]:
        data = [{"src": record["source"], "mt": record["reference"]}
                for record in records]
        output = model.predict(
            data,
            batch_size=self.config.cometkiwi_batch_size,
            gpus=self.resolve_gpus(),
        )
        scores = getattr(output, "scores", output)
        return [float(score) for score in scores]

    @staticmethod
    def threshold_for_scores(scores: list[float]) -> float:
        q1 = float(np.percentile(scores, 25))
        q10 = float(np.percentile(scores, 10))
        q3 = float(np.percentile(scores, 75))
        iqr_low = q1 - 1.5 * (q3 - q1)
        return max(q10, iqr_low)

    def apply(
        self,
        records: list[dict[str, Any]],
        qa_results: dict[str, QAResult],
        manifest: dict[str, Any],
    ) -> None:
        if not self.config.enable_cometkiwi or not records:
            manifest["qa"]["cometkiwi_status"] = "disabled"
            return

        model, error = self.load_model()
        if model is None:
            manifest["qa"]["cometkiwi_status"] = "skipped"
            manifest["qa"]["cometkiwi_skip_reason"] = error
            LOGGER.warning("Skipping COMETKiwi: %s", error)
            return

        scores = self.score(model, records)
        threshold = self.threshold_for_scores(scores)

        for record, score in zip(records, scores):
            qa = qa_results[record["id"]]
            qa.cometkiwi_score = round(score, 4)
            qa.cometkiwi_threshold = round(threshold, 4)
            qa.cometkiwi_flagged = score <= threshold
            if qa.cometkiwi_flagged and "cometkiwi_low_score" not in qa.flags:
                qa.flags.append("cometkiwi_low_score")

        manifest["qa"]["cometkiwi_status"] = "completed"
        manifest["qa"]["cometkiwi_model"] = self.config.cometkiwi_model
        manifest["qa"]["cometkiwi_threshold"] = round(threshold, 4)
        manifest["qa"]["cometkiwi_flagged_count"] = sum(
            1 for qa in qa_results.values() if qa.cometkiwi_flagged
        )


class TranslationJudge:
    def __init__(self, config: PipelineConfig, client: Any | None) -> None:
        self.config = config
        self.client = client
        self.audit_rng = random.Random(config.seed + 999)

    @staticmethod
    def qa_context(qa: QAResult | None) -> str:
        if qa is None:
            return "No automated QA hints are available."

        hints = [
            f"deterministic_flags={qa.flags}",
            f"source_reference_word_ratio={qa.source_reference_word_ratio}",
            f"copy_rate={qa.copy_rate}",
        ]
        if qa.cometkiwi_score is not None:
            hints.append(
                "cometkiwi="
                f"score {qa.cometkiwi_score}, "
                f"threshold {qa.cometkiwi_threshold}, "
                f"flagged {qa.cometkiwi_flagged}"
            )
        if qa.heading_numbers_expected or qa.heading_numbers_found:
            hints.append(
                "heading_numbers="
                f"expected {qa.heading_numbers_expected}, "
                f"found {qa.heading_numbers_found}"
            )
        return "\n".join(f"- {hint}" for hint in hints)

    @staticmethod
    def build_messages(
        record: dict[str, Any],
        qa: QAResult | None = None,
    ) -> list[dict[str, str]]:
        print(TranslationJudge.qa_context(qa))
        print(record['reference'])
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
            "requires_regeneration": "boolean",
            "major_issues": ["short English strings"],
            "minor_issues": ["short English strings"],
            "issues": ["combined short English strings"],
            "rationale": "one short English sentence",
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are an impartial English-to-Indonesian translation quality judge. "
                    "Your job is to find translation defects that would make a machine-generated "
                    "Indonesian reference unreliable for model evaluation. Use Indonesian language "
                    "standards and educational translation expectations, but write all issue strings "
                    "and the rationale in English. Return valid JSON only. Do not add markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Evaluate the Indonesian translation against the English source. "
                    "Do not reward literal translation if it creates unnatural Indonesian, and do not "
                    "penalize accepted Indonesian alternatives or standard retained international terms.\n\n"
                    f"{ADAPTED_MQM_NABABAN_BAKER_RUBRIC}\n\n"
                    f"{JUDGE_CALIBRATION_GUIDANCE}\n\n"
                    "Automated QA hints are provided only to focus your inspection. They are not the final verdict:\n"
                    f"{TranslationJudge.qa_context(qa)}\n\n"
                    "Set critical_error=true for wrong language, major omission, hallucination, meaning reversal, "
                    "severe grammar or terminology failure, incoherent Indonesian, or severe structure loss that "
                    "makes the translation unsuitable as a reference. Set requires_regeneration=true if the "
                    "translation should be regenerated before being used as a reference dataset row.\n\n"
                    f"Return JSON with this schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
                    f"English source:\n<source>\n{record['source']}\n</source>\n\n"
                    f"Indonesian translation:\n<translation>\n{record['reference']}\n</translation>"
                ),
            },
        ]

    def should_judge_record(
        self,
        record_id: str,
        qa: QAResult,
        force_ids: set[str] | None = None,
    ) -> bool:
        if force_ids and record_id in force_ids:
            return True
        if self.config.judge_mode == "all":
            return True
        if qa.cometkiwi_flagged:
            return True
        hard_flags = [flag for flag in qa.flags if flag not in {
            "cometkiwi_low_score"}]
        if hard_flags:
            return True
        if self.config.judge_mode == "flagged":
            return False
        return self.audit_rng.random() < self.config.judge_random_audit_rate

    async def judge_record(
        self,
        record: dict[str, Any],
        qa: QAResult | None = None,
    ) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("SEA-LION API client unavailable")
        response = await self.client.chat(
            model=self.config.judge_model,
            messages=self.build_messages(record, qa),
            temperature=0.0,
            max_tokens=1024,
        )
        score = parse_json_from_model_output(response)
        overall = _score_value(score, "overall")
        accuracy = _score_value(
            score, "accuracy", default=_score_value(score, "adequacy"))
        critical = _bool_value(score.get("critical_error", False))
        requires_regeneration = _bool_value(
            score.get("requires_regeneration", False))
        severe_indonesian_issue = any(
            _score_value(score, key, default=5.0) <= 2.0 for key in RUBRIC_SCORE_KEYS
        )
        flagged = (
            critical
            or requires_regeneration
            or overall < self.config.judge_min_overall
            or accuracy < self.config.judge_min_accuracy
            or severe_indonesian_issue
        )
        return {
            "id": record["id"],
            "raw_response": response,
            "score": score,
            "flagged": flagged,
        }

    async def apply(
        self,
        records: list[dict[str, Any]],
        qa_results: dict[str, QAResult],
        manifest: dict[str, Any],
        force_ids: set[str] | None = None,
    ) -> None:
        if not self.config.enable_judge:
            manifest["qa"]["judge_status"] = "disabled"
            return
        if self.client is None:
            manifest["qa"]["judge_status"] = "skipped"
            manifest["qa"]["judge_skip_reason"] = "SEA-LION API client unavailable"
            return

        judged = 0
        flagged = 0
        not_selected = 0
        selected_records: list[dict[str, Any]] = []
        for record in records:
            qa = qa_results[record["id"]]
            if not self.should_judge_record(record["id"], qa, force_ids=force_ids):
                qa.judge_selected = False
                qa.judge_skip_reason = "not_selected_by_judge_mode"
                not_selected += 1
                continue

            judged += 1
            qa.judge_selected = True
            qa.judge_skip_reason = None
            selected_records.append(record)

        concurrency = max(1, self.config.api_concurrency)
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None, Exception | None]:
            async with semaphore:
                try:
                    return (
                        record["id"],
                        await self.judge_record(record, qa_results.get(record["id"])),
                        None,
                    )
                except Exception as exc:
                    return record["id"], None, exc

        tasks = [asyncio.create_task(run_one(record))
                 for record in selected_records]
        for task in asyncio.as_completed(tasks):
            record_id, result, error = await task
            qa = qa_results[record_id]
            if error is not None:
                qa.judge_flagged = True
                flagged += 1
                if "judge_error" not in qa.flags:
                    qa.flags.append("judge_error")
                qa.notes.append(f"judge_error: {error}")
                LOGGER.warning("Judge failed for %s: %s", record_id, error)
                continue

            assert result is not None
            qa.judge_raw_response = result["raw_response"]
            qa.judge_score = result["score"]
            qa.judge_flagged = bool(result["flagged"])
            if qa.judge_flagged:
                flagged += 1
                if "judge_low_score" not in qa.flags:
                    qa.flags.append("judge_low_score")

        manifest["qa"]["judge_status"] = "completed"
        manifest["qa"]["judge_model"] = self.config.judge_model
        manifest["qa"]["judge_mode"] = self.config.judge_mode
        manifest["qa"]["judge_concurrency"] = concurrency
        manifest["qa"]["judge_random_audit_rate"] = self.config.judge_random_audit_rate
        manifest["qa"]["judge_reviewed_count"] = judged
        manifest["qa"]["judge_not_selected_count"] = not_selected
        manifest["qa"]["judge_flagged_count"] = flagged
