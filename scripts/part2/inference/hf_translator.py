"""Hugging Face chat-model inference for English-to-Indonesian translation."""

from __future__ import annotations

import logging
from typing import Any

from scripts.part1.translation.generation import INDONESIAN_TRANSLATION_GUIDANCE

from ..config import EvaluationInput, Part2Config, PredictionRecord
from ..utils import clean_translation_output, word_count


LOGGER = logging.getLogger("part2_evaluation")


TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional English-to-Indonesian translator for educational and textbook-style "
    "materials. Produce faithful, fluent Indonesian that reads like standard written Indonesian, "
    "not a word-for-word English rendering.\n\n"
    f"{INDONESIAN_TRANSLATION_GUIDANCE}\n\n"
    "Return only the Indonesian translation. Do not add commentary, explanations, labels, or "
    "markdown fences."
)


TRANSLATION_USER_TEMPLATE = (
    "Translate the following English educational text into standard formal Indonesian. "
    "Apply all Indonesian translation requirements from the system message.\n\n"
    "<text>\n"
    "{source}\n"
    "</text>"
)


class HuggingFaceTranslator:
    def __init__(self, config: Part2Config) -> None:
        self.config = config
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None

    def _torch_dtype(self) -> Any:
        if self.torch is None or self.config.torch_dtype == "auto":
            return "auto"
        if not hasattr(self.torch, self.config.torch_dtype):
            raise ValueError(
                f"Unsupported torch dtype: {self.config.torch_dtype}")
        return getattr(self.torch, self.config.torch_dtype)

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        LOGGER.info("Loading model: %s", self.config.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {"torch_dtype": self._torch_dtype()}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = self.config.device_map
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )
        self.model.eval()

    def _model_device(self) -> Any:
        assert self.model is not None
        return next(self.model.parameters()).device

    def build_prompt(self, source: str) -> str:
        assert self.tokenizer is not None
        messages = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": TRANSLATION_USER_TEMPLATE.format(
                source=source)},
        ]
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"System: {TRANSLATION_SYSTEM_PROMPT}\n\n"
            f"User: {TRANSLATION_USER_TEMPLATE.format(source=source)}\n\nAssistant:"
        )

    def translate_batch(self, rows: list[EvaluationInput]) -> list[PredictionRecord]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.torch is not None

        prompts = [self.build_prompt(row.source) for row in rows]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(self._model_device())
                   for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])

        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        hypotheses = self.tokenizer.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )

        predictions = []
        for row, hypothesis in zip(rows, hypotheses):
            cleaned = clean_translation_output(hypothesis)
            predictions.append(
                PredictionRecord(
                    id=row.id,
                    model_id=self.config.model_id,
                    source=row.source,
                    reference=row.reference,
                    hypothesis=cleaned,
                    source_language=row.source_language,
                    target_language=row.target_language,
                    source_word_count=row.source_word_count,
                    source_char_count=row.source_char_count,
                    source_length_bucket=row.source_length_bucket,
                    hypothesis_word_count=word_count(cleaned),
                )
            )
        return predictions


class PredictionRunner:
    def __init__(self, config: Part2Config) -> None:
        self.config = config
        self.translator = HuggingFaceTranslator(config)

    def run(
        self,
        rows: list[EvaluationInput],
        existing: dict[str, dict[str, Any]] | None = None,
    ) -> list[PredictionRecord]:
        existing = existing or {}
        predictions: list[PredictionRecord] = []
        pending: list[EvaluationInput] = []

        for row in rows:
            if self.config.resume and row.id in existing:
                predictions.append(PredictionRecord(**existing[row.id]))
            else:
                pending.append(row)

        for start in range(0, len(pending), self.config.batch_size):
            batch = pending[start:start + self.config.batch_size]
            try:
                predictions.extend(self.translator.translate_batch(batch))
            except Exception as exc:
                LOGGER.exception(
                    "Inference failed for batch starting at %s", start)
                for row in batch:
                    predictions.append(
                        PredictionRecord(
                            id=row.id,
                            model_id=self.config.model_id,
                            source=row.source,
                            reference=row.reference,
                            hypothesis="",
                            source_language=row.source_language,
                            target_language=row.target_language,
                            source_word_count=row.source_word_count,
                            source_char_count=row.source_char_count,
                            source_length_bucket=row.source_length_bucket,
                            hypothesis_word_count=0,
                            inference_error=str(exc),
                        )
                    )

            LOGGER.info("Predicted %s/%s rows", len(predictions), len(rows))

        return sorted(predictions, key=lambda record: record.id)
