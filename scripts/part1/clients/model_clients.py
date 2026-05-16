"""Model client factories for API generation and offline judging."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, AsyncIterator

from .sea_lion_client import SeaLionClient
from ..config.data_object import PipelineConfig


LOGGER = logging.getLogger("prepare_translation_dataset")


def quiet_model_loading_logs() -> None:
    """Reduce third-party model-loading noise while keeping pipeline logs visible."""

    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for logger_name in (
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "transformers",
        "transformers.modeling_utils",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        pass


class Qwen3VLChatClient:
    """Text-only chat adapter for Qwen3-VL local inference.

    The imports and model load are intentionally lazy so the normal SEA-LION API
    generation path stays light on local machines.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.model_id = config.qwen3vl_model
        self.dtype_name = config.qwen3vl_dtype
        self.device_map = config.qwen3vl_device_map
        self.attn_implementation = config.qwen3vl_attn_implementation
        self.load_in_4bit = config.qwen3vl_load_in_4bit
        self.bnb_4bit_quant_type = config.qwen3vl_bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = config.qwen3vl_bnb_4bit_use_double_quant
        self.max_new_tokens = config.qwen3vl_max_new_tokens
        self.processor: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None

    async def __aenter__(self) -> "Qwen3VLChatClient":
        await asyncio.to_thread(self._load)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None

    def _torch_dtype(self) -> Any:
        assert self.torch is not None
        if self.dtype_name == "auto":
            return "auto" if not self.load_in_4bit else self.torch.float16
        if not hasattr(self.torch, self.dtype_name):
            raise ValueError(f"Unsupported torch dtype: {self.dtype_name}")
        return getattr(self.torch, self.dtype_name)

    def _quantization_config(self) -> Any | None:
        if not self.load_in_4bit:
            return None
        assert self.torch is not None
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "qwen3vl_load_in_4bit=true requires transformers with bitsandbytes support. "
                "Install bitsandbytes and a recent transformers version."
            ) from exc
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self._torch_dtype(),
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        quiet_model_loading_logs()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        model_kwargs: dict[str, Any] = {
            "torch_dtype": self._torch_dtype(),
            "device_map": self.device_map,
        }
        quantization_config = self._quantization_config()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        LOGGER.info(
            "Loading local judge model: %s (4bit=%s, device_map=%s, dtype=%s)",
            self.model_id,
            self.load_in_4bit,
            self.device_map,
            self.dtype_name,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **model_kwargs)
        self.model.eval()

    @staticmethod
    def _to_qwen_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        qwen_messages = []
        for message in messages:
            qwen_messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )
        return qwen_messages

    def _chat_sync(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        if self.processor is None or self.model is None or self.torch is None:
            self._load()
        assert self.processor is not None
        assert self.model is not None
        assert self.torch is not None

        qwen_messages = self._to_qwen_messages(messages)
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retries: int = 0,
    ) -> str:
        _ = model, temperature, retries
        return await asyncio.to_thread(self._chat_sync, messages, max_tokens)


@asynccontextmanager
async def open_sealion_client(config: PipelineConfig) -> AsyncIterator[SeaLionClient]:
    import os

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in {config.api_key_env}. Add it to .env or export it in the environment."
        )
    client = SeaLionClient(
        api_key=api_key,
        base_url=config.api_base_url,
        requests_per_minute=config.requests_per_minute,
        timeout_seconds=config.timeout_seconds,
        max_connections=config.api_concurrency,
    )
    async with client:
        yield client


@asynccontextmanager
async def open_generator_client(config: PipelineConfig) -> AsyncIterator[SeaLionClient]:
    if config.generator_provider != "sealion_api":
        raise ValueError(
            f"Unsupported generator provider: {config.generator_provider}")
    async with open_sealion_client(config) as client:
        yield client


def open_judge_client(config: PipelineConfig) -> AbstractAsyncContextManager[Any]:
    if config.judge_provider == "sealion_api":
        return open_sealion_client(config)
    if config.judge_provider == "qwen3vl_local":
        return Qwen3VLChatClient(config)
    raise ValueError(f"Unsupported judge provider: {config.judge_provider}")
