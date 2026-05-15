"""Async OpenAI-compatible client for the SEA-LION API."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from ..config.data_object import DEFAULT_API_BASE_URL


LOGGER = logging.getLogger("prepare_translation_dataset")


class AsyncRateLimiter:
    """Spaces request starts so concurrent tasks still respect a requests/minute cap."""

    def __init__(self, requests_per_minute: int) -> None:
        self.min_interval = 60.0 / max(1, requests_per_minute)
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request_time = asyncio.get_running_loop().time()


class SeaLionClient:
    """Small async chat-completions client backed by httpx."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_API_BASE_URL,
        requests_per_minute: int = 10,
        timeout_seconds: int = 120,
        max_connections: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.requests_per_minute = max(1, requests_per_minute)
        self.timeout_seconds = timeout_seconds
        self.max_connections = max(1, max_connections)
        self.rate_limiter = AsyncRateLimiter(self.requests_per_minute)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SeaLionClient":
        timeout = httpx.Timeout(self.timeout_seconds)
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_connections,
        )
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "SeaLionClient must be used as an async context manager")
        return self._client

    @staticmethod
    def _normalize_content(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            content = value.strip()
            return content or None
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
            content = "".join(parts).strip()
            return content or None
        if isinstance(value, dict):
            for key in ("text", "content", "output_text", "reasoning_content"):
                content = SeaLionClient._normalize_content(value.get(key))
                if content:
                    return content
        content = str(value).strip()
        return content or None

    @classmethod
    def _extract_chat_content(cls, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    for key in ("content", "text", "output_text", "reasoning_content"):
                        content = cls._normalize_content(message.get(key))
                        if content:
                            return content
                for key in ("content", "text", "output_text", "reasoning_content"):
                    content = cls._normalize_content(choice.get(key))
                    if content:
                        return content

        for key in ("content", "text", "output_text", "response", "reasoning_content"):
            content = cls._normalize_content(data.get(key))
            if content:
                return content

        preview = json.dumps(data, ensure_ascii=False)[:1000]
        raise RuntimeError(
            f"SEA-LION API response did not include assistant content: {preview}")

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(retries + 1):
            await self.rate_limiter.wait()
            try:
                response = await self.client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return self._extract_chat_content(data)
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code in {
                    429, 500, 502, 503, 504}
                detail = exc.response.text[:500]
                if not retryable or attempt >= retries:
                    raise RuntimeError(
                        f"SEA-LION API error {exc.response.status_code}: {detail}"
                    ) from exc
                sleep_seconds = min(60, (2**attempt) * 5)
                LOGGER.warning(
                    "Retrying SEA-LION API call after HTTP %s (%ss)",
                    exc.response.status_code,
                    sleep_seconds,
                )
                await asyncio.sleep(sleep_seconds)
            except (httpx.RequestError, TimeoutError) as exc:
                if attempt >= retries:
                    raise RuntimeError(
                        f"SEA-LION API request failed: {exc}") from exc
                sleep_seconds = min(60, (2**attempt) * 5)
                LOGGER.warning(
                    "Retrying SEA-LION API call after network error (%ss)", sleep_seconds)
                await asyncio.sleep(sleep_seconds)

        raise RuntimeError("SEA-LION API call failed unexpectedly")
