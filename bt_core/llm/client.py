"""Ollama LLM client for BT.

Thin async wrapper around ollama.AsyncClient (which is natively async, so
no asyncio.to_thread is needed here) — adds a configurable timeout and
structured logging with latency, and returns BT's own typed Pydantic models
instead of the ollama library's internal types, so nothing downstream
depends on ollama's type definitions directly.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import ollama
from pydantic import BaseModel

from bt_core.config import LlmConfig
from bt_core.logging_setup import get_logger

log = get_logger(__name__)

_RETRY_DELAY_S = 0.5


class ToolCall(BaseModel):
    """A single tool invocation requested by the LLM."""

    name: str
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    """A single message in a chat exchange with the LLM."""

    role: str
    content: str = ""
    tool_name: str | None = None


class ChatResult(BaseModel):
    """The LLM's response to a chat call."""

    content: str
    tool_calls: list[ToolCall] = []
    model: str
    latency_ms: int


class OllamaClient:
    """Async client for chatting with local Ollama models, with tool calling."""

    def __init__(self, config: LlmConfig) -> None:
        """Initialize the client.

        Args:
            config: LLM section of BT's settings (host, model names,
                timeout, temperature).
        """
        self._config = config
        self._client = ollama.AsyncClient(host=config.host)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """Send a chat request to Ollama and return the model's reply.

        Args:
            messages: Conversation history, oldest first.
            model: Model name to use. Defaults to ``config.main_model``.
            tools: Tool schemas the model may call, in Ollama's function-
                calling format (see ``bt_core.tools`` for schema builders).
                Pass None for a plain chat turn with no tool access.

        Returns:
            The parsed response: text content and/or requested tool calls.

        Raises:
            TimeoutError: If the model doesn't respond within
                ``config.timeout_s``, on the second attempt (one retry is
                attempted first for transient failures — a slow local
                Ollama daemon under load timing out once doesn't
                automatically mean the request itself is bad).
        """
        target_model = model or self._config.main_model
        raw_messages = [m.model_dump(exclude_none=True) for m in messages]

        start = time.monotonic()
        try:
            response = await self._chat_once(target_model, raw_messages, tools)
        except (TimeoutError, ConnectionError) as exc:
            log.warning(
                "llm_chat_failed_retrying",
                model=target_model,
                error=type(exc).__name__,
            )
            await asyncio.sleep(_RETRY_DELAY_S)
            try:
                response = await self._chat_once(target_model, raw_messages, tools)
            except (TimeoutError, ConnectionError):
                log.error(
                    "llm_chat_failed_after_retry",
                    model=target_model,
                    timeout_s=self._config.timeout_s,
                )
                raise
        latency_ms = int((time.monotonic() - start) * 1000)

        tool_calls = [
            ToolCall(name=call.function.name, arguments=dict(call.function.arguments))
            for call in (response.message.tool_calls or [])
        ]
        result = ChatResult(
            content=response.message.content or "",
            tool_calls=tool_calls,
            model=target_model,
            latency_ms=latency_ms,
        )
        log.info(
            "llm_chat_completed",
            model=target_model,
            latency_ms=latency_ms,
            tool_call_count=len(tool_calls),
            content_length=len(result.content),
        )
        return result

    async def _chat_once(
        self,
        model: str,
        raw_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ollama.ChatResponse:
        """Make a single chat attempt against Ollama, no retry.

        Args:
            model: Model name to use.
            raw_messages: Messages already converted to plain dicts.
            tools: Tool schemas, or None.

        Returns:
            The raw ollama response.

        Raises:
            TimeoutError: If Ollama doesn't respond within config.timeout_s.
        """
        return await asyncio.wait_for(
            self._client.chat(
                model=model,
                messages=raw_messages,
                tools=tools,
                options={"temperature": self._config.temperature},
            ),
            timeout=self._config.timeout_s,
        )
