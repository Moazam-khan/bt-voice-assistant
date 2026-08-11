"""Shared infrastructure for every tool BT can execute.

Concrete tools (bt_core/tools/apps.py, web.py, system.py, ...) inherit from
Tool and only implement _args_model() and _run(). The base class handles
what every tool needs identically: building the LLM function-calling
schema, validating arguments, enforcing a timeout, catching failures into a
spoken-friendly message instead of a raw traceback, and structured logging
of every execution.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel

from bt_core.logging_setup import get_logger

log = get_logger(__name__)


class PermissionTier(str, Enum):
    """How much trust a tool requires before BT may run it."""

    SAFE = "safe"
    CONFIRM = "confirm"
    ADMIN = "admin"


class ToolResult(BaseModel):
    """The outcome of running a tool — never raises, always speakable."""

    success: bool
    message: str


class ToolError(Exception):
    """Raise from _run() for an expected, user-facing failure.

    The message is spoken to the user as-is (e.g. "I don't know how to
    open 'foo'. I know: chrome, notepad..."), unlike an unexpected
    exception, which is logged with a traceback but replaced with a
    generic message so internal details never reach the user.
    """


class Tool(ABC):
    """Base class for every tool BT can execute.

    Attributes:
        name: The tool's identifier, used by the LLM to call it.
        description: One sentence describing what the tool does, shown to
            the LLM to help it decide when to use this tool.
        permission_tier: How much trust this tool requires. Fixed per tool
            by its inherent risk, not user-configurable.
    """

    name: str
    description: str
    permission_tier: PermissionTier

    def __init__(self, timeout_s: float) -> None:
        """Initialize the tool.

        Args:
            timeout_s: Max seconds this tool may run before being aborted.
                Comes from config (tools.timeout_s), not hardcoded.
        """
        self._timeout_s = timeout_s

    def schema(self) -> dict[str, Any]:
        """Build this tool's Ollama function-calling schema.

        Returns:
            A dict in Ollama's tool schema format, derived from this
            tool's Pydantic argument model.
        """
        json_schema = self._args_model().model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", []),
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate arguments, run the tool under a timeout, and log the outcome.

        Args:
            arguments: Raw arguments from the LLM's tool call.

        Returns:
            A ToolResult. Never raises — validation errors, timeouts, and
            unexpected failures are all captured as a spoken-friendly
            failure message instead of propagating a traceback.
        """
        start = time.monotonic()
        try:
            args = self._args_model()(**arguments)
        except Exception:
            log.error("tool_invalid_arguments", tool=self.name, arguments=arguments, exc_info=True)
            return ToolResult(success=False, message=f"I got confused about what to pass to {self.name}.")

        try:
            message = await asyncio.wait_for(self._run(args), timeout=self._timeout_s)
            latency_ms = int((time.monotonic() - start) * 1000)
            log.info(
                "tool_executed",
                tool=self.name,
                arguments=arguments,
                success=True,
                latency_ms=latency_ms,
            )
            return ToolResult(success=True, message=message)
        except TimeoutError:
            latency_ms = int((time.monotonic() - start) * 1000)
            log.error("tool_timeout", tool=self.name, arguments=arguments, latency_ms=latency_ms)
            return ToolResult(success=False, message=f"{self.name} took too long, so I gave up.")
        except ToolError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            log.warning(
                "tool_expected_failure",
                tool=self.name,
                arguments=arguments,
                latency_ms=latency_ms,
                reason=str(exc),
            )
            return ToolResult(success=False, message=str(exc))
        except Exception:
            latency_ms = int((time.monotonic() - start) * 1000)
            log.error(
                "tool_failed",
                tool=self.name,
                arguments=arguments,
                latency_ms=latency_ms,
                exc_info=True,
            )
            return ToolResult(success=False, message=f"Something went wrong running {self.name}.")

    @abstractmethod
    def _args_model(self) -> type[BaseModel]:
        """Return the Pydantic model describing this tool's arguments."""

    @abstractmethod
    async def _run(self, args: BaseModel) -> str:
        """Perform the tool's action.

        Args:
            args: Validated arguments, an instance of ``self._args_model()``.

        Returns:
            A spoken-friendly summary of what happened.
        """
