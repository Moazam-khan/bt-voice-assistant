"""Tool registry for BT: aggregates all tools and dispatches LLM tool calls.

Enforces each tool's permission tier: SAFE tools always run; CONFIRM tools
only run when the caller passes confirmed=True; ADMIN tools additionally
require admin_enabled. The registry does not implement the confirmation
conversation itself — that belongs to the orchestrator (Phase 7), which
owns the voice loop and can actually ask the user aloud.
"""

from __future__ import annotations

from typing import Any

from bt_core.config import BTSettings
from bt_core.logging_setup import get_logger
from bt_core.tools.apps import OpenAppTool
from bt_core.tools.base import PermissionTier, Tool, ToolResult
from bt_core.tools.system import GetTimeTool, SystemCommandTool
from bt_core.tools.web import GoogleSearchTool, OpenWebsiteTool

log = get_logger(__name__)


class ToolRegistry:
    """Holds all of BT's tools and dispatches calls to them by name."""

    def __init__(self, tools: list[Tool], admin_enabled: bool = False) -> None:
        """Initialize the registry.

        Args:
            tools: The tools BT can execute, already constructed with
                their config-driven dependencies.
            admin_enabled: Whether ADMIN-tier tools may run at all. Off by
                default per this project's least-privilege standard.
        """
        self._tools = {tool.name: tool for tool in tools}
        self._admin_enabled = admin_enabled

    def schemas(self) -> list[dict[str, Any]]:
        """Return every tool's schema, for OllamaClient.chat(tools=...)."""
        return [tool.schema() for tool in self._tools.values()]

    def get_tool(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    async def execute(
        self, name: str, arguments: dict[str, Any], confirmed: bool = False
    ) -> ToolResult:
        """Run a tool by name, enforcing its permission tier.

        Args:
            name: The tool's registered name.
            arguments: Raw arguments from the LLM's tool call.
            confirmed: Whether the user has already approved this specific
                call. Required for CONFIRM/ADMIN-tier tools.

        Returns:
            The tool's result, or a ToolResult explaining why it didn't
            run (unknown tool, needs confirmation, or admin disabled).
        """
        tool = self._tools.get(name)
        if tool is None:
            log.warning("tool_not_found", tool=name)
            return ToolResult(success=False, message=f"I don't have a tool called {name}.")

        if tool.permission_tier == PermissionTier.ADMIN and not self._admin_enabled:
            log.warning("tool_blocked_admin_disabled", tool=name)
            return ToolResult(success=False, message=f"{name} requires admin mode, which is off.")

        if tool.permission_tier != PermissionTier.SAFE and not confirmed:
            log.info("tool_needs_confirmation", tool=name, arguments=arguments)
            return ToolResult(success=False, message=f"{name} needs your confirmation before I run it.")

        return await tool.execute(arguments)


def build_default_registry(settings: BTSettings) -> ToolRegistry:
    """Build the registry with BT's core tools, wired to their config.

    Args:
        settings: BT's full validated settings.

    Returns:
        A ToolRegistry containing get_time, open_app, open_website,
        google_search, and system_command.
    """
    timeout_s = settings.tools.timeout_s
    tools: list[Tool] = [
        GetTimeTool(timeout_s=timeout_s),
        OpenAppTool(timeout_s=timeout_s, apps=settings.apps),
        OpenWebsiteTool(timeout_s=timeout_s),
        GoogleSearchTool(timeout_s=timeout_s),
        SystemCommandTool(timeout_s=timeout_s),
    ]
    return ToolRegistry(tools=tools)
