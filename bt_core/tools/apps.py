"""Application launching tool for BT.

App names are mapped to actual executables via config.yaml's `apps`
section, since install paths vary by machine and by user profile — this
tool never guesses or hardcodes a path itself.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from bt_core.tools.base import PermissionTier, Tool, ToolError


class OpenAppArgs(BaseModel):
    """Arguments for open_app."""

    name: str = Field(description="The application to open, e.g. 'chrome', 'vscode', 'notepad'")


class OpenAppTool(Tool):
    """Launches a configured application by its friendly name."""

    name = "open_app"
    description = "Open an application by name, e.g. Chrome, VS Code, Excel, Notepad"
    permission_tier = PermissionTier.SAFE

    def __init__(self, timeout_s: float, apps: dict[str, Path]) -> None:
        """Initialize the tool.

        Args:
            timeout_s: Max seconds allowed for the launch to complete.
            apps: Friendly app name -> executable path, from config.yaml.
        """
        super().__init__(timeout_s)
        self._apps = {key.lower(): path for key, path in apps.items()}

    def _args_model(self) -> type[BaseModel]:
        return OpenAppArgs

    async def _run(self, args: OpenAppArgs) -> str:
        app_path = self._apps.get(args.name.strip().lower())
        if app_path is None:
            known = ", ".join(sorted(self._apps))
            raise ToolError(f"I don't know how to open '{args.name}'. I know: {known}")
        await asyncio.to_thread(subprocess.Popen, [str(app_path)])
        return f"Opened {args.name}"
