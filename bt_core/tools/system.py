"""System control tools for BT: get_time and system_command.

system_command performs OS-level actions (shutdown, lock, sleep, volume).
Shutdown/sleep can interrupt unsaved work, so this tool is CONFIRM tier —
BT must not run it without the user's go-ahead.
"""

from __future__ import annotations

import asyncio
import ctypes
import subprocess
from datetime import datetime
from typing import Literal

import pyautogui
from pydantic import BaseModel, Field

from bt_core.tools.base import PermissionTier, Tool, ToolError


class GetTimeArgs(BaseModel):
    """Arguments for get_time — none needed."""


class GetTimeTool(Tool):
    """Returns the current local time."""

    name = "get_time"
    description = "Get the current local time"
    permission_tier = PermissionTier.SAFE

    def _args_model(self) -> type[BaseModel]:
        return GetTimeArgs

    async def _run(self, args: GetTimeArgs) -> str:
        return datetime.now().strftime("%I:%M %p")


SystemAction = Literal["shutdown", "lock", "sleep", "volume_up", "volume_down", "mute"]


class SystemCommandArgs(BaseModel):
    """Arguments for system_command."""

    action: SystemAction = Field(description="The system action to perform")


class SystemCommandTool(Tool):
    """Performs a Windows system action: shutdown, lock, sleep, or volume control."""

    name = "system_command"
    description = "Control the PC: shut down, lock, sleep, or change volume"
    permission_tier = PermissionTier.CONFIRM

    def _args_model(self) -> type[BaseModel]:
        return SystemCommandArgs

    async def _run(self, args: SystemCommandArgs) -> str:
        match args.action:
            case "lock":
                await asyncio.to_thread(ctypes.windll.user32.LockWorkStation)
                return "Locked the PC"
            case "shutdown":
                await asyncio.to_thread(subprocess.run, ["shutdown", "/s", "/t", "5"], check=True)
                return "Shutting down in 5 seconds"
            case "sleep":
                await asyncio.to_thread(
                    subprocess.run,
                    ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                    check=True,
                )
                return "Putting the PC to sleep"
            case "volume_up":
                await asyncio.to_thread(pyautogui.press, "volumeup")
                return "Turned the volume up"
            case "volume_down":
                await asyncio.to_thread(pyautogui.press, "volumedown")
                return "Turned the volume down"
            case "mute":
                await asyncio.to_thread(pyautogui.press, "volumemute")
                return "Muted the volume"
        raise ToolError(f"Unknown system action: {args.action}")
