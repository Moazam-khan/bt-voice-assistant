"""Local system resource stats for BT's sidebar widget.

Unlike weather, this needs no internet and no external service — psutil
reads CPU/RAM usage directly from the OS. psutil's calls are blocking
(cpu_percent() intentionally blocks for `interval` seconds to measure
accurately), so wrapped in asyncio.to_thread despite being cheap.
"""

from __future__ import annotations

import asyncio

import psutil
from pydantic import BaseModel

_CPU_SAMPLE_INTERVAL_S = 0.3


class SystemStats(BaseModel):
    """A snapshot of local CPU/RAM usage."""

    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float


async def get_system_stats() -> SystemStats:
    """Get a current snapshot of CPU and RAM usage.

    Returns:
        Current system resource usage.
    """
    return await asyncio.to_thread(_snapshot)


def _snapshot() -> SystemStats:
    """Blocking psutil read, run off the event loop via to_thread."""
    cpu_percent = psutil.cpu_percent(interval=_CPU_SAMPLE_INTERVAL_S)
    memory = psutil.virtual_memory()
    return SystemStats(
        cpu_percent=cpu_percent,
        ram_percent=memory.percent,
        ram_used_gb=memory.used / (1024**3),
        ram_total_gb=memory.total / (1024**3),
    )
