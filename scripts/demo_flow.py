"""Manual test harness for BT's text-to-tool-to-reply loop.

Not part of bt_core — this is a throwaway script to prove the LLM +
tool-calling round trip works end to end before wiring in real audio
(mic/TTS aren't built yet). Simulates STT output by taking typed input.

Run interactively:
    python scripts/demo_flow.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt_core.config import get_settings
from bt_core.llm.client import ChatMessage, OllamaClient
from bt_core.logging_setup import configure_logging, get_logger

log = get_logger(__name__)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a real tool by name. Placeholder until Phase 5 builds the registry."""
    if name == "get_time":
        return datetime.now().strftime("%I:%M %p")
    return f"Unknown tool: {name}"


async def _handle_turn(client: OllamaClient, system_prompt: str, user_text: str, model: str) -> str:
    """Run one full turn: user text -> LLM -> optional tool call -> final reply."""
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_text),
    ]
    first = await client.chat(messages=messages, model=model, tools=_TOOLS)

    if not first.tool_calls:
        return first.content

    call = first.tool_calls[0]
    tool_result = _execute_tool(call.name, call.arguments)
    print(f"  [tool call] {call.name}({call.arguments}) -> {tool_result}")

    messages.append(ChatMessage(role="assistant", content=first.content))
    messages.append(ChatMessage(role="tool", content=tool_result, tool_name=call.name))
    second = await client.chat(messages=messages, model=model, tools=_TOOLS)
    return second.content


async def main() -> None:
    """Interactive loop: type a message, see BT's tool decision and reply."""
    settings = get_settings()
    configure_logging(settings.logging)
    system_prompt = (settings.paths.prompts_dir / "system.txt").read_text(encoding="utf-8")
    client = OllamaClient(settings.llm)

    print("BT demo flow. Type a message (or 'quit'). Try: 'what time is it?'")
    while True:
        user_text = input("> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        reply = await _handle_turn(client, system_prompt, user_text, settings.llm.main_model)
        print(f"BT: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
