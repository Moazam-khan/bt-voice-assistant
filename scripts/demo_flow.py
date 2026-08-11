"""Manual test harness for BT's text-to-tool-to-reply loop.

Not part of bt_core — this is a throwaway script to prove the LLM +
tool-calling round trip works end to end before wiring in real audio
(mic isn't built yet). Simulates STT output by taking typed input, and
uses the real Phase 5 tool registry, so this exercises the actual tools
(open_app, open_website, google_search, system_command, get_time).

Run interactively:
    python scripts/demo_flow.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt_core.config import get_settings
from bt_core.llm.client import ChatMessage, OllamaClient
from bt_core.logging_setup import configure_logging, get_logger
from bt_core.tools.base import PermissionTier
from bt_core.tools.registry import ToolRegistry, build_default_registry

log = get_logger(__name__)


async def _handle_turn(
    client: OllamaClient, registry: ToolRegistry, system_prompt: str, user_text: str, model: str
) -> str:
    """Run one full turn: user text -> LLM -> optional tool call -> final reply."""
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_text),
    ]
    first = await client.chat(messages=messages, model=model, tools=registry.schemas())

    if not first.tool_calls:
        return first.content

    call = first.tool_calls[0]
    tool = registry.get_tool(call.name)

    confirmed = True
    if tool is not None and tool.permission_tier != PermissionTier.SAFE:
        answer = input(f"  BT wants to run '{call.name}' with {call.arguments}. Allow? (y/n) ")
        confirmed = answer.strip().lower() == "y"

    result = await registry.execute(call.name, call.arguments, confirmed=confirmed)
    print(f"  [tool call] {call.name}({call.arguments}) -> {result.message}")

    messages.append(ChatMessage(role="assistant", content=first.content))
    messages.append(ChatMessage(role="tool", content=result.message, tool_name=call.name))
    second = await client.chat(messages=messages, model=model, tools=registry.schemas())
    return second.content


async def main() -> None:
    """Interactive loop: type a message, see BT's tool decision and reply."""
    settings = get_settings()
    configure_logging(settings.logging)
    system_prompt = (settings.paths.prompts_dir / "system.txt").read_text(encoding="utf-8")
    client = OllamaClient(settings.llm)
    registry = build_default_registry(settings)

    print("BT demo flow. Type a message (or 'quit'). Try: 'open chrome' or 'what time is it?'")
    while True:
        user_text = input("> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        reply = await _handle_turn(client, registry, system_prompt, user_text, settings.llm.main_model)
        print(f"BT: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
