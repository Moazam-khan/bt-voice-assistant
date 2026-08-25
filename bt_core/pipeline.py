"""BT's core conversation pipeline: audio -> VAD -> STT -> LLM -> tools -> TTS.

Pipeline owns the per-utterance state machine: it buffers audio between a
VAD speech-start and speech-end event, transcribes the buffered utterance,
runs it through the LLM (with tool calling, up to a few rounds), and
synthesizes the final reply. Each stage's failure is caught and logged
with a fallback spoken response — one bad turn never crashes the loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from enum import Enum

import numpy as np

from bt_core.audio.vad import SpeechEvent, VoiceActivityDetector
from bt_core.audio.wakeword import WakeWordDetector
from bt_core.llm.client import ChatMessage, OllamaClient
from bt_core.logging_setup import get_logger
from bt_core.memory.structured import ConversationStore, ConversationTurn
from bt_core.memory.vector import SemanticMemory
from bt_core.stt.transcriber import Transcriber
from bt_core.tools.base import PermissionTier, ToolResult
from bt_core.tools.registry import ToolRegistry
from bt_core.tts.synthesizer import Synthesizer

log = get_logger(__name__)

_MAX_TOOL_ROUNDS = 3
_FALLBACK_REPLY = "Sorry, something went wrong on my end."
_EMPTY_REPLY_FALLBACK = "Done."


class _ListenState(Enum):
    """Whether BT is idle (waiting for its wake word) or actively listening."""

    WAITING_FOR_WAKE_WORD = "waiting_for_wake_word"
    LISTENING_FOR_COMMAND = "listening_for_command"


class Pipeline:
    """Wires wake word, VAD, STT, LLM, tools, and TTS into one conversation loop.

    BT stays idle (only running wake word detection) until its wake phrase
    is heard, then listens for one command via VAD, handles it, and returns
    to idle — it does not respond to everything it hears.
    """

    def __init__(
        self,
        wake_word: WakeWordDetector,
        vad: VoiceActivityDetector,
        transcriber: Transcriber,
        llm_client: OllamaClient,
        tool_registry: ToolRegistry,
        synthesizer: Synthesizer,
        conversation_store: ConversationStore,
        semantic_memory: SemanticMemory,
        system_prompt: str,
        main_model: str,
        history_turns: int,
        on_status_change: Callable[[str], None] | None = None,
        on_user_text: Callable[[str], None] | None = None,
        on_assistant_text: Callable[[str], None] | None = None,
        on_tool_used: Callable[[str, bool, str], None] | None = None,
        on_confirmation_needed: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialize the pipeline with its already-built dependencies.

        Args:
            wake_word: Wake word detector, already loaded from config.
            vad: Voice activity detector, already constructed from config.
            transcriber: STT engine, already loaded via Transcriber.load().
            llm_client: Ollama chat client.
            tool_registry: BT's available tools.
            synthesizer: TTS engine, already loaded via Synthesizer.load().
            conversation_store: SQLite conversation history.
            semantic_memory: ChromaDB semantic recall.
            system_prompt: BT's system prompt text.
            main_model: The LLM model name to use for tool-calling turns.
            history_turns: How many recent user/assistant turns to replay
                as context on every LLM call, so BT remembers what was
                just said in this conversation — not just the fuzzy,
                topic-based recall semantic_memory provides.
            on_status_change: Optional UI hook, called with one of "idle",
                "listening", "thinking", "speaking" as the pipeline's state
                changes. The pipeline has no awareness of what (if
                anything) is listening — e.g. a chat window.
            on_user_text: Optional UI hook, called with the transcribed
                user utterance once STT completes.
            on_assistant_text: Optional UI hook, called with BT's final
                text reply once the LLM/tool loop completes.
            on_tool_used: Optional UI hook, called with (tool_name, success,
                message) every time a tool actually runs — including tools
                that were denied confirmation, so the UI can show what was
                attempted either way.
            on_confirmation_needed: Optional UI hook, called with
                (tool_name, description) when a CONFIRM-tier tool wants to
                run. The pipeline then awaits :meth:`respond_confirmation`
                being called back before continuing — the hook itself is
                fire-and-forget, only used to tell the UI to show a prompt.
        """
        self._wake_word = wake_word
        self._vad = vad
        self._transcriber = transcriber
        self._llm = llm_client
        self._tools = tool_registry
        self._tts = synthesizer
        self._conversation = conversation_store
        self._memory = semantic_memory
        self._system_prompt = system_prompt
        self._main_model = main_model
        self._history_turns = history_turns
        self._on_status_change = on_status_change or (lambda status: None)
        self._on_user_text = on_user_text or (lambda text: None)
        self._on_assistant_text = on_assistant_text or (lambda text: None)
        self._on_tool_used = on_tool_used or (lambda name, success, message: None)
        self._on_confirmation_needed = on_confirmation_needed or (lambda name, description: None)
        self._state = _ListenState.WAITING_FOR_WAKE_WORD
        self._recording = False
        self._utterance_buffer: list[np.ndarray] = []
        self._confirmation_future: asyncio.Future[bool] | None = None

    def trigger_listening(self) -> None:
        """Manually start listening for a command, bypassing the wake word.

        For UI-driven activation (e.g. a "Start" button) instead of
        saying the wake phrase. No-op if already listening for a command.
        """
        if self._state == _ListenState.WAITING_FOR_WAKE_WORD:
            self._vad.reset()
            self._state = _ListenState.LISTENING_FOR_COMMAND
            self._on_status_change("listening")

    def respond_confirmation(self, allowed: bool) -> None:
        """Resolve a pending CONFIRM-tier tool prompt from the UI.

        Called externally (e.g. from an Allow/Deny button) once per
        confirmation request. A no-op if nothing is currently pending, or
        if it's somehow called twice for the same prompt — the tool-calling
        loop only ever awaits one confirmation at a time.

        Args:
            allowed: Whether the user approved the pending tool call.
        """
        if self._confirmation_future is not None and not self._confirmation_future.done():
            self._confirmation_future.set_result(allowed)

    async def _request_confirmation(self, tool_name: str, description: str) -> bool:
        """Ask the UI to confirm a tool call and wait for the response.

        Args:
            tool_name: The tool requesting confirmation.
            description: The tool's one-line description, shown to the user
                so they know what they're approving.

        Returns:
            Whether the user allowed the tool to run.
        """
        self._confirmation_future = asyncio.get_running_loop().create_future()
        self._on_confirmation_needed(tool_name, description)
        try:
            return await self._confirmation_future
        finally:
            self._confirmation_future = None

    async def handle_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        """Feed one audio chunk through the wake word / VAD state machine.

        Args:
            chunk: A raw audio chunk from the microphone.

        Returns:
            Synthesized reply audio if this chunk completed a full
            wake-word -> command -> response turn, else None.
        """
        if self._state == _ListenState.WAITING_FOR_WAKE_WORD:
            if self._wake_word.process(chunk):
                self._wake_word.reset()
                self._vad.reset()
                self._state = _ListenState.LISTENING_FOR_COMMAND
                self._on_status_change("listening")
            return None

        events = self._vad.process(chunk)
        mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk

        if SpeechEvent.START in events:
            self._recording = True
            self._utterance_buffer = [mono.astype(np.float32)]
        elif self._recording:
            self._utterance_buffer.append(mono.astype(np.float32))

        if SpeechEvent.END in events and self._recording:
            self._recording = False
            self._state = _ListenState.WAITING_FOR_WAKE_WORD
            audio = np.concatenate(self._utterance_buffer)
            self._utterance_buffer = []
            return await self.handle_utterance(audio)
        return None

    async def handle_utterance(self, audio: np.ndarray) -> np.ndarray:
        """Run one full turn: audio -> text -> LLM/tools -> reply audio.

        Args:
            audio: A complete utterance's worth of mono float32 audio.

        Returns:
            Synthesized speech for BT's reply. Falls back to a spoken
            apology if any stage fails, rather than raising.
        """
        self._on_status_change("thinking")
        try:
            text = await self._transcriber.transcribe(audio)
        except Exception:
            log.error("pipeline_stt_failed", exc_info=True)
            return await self._synthesize_safely(_FALLBACK_REPLY)

        if not text:
            log.info("pipeline_empty_transcription")
            self._on_status_change("idle")
            return np.zeros(0, dtype=np.float32)

        return await self._handle_text(text)

    async def handle_text(self, text: str) -> np.ndarray:
        """Run one full turn from typed text instead of spoken audio.

        For a UI text box as an alternative input to voice — goes through
        the identical LLM/tools/memory/TTS path handle_utterance uses
        after transcription, so typing and speaking behave identically.

        Args:
            text: The user's typed message.

        Returns:
            Synthesized speech for BT's reply, played back the same way a
            spoken command's reply would be.
        """
        self._on_status_change("thinking")
        return await self._handle_text(text)

    async def _handle_text(self, text: str) -> np.ndarray:
        """Shared LLM/tools/memory/TTS logic for both voice and typed input."""
        self._on_user_text(text)
        reply_text = await self._run_llm_turn(text)
        self._on_assistant_text(reply_text)

        now = datetime.now()
        await self._conversation.add_turn(ConversationTurn(role="user", content=text, timestamp=now))
        await self._conversation.add_turn(
            ConversationTurn(role="assistant", content=reply_text, timestamp=now)
        )
        await self._memory.remember(f"User said: {text}\nBT replied: {reply_text}")

        self._on_status_change("speaking")
        result = await self._synthesize_safely(reply_text)
        self._on_status_change("idle")
        return result

    async def _run_llm_turn(self, user_text: str) -> str:
        """Run the LLM + tool-calling loop for one user utterance.

        Args:
            user_text: The transcribed user request.

        Returns:
            BT's final text reply, ready to be spoken.
        """
        messages = [ChatMessage(role="system", content=self._system_prompt)]

        matches = await self._memory.recall(user_text, limit=3)
        if matches:
            context = "\n".join(f"- {m.text}" for m in matches)
            messages.append(
                ChatMessage(role="system", content=f"Relevant past context:\n{context}")
            )

        recent_turns = await self._conversation.get_recent(limit=self._history_turns)
        messages.extend(ChatMessage(role=turn.role, content=turn.content) for turn in recent_turns)

        messages.append(ChatMessage(role="user", content=user_text))
        try:
            for _ in range(_MAX_TOOL_ROUNDS):
                result = await self._llm.chat(
                    messages=messages, model=self._main_model, tools=self._tools.schemas()
                )
                if not result.tool_calls:
                    return result.content or _EMPTY_REPLY_FALLBACK

                messages.append(ChatMessage(role="assistant", content=result.content))
                for call in result.tool_calls:
                    tool_result = await self._execute_tool_call(call.name, call.arguments)
                    self._on_tool_used(call.name, tool_result.success, tool_result.message)
                    messages.append(
                        ChatMessage(role="tool", content=tool_result.message, tool_name=call.name)
                    )
            return _EMPTY_REPLY_FALLBACK
        except Exception:
            log.error("pipeline_llm_failed", exc_info=True)
            return _FALLBACK_REPLY

    async def _execute_tool_call(self, name: str, arguments: dict[str, object]) -> ToolResult:
        """Run one tool call, asking the UI to confirm first if its tier requires it.

        SAFE-tier tools run immediately, exactly as before. CONFIRM-tier
        tools pause here and wait for :meth:`respond_confirmation` — if
        denied, the tool never actually runs, and the LLM is told so
        directly instead of receiving the registry's generic "needs your
        confirmation" message (which would read as if nothing had been
        asked yet).

        Args:
            name: The tool's registered name.
            arguments: Raw arguments from the LLM's tool call.

        Returns:
            The tool's result, or a result explaining a denial.
        """
        tool = self._tools.get_tool(name)
        if tool is not None and tool.permission_tier == PermissionTier.CONFIRM:
            allowed = await self._request_confirmation(name, tool.description)
            if not allowed:
                log.info("tool_denied_by_user", tool=name, arguments=arguments)
                return ToolResult(success=False, message=f"The user did not approve running {name}.")
            return await self._tools.execute(name, arguments, confirmed=True)

        confirmed = tool is not None and tool.permission_tier == PermissionTier.SAFE
        return await self._tools.execute(name, arguments, confirmed=confirmed)

    async def _synthesize_safely(self, text: str) -> np.ndarray:
        """Synthesize speech, logging and returning silence on failure.

        Args:
            text: Text to speak.

        Returns:
            Audio samples, or an empty array if synthesis failed.
        """
        if not text:
            return np.zeros(0, dtype=np.float32)
        try:
            return await self._tts.synthesize(text)
        except Exception:
            log.error("pipeline_tts_failed", exc_info=True)
            return np.zeros(0, dtype=np.float32)
