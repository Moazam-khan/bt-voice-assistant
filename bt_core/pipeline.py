"""BT's core conversation pipeline: audio -> VAD -> STT -> LLM -> tools -> TTS.

Pipeline owns the per-utterance state machine: it buffers audio between a
VAD speech-start and speech-end event, transcribes the buffered utterance,
runs it through the LLM (with tool calling, up to a few rounds), and
synthesizes the final reply. Each stage's failure is caught and logged
with a fallback spoken response — one bad turn never crashes the loop.
"""

from __future__ import annotations

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
from bt_core.tools.base import PermissionTier
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
        self._state = _ListenState.WAITING_FOR_WAKE_WORD
        self._recording = False
        self._utterance_buffer: list[np.ndarray] = []

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
        try:
            text = await self._transcriber.transcribe(audio)
        except Exception:
            log.error("pipeline_stt_failed", exc_info=True)
            return await self._synthesize_safely(_FALLBACK_REPLY)

        if not text:
            log.info("pipeline_empty_transcription")
            return np.zeros(0, dtype=np.float32)

        reply_text = await self._run_llm_turn(text)

        now = datetime.now()
        await self._conversation.add_turn(ConversationTurn(role="user", content=text, timestamp=now))
        await self._conversation.add_turn(
            ConversationTurn(role="assistant", content=reply_text, timestamp=now)
        )
        await self._memory.remember(f"User said: {text}\nBT replied: {reply_text}")

        return await self._synthesize_safely(reply_text)

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
                    tool = self._tools.get_tool(call.name)
                    confirmed = tool is not None and tool.permission_tier == PermissionTier.SAFE
                    tool_result = await self._tools.execute(
                        call.name, call.arguments, confirmed=confirmed
                    )
                    messages.append(
                        ChatMessage(role="tool", content=tool_result.message, tool_name=call.name)
                    )
            return _EMPTY_REPLY_FALLBACK
        except Exception:
            log.error("pipeline_llm_failed", exc_info=True)
            return _FALLBACK_REPLY

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
