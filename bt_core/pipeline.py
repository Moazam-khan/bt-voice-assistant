"""BT's core conversation pipeline: audio -> VAD -> STT -> LLM -> tools -> TTS.

Pipeline owns the per-utterance state machine: it buffers audio between a
VAD speech-start and speech-end event, transcribes the buffered utterance,
runs it through the LLM (with tool calling, up to a few rounds), and
synthesizes the final reply. Each stage's failure is caught and logged
with a fallback spoken response — one bad turn never crashes the loop.
"""

from __future__ import annotations

import numpy as np

from bt_core.audio.vad import SpeechEvent, VoiceActivityDetector
from bt_core.llm.client import ChatMessage, OllamaClient
from bt_core.logging_setup import get_logger
from bt_core.stt.transcriber import Transcriber
from bt_core.tools.base import PermissionTier
from bt_core.tools.registry import ToolRegistry
from bt_core.tts.synthesizer import Synthesizer

log = get_logger(__name__)

_MAX_TOOL_ROUNDS = 3
_FALLBACK_REPLY = "Sorry, something went wrong on my end."
_EMPTY_REPLY_FALLBACK = "Done."


class Pipeline:
    """Wires VAD, STT, LLM, tools, and TTS into one conversation loop."""

    def __init__(
        self,
        vad: VoiceActivityDetector,
        transcriber: Transcriber,
        llm_client: OllamaClient,
        tool_registry: ToolRegistry,
        synthesizer: Synthesizer,
        system_prompt: str,
        main_model: str,
    ) -> None:
        """Initialize the pipeline with its already-built dependencies.

        Args:
            vad: Voice activity detector, already constructed from config.
            transcriber: STT engine, already loaded via Transcriber.load().
            llm_client: Ollama chat client.
            tool_registry: BT's available tools.
            synthesizer: TTS engine, already loaded via Synthesizer.load().
            system_prompt: BT's system prompt text.
            main_model: The LLM model name to use for tool-calling turns.
        """
        self._vad = vad
        self._transcriber = transcriber
        self._llm = llm_client
        self._tools = tool_registry
        self._tts = synthesizer
        self._system_prompt = system_prompt
        self._main_model = main_model
        self._recording = False
        self._utterance_buffer: list[np.ndarray] = []

    async def handle_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        """Feed one audio chunk through VAD; return reply audio if a turn completed.

        Args:
            chunk: A raw audio chunk from the microphone.

        Returns:
            Synthesized reply audio if this chunk completed an utterance
            (a full listen -> think -> respond turn), else None.
        """
        events = self._vad.process(chunk)
        mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk

        if SpeechEvent.START in events:
            self._recording = True
            self._utterance_buffer = [mono.astype(np.float32)]
        elif self._recording:
            self._utterance_buffer.append(mono.astype(np.float32))

        if SpeechEvent.END in events and self._recording:
            self._recording = False
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
        return await self._synthesize_safely(reply_text)

    async def _run_llm_turn(self, user_text: str) -> str:
        """Run the LLM + tool-calling loop for one user utterance.

        Args:
            user_text: The transcribed user request.

        Returns:
            BT's final text reply, ready to be spoken.
        """
        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content=user_text),
        ]
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
