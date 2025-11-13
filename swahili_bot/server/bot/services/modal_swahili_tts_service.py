"""
Pipecat wrapper for Modal Swahili CSM TTS service
"""

import asyncio
import json
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from swahili_bot.server.bot.services.modal_services import (
    ModalWebsocketTTSService,
    ModalTunnelManager,
)


class ModalSwahiliTTSService(ModalWebsocketTTSService):
    """Pipecat wrapper for Swahili CSM TTS service."""

    def __init__(
        self,
        *,
        modal_tunnel_manager: ModalTunnelManager,
        speaker_id: int = 22,
        max_tokens: int = 250,
        **kwargs,
    ):
        super().__init__(
            modal_tunnel_manager=modal_tunnel_manager,
            **kwargs,
        )

        self._speaker_id = speaker_id
        self._max_tokens = max_tokens
        self._sample_rate = 24000  # CSM uses 24kHz
        self._num_channels = 1

        logger.info(f"ModalSwahiliTTSService initialized (speaker_id={speaker_id})")

    def set_speaker_id(self, speaker_id: int):
        """Update the speaker ID."""
        self._speaker_id = speaker_id
        logger.info(f"Speaker ID updated to: {speaker_id}")

    async def run_tts(self, text: str) -> None:
        """Send text to the TTS service for synthesis."""
        try:
            websocket = self._get_websocket()

            logger.info(f"Synthesizing: '{text}' (speaker_id={self._speaker_id})")

            # Send prompt as JSON
            message = json.dumps({
                "type": "prompt",
                "text": text,
                "speaker_id": self._speaker_id,
                "max_tokens": self._max_tokens,
            })
            await websocket.send(message)

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            await self.push_error(ErrorFrame(f"TTS error: {e}"))

    async def _receive_messages(self):
        """Receive audio chunks from the WebSocket."""
        try:
            websocket = self._get_websocket()

            await self.push_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)

            async for message in websocket:
                # Message is binary PCM16 audio data
                audio_bytes = message

                # Emit audio frame
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                    ),
                    FrameDirection.DOWNSTREAM,
                )

            await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

        except asyncio.CancelledError:
            logger.info("Receive task cancelled")
            await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
            await self.push_error(ErrorFrame(f"TTS receive error: {e}"))
            await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
