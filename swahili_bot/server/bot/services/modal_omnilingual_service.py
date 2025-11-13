"""
Pipecat wrapper for Modal Omnilingual ASR STT service
"""

import asyncio
import base64
import json
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from swahili_bot.server.bot.services.modal_services import (
    ModalWebsocketSegmentedSTTService,
    ModalTunnelManager,
)


class ModalOmnilingualSTTService(ModalWebsocketSegmentedSTTService):
    """Pipecat wrapper for Omnilingual ASR STT service."""

    def __init__(
        self,
        *,
        modal_tunnel_manager: ModalTunnelManager,
        **kwargs,
    ):
        super().__init__(
            modal_tunnel_manager=modal_tunnel_manager,
            **kwargs,
        )

        logger.info("ModalOmnilingualSTTService initialized")

    async def run_stt(self, audio: bytes) -> None:
        """Send audio to the STT service for transcription."""
        try:
            websocket = self._get_websocket()

            # Encode audio as base64
            audio_b64 = base64.b64encode(audio).decode("utf-8")

            # Send as JSON message
            message = json.dumps({"type": "audio", "audio": audio_b64})
            await websocket.send(message)

        except Exception as e:
            logger.error(f"Error in run_stt: {e}")
            await self.push_error(ErrorFrame(f"STT error: {e}"))

    async def _receive_messages(self):
        """Receive transcription messages from the WebSocket."""
        try:
            websocket = self._get_websocket()

            async for message in websocket:
                # Message is plain text (the transcript)
                transcript = message.strip()

                if transcript:
                    logger.info(f"Received transcript: {transcript}")

                    # Emit transcription frame
                    await self.push_frame(
                        TranscriptionFrame(
                            text=transcript,
                            user_id="user",
                            timestamp=self.get_clock().get_time(),
                        ),
                        FrameDirection.DOWNSTREAM,
                    )

        except asyncio.CancelledError:
            logger.info("Receive task cancelled")
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
            await self.push_error(ErrorFrame(f"STT receive error: {e}"))
