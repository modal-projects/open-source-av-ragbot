import json
from typing import AsyncGenerator
from loguru import logger
import base64
import sys

from pipecat.frames.frames import (
    ErrorFrame, 
    Frame, 
    TranscriptionFrame, 
    StartFrame, 
    EndFrame, 
    CancelFrame, 
)
from pipecat.utils.time import time_now_iso8601

from .modal_services import ModalWebsocketSegmentedSTTService

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass

class ModalParakeetSegmentedSTTService(ModalWebsocketSegmentedSTTService):
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(**kwargs)

    async def start(self, frame: StartFrame):
        """Start the Parakeet service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        # turn off vad
        vad_msg = {
            "type": "set_vad",
            "vad": False
        }
        await self._websocket.send(json.dumps(vad_msg))

    async def _receive_messages(self):
        """Receive and process messages from WebSocket.
        """
        async for message in self._get_websocket():
            if isinstance(message, str):
                # replace moodle and Moodle with Modal
                message = message.replace("moodle", "Modal").replace("Moodle", "Modal")
                await self.push_frame(TranscriptionFrame(message, "", time_now_iso8601()))
                await self._handle_transcription(message, True)
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()
                logger.info(f"Received transcription: {message}")
            else:
                logger.warning(f"Received non-string message: {type(message)}")
    
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:

        if not self._websocket:
            logger.error("Not connected to Parakeet.")
            yield ErrorFrame("Not connected to Parakeet.", fatal=True)
            return
        await self.start_ttfb_metrics()
        try:
            audio_msg = {
                "type": "audio",
                "audio": base64.b64encode(audio).decode("utf-8")
            }
            await self._websocket.send(json.dumps(audio_msg))
        except Exception as e:
            logger.error(f"Failed to send audio to Parakeet: {e}")
            yield ErrorFrame(f"Failed to send audio to Parakeet:  {e}")
            return

        yield None