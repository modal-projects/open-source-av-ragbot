import json
from typing import AsyncGenerator, Optional
from loguru import logger
import base64

from pipecat.frames.frames import (
    ErrorFrame, 
    Frame, 
    TranscriptionFrame, 
    StartFrame, 
    EndFrame, 
    CancelFrame, 
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.time import time_now_iso8601

from .modal_websocket_service import ModalWebsocketService

class ModalSegmentedSTTService(SegmentedSTTService, ModalWebsocketService):
    def __init__(
        self, 
        **kwargs
    ):
        SegmentedSTTService.__init__(self, **kwargs)
        ModalWebsocketService.__init__(self, **kwargs)
        # modal_ws_kwargs = {}
        # if "websocket_url" in kwargs:
        #     modal_ws_kwargs["websocket_url"] = kwargs.pop("websocket_url")
        # if "dict_name" in kwargs:
        #     modal_ws_kwargs["dict_name"] = kwargs.pop("dict_name")
        # if "dict_url_key" in kwargs:
        #     modal_ws_kwargs["dict_url_key"] = kwargs.pop("dict_url_key")
        # if "reconnect_on_error" in kwargs:
        #     modal_ws_kwargs["reconnect_on_error"] = kwargs.pop("reconnect_on_error", True)
        
    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns the active WebSocket connection instance, raising an exception
        if no connection is currently established.

        Returns:
            The active WebSocket connection instance.

        Raises:
            Exception: If no WebSocket connection is currently active.
        """
        print(f"Get WebSocket: {self._websocket_url}")
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the Websocket service.

        Initializes the service by constructing the WebSocket URL with all configured
        parameters and establishing the connection to begin transcription processing.

        Args:
            frame: The start frame containing initialization parameters and metadata.
        """
        print(f"Start: {self._websocket_url}")
        await super().start(frame)
        await self._connect()
        # turn off vad
        vad_msg = {
            "type": "set_vad",
            "vad": False
        }
        await self._websocket.send(json.dumps(vad_msg))

    async def stop(self, frame: EndFrame):
        """Stop the Websocket service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

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
                print(f"Received transcription: {message}")
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