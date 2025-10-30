

import traceback
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame, 
    Frame, 
    StartFrame, 
    EndFrame, 
    CancelFrame, 
    TTSAudioRawFrame,
)
from pipecat.services.tts_service import WebsocketTTSService

from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State
import json

import modal
import uuid


class KokoroTTSService(WebsocketTTSService):
    def __init__(
        self, 
        # websocket_url: str = "wss://modal-labs-shababo-dev--realtime-stt-transcriber-webapp.modal.run/ws", 
        # websocket_url: str = "wss://modal-labs-shababo-dev--kokoro-tts-kokorotts-webapp.modal.run/ws", 
        websocket_url: str = "wss://30fq5i3jeaicmk.r447.modal.host/ws",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._id = str(uuid.uuid4())
        tts_dict = modal.Dict.from_name("kokoro-tts-dict", create_if_missing=True)
        self._websocket_url = tts_dict.get("websocket_url")
        self._receive_task = None
        # self.opus_decoder = OpusDecoder()
        # self.opus_decoder.set_channels(1)
        # self.opus_decoder.set_sampling_frequency(24000)

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error(error)

    async def _connect(self):
        """Connect to WebSocket and start background tasks."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))
    
    async def _disconnect(self):
        """Disconnect from WebSocket and clean up tasks."""
        try:
            # Cancel background tasks BEFORE closing websocket
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None

            # Now close the websocket
            await self._disconnect_websocket()

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            # Reset state only after everything is cleaned up
            self._websocket = None

    async def _connect_websocket(self):
        """Establish WebSocket connection to API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._websocket = await websocket_connect(
                self._websocket_url,
            )
            logger.debug("Connected to KokoroTTS Websocket")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from KokoroTTS Websocket")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns the active WebSocket connection instance, raising an exception
        if no connection is currently established.

        Returns:
            The active WebSocket connection instance.

        Raises:
            Exception: If no WebSocket connection is currently active.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service.

        Initializes the service by constructing the WebSocket URL with all configured
        parameters and establishing the connection to begin transcription processing.

        Args:
            frame: The start frame containing initialization parameters and metadata.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the  STT service.

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

    async def _receive_messages(self):
        """Receive and process messages from WebSocket.
        """
        async for message in self._get_websocket():
            # if isinstance(message, str):
            try:
                await self.stop_ttfb_metrics()
                # decoded_message = self.opus_decoder.decode(memoryview(bytearray(message)))
                # await self.push_frame(TTSAudioRawFrame(bytes(decoded_message), self.sample_rate, 1))
                await self.push_frame(TTSAudioRawFrame(message, self.sample_rate, 1))
                print(f"Received audio data of length {len(message)} bytes")
            except Exception as e:
                logger.error(f"Error decoding audio: {e}:{traceback.format_exc()}")
                # yield ErrorFrame(f"Error decoding audio: {e}")
                # return
                raise e
                # await self.stop_processing_metrics()
            # else:
            #     logger.warning(f"Received non-string message: {type(message)}")

    async def start_metrics(self):
        """Start TTFB and processing metrics collection."""
        # TTFB (Time To First Byte) metrics are currently disabled for Deepgram Flux.
        # Ideally, TTFB should measure the time from when a user starts speaking
        # until we receive the first transcript. However, Deepgram Flux delivers
        # both the "user started speaking" event and the first transcript simultaneously,
        # making this timing measurement meaningless in this context.
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
    
    async def run_tts(self, prompt: str) -> AsyncGenerator[Frame, None]:

        if not self._websocket:
            logger.error("Not connected to KokoroTTS.")
            yield ErrorFrame("Not connected to KokoroTTS.", fatal=True)
            return

        try:
            await self._websocket.send(prompt)
        except Exception as e:
            logger.error(f"Failed to send audio to KokoroTTS: {e}")
            yield ErrorFrame(f"Failed to send audio to KokoroTTS:  {e}")
            return

        yield None

    
