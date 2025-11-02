

import traceback
from typing import AsyncGenerator, Optional
from loguru import logger
import json
import re

# from pyogg import OpusDecoder
from pipecat.frames.frames import (
    ErrorFrame, 
    Frame, 
    StartFrame, 
    EndFrame, 
    CancelFrame, 
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    InterruptionFrame,
)
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.processors.frame_processor import FrameDirection

from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from server.bot.processors.unison_speaker_mixer import TTSSpeakerAudioRawFrame

import modal

_MODAL_PHONETIC_TEXT = "[Modal](/məʊdᵊl/)"
_MOE_PHONETIC_TEXT = "[Moe](/məʊ/)"
_DAL_PHONETIC_TEXT = "[Dal](/dæl/)" # alt: dæl

class KokoroTTSService(WebsocketTTSService):
    def __init__(
        self, 
        speaker: str = None,
        voice: str = None,
        speed: float = 1.0,
        **kwargs
    ):

        super().__init__(
            pause_frame_processing=True,
            push_stop_frames=True,
            push_text_frames=False,
            stop_frame_timeout_s=1.0,
            **kwargs
        )
        self._speaker = speaker
        self._voice = voice
        self._speed = speed
        
        tts_dict = modal.Dict.from_name("kokoro-tts-dict", create_if_missing=True)
        self._websocket_url = tts_dict.get("websocket_url")
        self._receive_task = None
        self._running = False

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

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._running = False

        await super().push_frame(frame, direction)

    async def _receive_messages(self):
        """Receive and process messages from WebSocket.
        """
        async for message in self._get_websocket():
            try:
                await self.stop_ttfb_metrics()

                if self._speaker:
                    await self.push_frame(TTSSpeakerAudioRawFrame(message, self.sample_rate, 1, speaker=self._speaker))
                else:
                    await self.push_frame(TTSAudioRawFrame(message, self.sample_rate, 1))
                print(f"Received audio data of length {len(message)} bytes")
            except Exception as e:
                logger.error(f"Error decoding audio: {e}:{traceback.format_exc()}")
                await self.push_error(ErrorFrame(f"Error decoding audio: {e}"))
    
    async def run_tts(self, prompt: str) -> AsyncGenerator[Frame, None]:

        if not self._websocket:
            logger.error("Not connected to KokoroTTS.")
            yield ErrorFrame("Not connected to KokoroTTS.", fatal=True)
            return

        try:

            if not self._running:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._running = True

            # The \b symbol in a regex pattern represents a "word boundary".
            # It matches the position between a word character (like a letter or number) and a non-word character (like a space or punctuation), 
            # or the start/end of the string. This ensures that only whole words are matched and replaced, not parts of longer words.
            prompt = re.sub(r'\bModal\b', _MODAL_PHONETIC_TEXT, prompt)
            prompt = re.sub(r'\bmodal\b', _MODAL_PHONETIC_TEXT, prompt)
            prompt = re.sub(r'\bMoe\b', _MOE_PHONETIC_TEXT, prompt)
            prompt = re.sub(r'\bmoe\b', _MOE_PHONETIC_TEXT, prompt)
            prompt = re.sub(r'\bDal\b', _DAL_PHONETIC_TEXT, prompt)
            prompt = re.sub(r'\bdal\b', _DAL_PHONETIC_TEXT, prompt)

            tts_msg = {
                "type": "prompt",
                "text": prompt.strip(),
                "voice": self._voice,
                "speed": self._speed,
            }
            print(f"Sending prompt: {tts_msg}")
            await self._websocket.send(json.dumps(tts_msg))
        except Exception as e:
            logger.error(f"Failed to send audio to KokoroTTS: {e}")
            yield ErrorFrame(f"Failed to send audio to KokoroTTS:  {e}")

        yield None

    
