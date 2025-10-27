

from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame, 
    Frame, 
    TranscriptionFrame, 
    StartFrame, 
    EndFrame, 
    CancelFrame, 
    InterimTranscriptionFrame, 
    UserStartedSpeakingFrame, 
    UserStoppedSpeakingFrame, 
    AudioRawFrame,
    InputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService, STTService, WebsocketSTTService
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.time import time_now_iso8601
from pipecat.audio.resamplers.resampy_resampler import ResampyResampler

from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State
import json

import modal
import uuid

class ModalWebsocketSTTService(WebsocketSTTService):
    def __init__(
        self, 
        # websocket_url: str = "wss://modal-labs-shababo-dev--realtime-stt-transcriber-webapp.modal.run/ws", 
        # websocket_url: str = "wss://modal-labs-shababo-dev--streaming-parakeet-transcriber-webapp.modal.run/ws", 
        websocket_url: str = "wss://modal-labs-shababo-dev--streaming-kyutai-stt-streamingky-2e1400.modal.run/ws", 
        reconnect_on_error: bool = True,
        **kwargs
    ):
        # SegmentedSTTService.__init__(self, **kwargs)
        # WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error, **kwargs)
        super().__init__(**kwargs)
        self._id = str(uuid.uuid4())
        # parakeet_dict = modal.Dict.from_name("streaming-kyutai-dict", create_if_missing=True)
        kyutai_server_dict = modal.Dict.from_name("kyutai-server-dict", create_if_missing=True)
        self._websocket_url = kyutai_server_dict.get("websocket_url")
        # self._websocket_url = websocket_url
        self._receive_task = None
        self._resampler = ResampyResampler()
        self.is_speaking = False
        self.aggregated_transcript = ""

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
                # additional_headers={"Authorization": f"Token {self._api_key}"},
            )
            logger.debug("Connected to Parakeet Websocket")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                # await self._send_close_stream()
                logger.debug("Disconnecting from Parakeet Websocket")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    async def _send_close_stream(self) -> None:
        if self._websocket:
            logger.debug("Sending CloseStream message to Parakeet")
            message = {"type": "CloseStream"}
            await self._websocket.send(json.dumps(message))

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

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

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
            if isinstance(message, str):
                # msg_dict = json.loads(message)
                # if msg_dict.get("type") == "final_transcript":
                #     await self.push_frame(TranscriptionFrame(msg_dict["text"], "", time_now_iso8601()))
                #     await self._handle_transcription(message, True)
                #     await self.stop_processing_metrics()
                #     await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
                #     await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
                # elif msg_dict.get("type") == "realtime_transcript":
                #     await self.push_frame(InterimTranscriptionFrame(msg_dict["text"], "", time_now_iso8601()))
                #     await self._handle_transcription(message, False)
                #     await self.stop_processing_metrics()
                if message == "[END_OF_TURN]":
                    await self.push_frame(TranscriptionFrame(self.aggregated_transcript, "", time_now_iso8601()))
                    self.aggregated_transcript = ""
                else:
                    self.aggregated_transcript += message
                    await self.push_frame(InterimTranscriptionFrame(self.aggregated_transcript, "", time_now_iso8601()))
                await self._handle_transcription(message, True)
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()
                print(f"Received transcription: {message}")
            else:
                logger.warning(f"Received non-string message: {type(message)}")

    # async def start_metrics(self):
    #     """Start TTFB and processing metrics collection."""
    #     # TTFB (Time To First Byte) metrics are currently disabled for Deepgram Flux.
    #     # Ideally, TTFB should measure the time from when a user starts speaking
    #     # until we receive the first transcript. However, Deepgram Flux delivers
    #     # both the "user started speaking" event and the first transcript simultaneously,
    #     # making this timing measurement meaningless in this context.
    #     await self.start_ttfb_metrics()
    #     await self.start_processing_metrics()

    # async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):

    #     if not frame.audio:
    #         return
        
    #     # target_sr = self._sample_rate if self._sample_rate != 0 else self.init_sample_rate
    #     resampled_audio_bytes = await self._resampler.resample(
    #         frame.audio, frame.sample_rate, 24000
    #     )

    #     if len(resampled_audio_bytes) > 0:
        
    #         frame = InputAudioRawFrame(
    #             audio=resampled_audio_bytes,
    #             sample_rate=24000,
    #             num_channels=frame.num_channels,
    #         )

    #     await super().process_audio_frame(frame, direction)
    
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:

        resampled_audio_bytes = await self._resampler.resample(
            audio, self.sample_rate, 24000
        )

        if not self._websocket:
            logger.error("Not connected to Parakeet.")
            yield ErrorFrame("Not connected to Parakeet.", fatal=True)
            return
        await self.start_ttfb_metrics()
        try:
            await self._websocket.send(resampled_audio_bytes)
        except Exception as e:
            logger.error(f"Failed to send audio to Parakeet: {e}")
            yield ErrorFrame(f"Failed to send audio to Parakeet:  {e}")
            return

        yield None

    
        
        # print("🔥 Starting STT...")
        # print(f"🔥 Audio length: {len(audio)} bytes")
        # await self.start_ttfb_metrics()
        # try:
        #     await self.start_processing_metrics()
        #     # await self.start_ttfb_metrics()

        #     response = await self.parakeet.transcribe.remote.aio(audio)

        #     text = response.strip()

        #     if text:
        #         await self.stop_ttfb_metrics()
        #         await self.stop_processing_metrics()

                
        #         # await self._handle_transcription(response, True)
        #         logger.debug(f"Transcription: [{response}]")
        #         yield TranscriptionFrame(response, "", time_now_iso8601())
        #     # else:
        #         # logger.warning("Received empty transcription from API")

        # except Exception as e:
        #     logger.exception(f"Exception during transcription: {e}")
        #     yield ErrorFrame(f"Error during transcription: {str(e)}")

    # async def process_frame(self, frame: Frame, direction: FrameDirection):
    #     """Process frames with Deepgram-specific handling.

    #     Args:
    #         frame: The frame to process.
    #         direction: The direction of frame processing.
    #     """
        
        

    #     if isinstance(frame, UserStartedSpeakingFrame):
            
    #         print("🟢 UserStartedSpeakingFrame")
    #         await self.start_ttfb_metrics()
    #         self.is_speaking = True

    #     elif isinstance(frame, UserStoppedSpeakingFrame):
    #         print("🔴 UserStoppedSpeakingFrame")
    #         await self.push_frame(
    #             TranscriptionFrame(
    #                 self.aggregated_transcript,
    #                 user_id=self._user_id,
    #                 timestamp=time_now_iso8601(),
    #                 # language="en",
    #                 # result=self.aggregated_transcript,
    #             ),
    #             direction=direction,
    #         )
    #         self.aggregated_transcript = ""

    #     await super().process_frame(frame, direction)