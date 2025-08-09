

from typing import AsyncGenerator, Optional
from loguru import logger

import websockets
import asyncio

from pipecat.frames.frames import (
    CancelFrame, 
    ErrorFrame, 
    EndFrame, 
    Frame, 
    StartFrame, 
    TranscriptionFrame, 
    InterimTranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    AudioRawFrame,
    InputAudioRawFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.time import time_now_iso8601
from pipecat.audio.utils import create_stream_resampler

import uuid

from server.services.stt.kyutai_stt import get_kyutai_server_url
WS_URL = get_kyutai_server_url().replace("http", "ws") + "/ws"

class KyutaiSTTService(STTService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._id = str(uuid.uuid4())
        self.ws = None
        self.is_final = False
        self.aggregated_transcript = ""
        self.receive_task = None
        self._resampler = create_stream_resampler()
        

    async def start(self, frame: StartFrame) -> None:
        """Start the Deepgram STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        print(f"Starting Kyutai STT service with WS URL: {WS_URL}")
        await super().start(frame)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect (attempt {attempt + 1}/{max_retries})...")
                ws = await websockets.connect(WS_URL, open_timeout=30)
                print("Connected to WebSocket")
                self.ws = ws
                break
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                else:
                    raise e
                
        async def receive_transcription():
            try:
                while True:
                    message = await ws.recv()
                    if message is None:
                        break
                    
                    # Parse the message (server sends b"\x01" + text)
                    if isinstance(message, bytes) and len(message) > 1 and message[0] == 1:
                        transcript = message[1:].decode('utf-8')
                        if len(transcript) > 0:
                            await self.stop_ttfb_metrics()
                            self.aggregated_transcript += transcript
                            
                            # For interim transcriptions, just push the frame without tracing
                            await self.push_frame(
                                InterimTranscriptionFrame(
                                    self.aggregated_transcript,
                                    user_id=self._user_id,
                                    timestamp=time_now_iso8601(),
                                    # language="en",
                                    # result=self.aggregated_transcript,
                                )
                            )
                        
                            print(f"📝 Received: {transcript}")
                    
                    # Handle UserStoppedSpeakingFrame signal (b"\x02")
                    elif isinstance(message, bytes) and len(message) == 1 and message[0] == 2:
                        print("📝 UserStoppedSpeakingFrame signal received - processing final transcription")
                        
                        # Now execute the final transcription block
                        if self.aggregated_transcript:
                            await self.push_frame(
                                TranscriptionFrame(
                                    self.aggregated_transcript,
                                    user_id=self._user_id,
                                    timestamp=time_now_iso8601(),
                                    # language="en",
                                    # result=self.aggregated_transcript,
                                )
                            )
                            await self.stop_processing_metrics()
                            self.aggregated_transcript = ""
                        
                        print("📝 Final transcription processed")
                        
            except websockets.exceptions.ConnectionClosed:
                print("📝 Transcription stream ended")

        self.receive_task = asyncio.create_task(receive_transcription())
                
    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self.receive_task:
            self.receive_task.cancel()
            self.receive_task = None
        if self.ws:
            await self.ws.close()
            self.ws = None

            
    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self.receive_task:
            self.receive_task.cancel()
            self.receive_task = None
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):

        if not frame.audio:
            return
        
        target_sr = self._sample_rate if self._sample_rate != 0 else self.init_sample_rate
        resampled_audio_bytes = await self._resampler.resample(
            frame.audio, frame.sample_rate, target_sr
        )
        # print("--------------------------------")
        # print(f"original audio: {len(frame.audio)}")
        # print(f"resampled_audio_bytes: {len(resampled_audio_bytes)}")
        # print("--------------------------------")

        if len(resampled_audio_bytes) == 0:
            # print(f"RESAMPLED AUDIO IS EMPTY BUT NOT OG AUDIO?, {len(frame.audio)}")
            # print(type(resampled_audio_bytes))
            # print(resampled_audio_bytes)
            # await super().process_audio_frame(frame, direction)
            return
        
        new_frame = InputAudioRawFrame(
            audio=resampled_audio_bytes,
            sample_rate=target_sr,
            num_channels=frame.num_channels,
        )
        await super().process_audio_frame(new_frame, direction)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Deepgram-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            
            self.aggregated_transcript = ""
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

        

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.ws.send(audio)
        yield None




