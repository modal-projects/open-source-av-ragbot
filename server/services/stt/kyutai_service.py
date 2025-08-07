

from typing import AsyncGenerator, Optional
from loguru import logger

import websocket

from pipecat.frames.frames import ErrorFrame, Frame, StartFrame, TranscriptionFrame
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.time import time_now_iso8601

import modal
import uuid

class KyutaiSTTService(STTService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._id = str(uuid.uuid4())
        # KyutaiSTT = modal.Cls.from_name("example-kyutai", "STT")
        # self.kyutai = KyutaiSTT()

    async def start(self, frame: StartFrame) -> None:
        """Start the Deepgram STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self.ws = websocket.connect(self.ws_url)


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
            await self.start_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        
        print("🔥 Starting STT...")
        print(f"🔥 Audio length: {len(audio)} bytes")
        
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            response = self.kyutai.transcribe.remote(audio)

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

            text = response.strip()

            if text:
                await self._handle_transcription(text, True)
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(text, "", time_now_iso8601())
            else:
                logger.warning("Received empty transcription from API")

        except Exception as e:
            logger.exception(f"Exception during transcription: {e}")
            yield ErrorFrame(f"Error during transcription: {str(e)}")

