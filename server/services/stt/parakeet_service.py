

from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.time import time_now_iso8601

import modal
import uuid

class ParakeetSTTService(SegmentedSTTService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._id = str(uuid.uuid4())
        Parakeet = modal.Cls.from_name("example-parakeet", "Parakeet")
        self.parakeet = Parakeet()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True
    
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        
        print("🔥 Starting STT...")
        print(f"🔥 Audio length: {len(audio)} bytes")
        
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            response = self.parakeet.transcribe.remote(audio)

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

