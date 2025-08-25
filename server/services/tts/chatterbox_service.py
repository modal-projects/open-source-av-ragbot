from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    EndFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class ChatterboxTTSService(TTSService):
    """Chatterbox Text-to-Speech service using Modal deployment with streaming audio.

    Args:
        aiohttp_session: aiohttp ClientSession for HTTP requests
        base_url: Modal TTS service base URL
        sample_rate: Output sample rate for audio generation
    """


    def __init__(
        self,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=True,
            push_stop_frames=False,
            sample_rate=sample_rate,
            # stop_frame_timeout_s=1.0, 
            # push_silence_after_stop=True,
            **kwargs,
        )
        
        from server.services.tts.chatterbox_tts import get_chatterbox_server_url

        self._base_url = base_url or get_chatterbox_server_url()
        self._session = aiohttp_session
        self._started = False

        

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (StartInterruptionFrame, TTSStoppedFrame)):
            # Reset timing on interruption or stop
            pass
        elif isinstance(frame, LLMFullResponseEndFrame):
            pass

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Chatterbox TTS streaming API.

        Makes a request to the Chatterbox Modal deployment to generate audio.
        Streams audio chunks in WAV format and tracks utterance duration.
        Handles WAV header removal for raw PCM audio streaming.

        Args:
            text: Text to convert to speech

        Yields:
            Audio and control frames
        """
        logger.info(f"{self}: Received TTS [{text}]")
        
        params = {
            "prompt": text,
        }

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                self._base_url, params=params
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"{self} error: {error_text}")
                    yield ErrorFrame(error=f"Chatterbox Service error: {error_text}")
                    return

                # Start TTS sequence if not already started
                if not self._started:
                    yield TTSStartedFrame()
                    await self.stop_ttfb_metrics()
                    self._started = True

                # Track the duration of this utterance based on the last character's end time
                async for audio_chunk in response.content.iter_chunked(6000):
                    if audio_chunk:
                        print(f"🚀 Audio chunk: {len(audio_chunk)} bytes")
                        if b'RIFF' in audio_chunk:
                            audio_chunk = audio_chunk[44:]
                        yield TTSAudioRawFrame(audio_chunk, self.sample_rate, 1)

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            self._started = False
            yield TTSStoppedFrame()
