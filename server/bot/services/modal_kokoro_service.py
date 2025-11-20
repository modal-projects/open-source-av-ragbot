import traceback
from typing import AsyncGenerator
from loguru import logger
import json
import re
import sys
import time

from pipecat.frames.frames import (
    ErrorFrame, 
    Frame, 
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    InterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

from server.bot.services.modal_services import ModalWebsocketTTSService
from server.bot.processors.unison_speaker_mixer import TTSSpeakerAudioRawFrame

from kokoro import KPipeline, KModel

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass

_MODAL_PHONETIC_TEXT = "[Modal](/məʊdᵊl/)"
_MOE_PHONETIC_TEXT = "[Moe](/məʊ/)"
_DAL_PHONETIC_TEXT = "[Dal](/dæl/)" # alt: dæl


class LocalKokoroTTSService(TTSService):
    def __init__(
        self, 
        speaker: str = None,
        voice: str = None,
        speed: float = 1.0,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._speaker = speaker
        self._voice = voice
        self._speed = speed

        self.model = KModel().to(device).eval()
        self.pipeline = KPipeline(model=self.model, lang_code='a', device="cuda")
            
        print("🔥 Warming up the model...")
        warmup_runs = 6
        warm_up_prompt = "Hello, we are Moe and Dal, your guides to Modal. We can help you get started with Modal, a platform that lets you run your Python code in the cloud without worrying about the infrastructure. We can walk you through setting up an account, installing the package, and running your first job."
        for _ in range(warmup_runs):
            for _ in self._stream_tts(warm_up_prompt):
                pass
        print("✅ Model warmed up!")
        

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._running = False

        await super().push_frame(frame, direction)
    
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

            print(f"Received prompt msg: {prompt}")
            start_time = time.perf_counter()
            for audio_chunk in self._stream_tts(prompt.strip(), voice=self._voice, speed=self._speed):
                await self.push_frame(TTSAudioRawFrame(audio_chunk, self.sample_rate, 1))
            end_time = time.perf_counter()
            print(f"Time taken to stream TTS: {end_time - start_time:.3f} seconds")


        except Exception as e:
            logger.error(f"Failed to send audio to KokoroTTS: {e}")
            yield ErrorFrame(f"Failed to send audio to KokoroTTS:  {e}")

        yield None

class ModalKokoroTTSService(ModalWebsocketTTSService):
    def __init__(
        self, 
        speaker: str = None,
        voice: str = None,
        speed: float = 1.0,
        **kwargs
    ):

        super().__init__(**kwargs)
        self._speaker = speaker
        self._voice = voice
        self._speed = speed
        self._running = False

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

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
                logger.info(f"Received audio data of length {len(message)} bytes")
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
            logger.info(f"Sending prompt: {tts_msg}")
            await self._websocket.send(json.dumps(tts_msg))
        except Exception as e:
            logger.error(f"Failed to send audio to KokoroTTS: {e}")
            yield ErrorFrame(f"Failed to send audio to KokoroTTS:  {e}")

        yield None

