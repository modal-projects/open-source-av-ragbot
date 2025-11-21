import traceback
from typing import AsyncGenerator
from loguru import logger
import json
import re
import sys
import time

from pydub import AudioSegment

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

DEFAULT_VOICE = 'am_puck'

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

    def _stream_tts(self, prompt: str, voice = None, speed = 1.3):

        if voice is None:
            voice = DEFAULT_VOICE

        try:
            stream_start = time.perf_counter()
            chunk_count = 0
            first_chunk_time = None

            # Generate streaming audio from the input text
            print(f"🎤 Starting streaming generation for prompt: {prompt}")
            
            for (gs, ps, chunk) in self.pipeline(
                prompt, 
                voice=voice,
                speed = speed,
            ):
                if first_chunk_time is None:
                    print(f"⏱️  Time to first chunk: {(time.perf_counter() - stream_start):.3f} seconds")
                
                print(f"gs: {gs}, ps: {ps}, chunk len: {len(chunk)}")
                chunk_count += 1
                if chunk_count % 10 == 0:  # Log every 10th chunk
                    print(f"📊 Streamed {chunk_count} chunks so far")
                
                try:
                    
                    # Ensure tensor is on CPU and convert to numpy for efficiency
                    audio_numpy = chunk.cpu().numpy()

                    audio_numpy = audio_numpy.clip(-1.0, 1.0) * 32767
                    audio_numpy = audio_numpy.astype('int16')

                    audio_segment = AudioSegment(
                        audio_numpy.tobytes(),
                        frame_rate=24000,
                        sample_width=2,
                        channels=1
                    )

                    def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
                        trim_ms = 0  # ms
                        while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
                            trim_ms += chunk_size

                        return trim_ms - chunk_size # return the index of the last chunk with silence for padding

                    speech_start_idx = detect_leading_silence(audio_segment)
                    audio_segment = audio_segment[speech_start_idx:]
                    yield audio_segment.raw_data
                    
                except Exception as e:
                    print(f"❌ Error converting chunk {chunk_count}: {e}")
                    print(f"   Chunk shape: {chunk.shape if hasattr(chunk, 'shape') else 'N/A'}")
                    print(f"   Chunk type: {type(chunk)}")
                    continue  # Skip this chunk and continue
            
            final_time = time.time()
            print(f"⏱️  Total streaming time: {final_time - stream_start:.3f} seconds")
            print(f"📊 Total chunks streamed: {chunk_count}")
            print("✅ KokoroTTS streaming complete!")

            
        except Exception as e:
            print(f"❌ Error creating stream generator: {e}")
            raise

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

        if not self._connect_websocket_task.done() or not self._websocket:
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

