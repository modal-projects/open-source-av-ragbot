from typing import AsyncGenerator, Optional
import asyncio
import aiohttp
from loguru import logger
import socket
from struct import pack
from struct import unpack
import uuid
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

import modal
from .kokoro_tts import FORMAT, recvbuffer, sendbuffer

class KokoroTTSService(TTSService):
    """Kokoro Text-to-Speech service using Modal deployment with streaming audio.

    Args:
        base_url: Modal TTS service base URL
        sample_rate: Output sample rate for audio generation
    """


    def __init__(
        self,
        # aiohttp_session: aiohttp.ClientSession,
        base_url: str = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        
        super().__init__(
            # aggregate_sentences=False,
            push_text_frames=True,
            push_stop_frames=False,
            sample_rate=sample_rate,
            # stop_frame_timeout_s=1.0, 
            # push_silence_after_stop=True,
            **kwargs,
        )
        
        from server.services.tts.kokoro_tts import get_kokoro_server_url

        self._base_url = base_url or get_kokoro_server_url()
        # self._session = aiohttp_session
        self.recv_loop = None
        self.send_loop = None
        self._started = False
        self.dict_id = None

        

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

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self.dict_id = uuid.uuid4()
        print(f"Starting Kokoro TTS service with id: {self.dict_id}")
        # if not self._started:
        with modal.Dict.ephemeral() as d:
            await d.put.aio(self.dict_id, True)
            await d.put.aio("server_ready", False)
            print("Kokoro TTS server ready")
            kokoro_cls = modal.Cls.from_name("kokoro-tts", "KokoroTTS")
            kokoro_cls().run.spawn(self.dict_id, d)
            # await asyncio.sleep(10)
            while not await d.get.aio("server_ready"):
                await asyncio.sleep(0.1)
            # await asyncio.sleep(1.0)
            print("Kokoro TTS server ready")
            print("Getting host and port from Kokoro TTS")
            host = await d.get.aio("host")
            port = await d.get.aio("port")
            print(f"Connecting to Kokoro TTS at {host}:{port}...")
            # self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.socket.connect((host, port))
            self.reader, self.writer = await asyncio.open_connection(
                host, port
            )
            print(f"Connected to Kokoro TTS at {host}:{port}")

            self.send_queue = asyncio.Queue()
            
            async def recv_loop():
                while True:
                    async for recv_bytes in recvbuffer(self.reader):
                        # print(f"Received bytes of length {len(recv_bytes)}")
                        if recv_bytes is None:
                            continue
                        
                        if recv_bytes == "<tts_end>".encode("utf-8"):
                            # await d.put.aio(id, False)
                            await self.push_frame(TTSStoppedFrame())
                            continue
                        print(f"Received audio bytes of length {len(recv_bytes)}")
                        await self.stop_ttfb_metrics()
                        await self.push_frame(TTSAudioRawFrame(recv_bytes, self.sample_rate, 1))
                    # await asyncio.sleep(0.001)

            # async def send_loop():
            #     while True:
            #         text = await self.send_queue.get()
            #         if text is None:
            #             await asyncio.sleep(0.001)
            #         else:
            #             await sendbuffer(self.writer, text.encode("utf-8"))
                    
            # self.send_loop = asyncio.create_task(send_loop())
            self.recv_loop = asyncio.create_task(recv_loop())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        # self.socket.close()
        await sendbuffer(self.writer, "<close>".encode("utf-8"))
        self.writer.close()
        await self.writer.wait_closed()
        
        if self.recv_loop:
            self.recv_loop.cancel()
            self.recv_loop = None

        if self.send_loop:
            self.send_loop.cancel()
            self.send_loop = None

        # await kokoro_tts_dict.put.aio(self.dict_id, False)
        self._started = False


    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Kokoro TTS streaming API.

        Makes a request to the Kokoro Modal deployment to generate audio.
        Streams audio chunks in WAV format and tracks utterance duration.
        Handles WAV header removal for raw PCM audio streaming.

        Args:
            text: Text to convert to speech

        Yields:
            Audio and control frames
        """
        logger.info(f"{self}: Received TTS [{text}]")
        yield TTSStartedFrame()
        await self.start_ttfb_metrics()
        try:
            # await self.send_queue.put(text)
            await sendbuffer(self.writer, text.encode("utf-8"))
            # yield None

        except Exception as e:
            print(f"Error in run_tts: {e}")
            # kokoro_tts_dict.put("stop", True)
            # self.writer.close()
            # await self.writer.wait_closed()
            yield ErrorFrame(error=str(e))

        
        # params = {
        #     "prompt": text,
        # }

        # try:
        #     await self.start_ttfb_metrics()

        #     async with self._session.post(
        #         self._base_url, params=params
        #     ) as response:
                
        #         if response.status != 200:
        #             error_text = await response.text()
        #             logger.error(f"{self} error: {error_text}")
        #             yield ErrorFrame(error=f"Kokoro Service error: {error_text}")
        #             return

        #         # Start TTS sequence if not already started
        #         if not self._started:
        #             yield TTSStartedFrame()
        #             await self.stop_ttfb_metrics()
        #             self._started = True

        #         # Track the duration of this utterance based on the last character's end time
        #         async for audio_chunk in response.content.iter_chunked(8192):
        #             if audio_chunk:
        #                 print(f"🚀 Audio chunk: {len(audio_chunk)} bytes")
        #                 if b'RIFF' in audio_chunk:
        #                     audio_chunk = audio_chunk[44:]
        #                 yield TTSAudioRawFrame(audio_chunk, self.sample_rate, 1)

        # except Exception as e:
        #     logger.error(f"Error in run_tts: {e}")
        #     yield ErrorFrame(error=str(e))
        # finally:
        #     self._started = False
        #     yield TTSStoppedFrame()
