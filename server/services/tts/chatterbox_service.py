#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Tuple, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

class ChatterboxTTSService(TTSService):
    """ElevenLabs Text-to-Speech service using HTTP streaming with word timestamps.

    Args:
        aiohttp_session: aiohttp ClientSession
        base_url: API base URL
        sample_rate: Output sample rate
    """


    def __init__(
        self,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=True,
            push_stop_frames=True,
            # pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )


        self._base_url = base_url
        self._session = aiohttp_session

        # Track cumulative time to properly sequence word timestamps across utterances
        self._cumulative_time = 0
        self._started = False

        # Store previous text for context within a turn
        self._previous_text = ""

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    def _reset_state(self):
        """Reset internal state variables."""
        self._cumulative_time = 0
        self._started = False
        self._previous_text = ""
        logger.debug(f"{self}: Reset internal state")

    async def start(self, frame: StartFrame):
        """Initialize the service upon receiving a StartFrame."""
        await super().start(frame)
        self._reset_state()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (StartInterruptionFrame, TTSStoppedFrame)):
            # Reset timing on interruption or stop
            self._reset_state()

        elif isinstance(frame, LLMFullResponseEndFrame):
            # End of turn - reset previous text
            self._previous_text = ""

  

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using ElevenLabs streaming API with timestamps.

        Makes a request to the ElevenLabs API to generate audio and timing data.
        Tracks the duration of each utterance to ensure correct sequencing.
        Includes previous text as context for better prosody continuity.

        Args:
            text: Text to convert to speech

        Yields:
            Audio and control frames
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # handle Modal pronunciation
        text = text.replace('modal', 'moadle')
        text = text.replace('Modal', 'moadle')
        
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

                await self.start_tts_usage_metrics(text)

                # Start TTS sequence if not already started
                if not self._started:
                    yield TTSStartedFrame()
                    self._started = True

                # Track the duration of this utterance based on the last character's end time
                utterance_duration = 0
                # audio_bytes = b""
                async for audio_chunk in response.content.iter_chunked(8192):
                    if audio_chunk:
                        if b'RIFF' in audio_chunk:
                            audio_chunk = audio_chunk[44:]
                        # audio_bytes += audio_chunk  

                        await self.stop_ttfb_metrics()
                        utterance_duration += len(audio_chunk) / (self.sample_rate * 16 / 8)
                        yield TTSAudioRawFrame(audio_chunk, self.sample_rate, 1)

                # After processing all chunks, add the total utterance duration
                # to the cumulative time to ensure next utterance starts after this one
                if utterance_duration > 0:
                    self._cumulative_time += utterance_duration

                # Append the current text to previous_text for context continuity
                # Only add a space if there's already text
                if self._previous_text:
                    self._previous_text += " " + text
                else:
                    self._previous_text = text

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            await self.stop_ttfb_metrics()
            # Let the parent class handle TTSStoppedFrame
