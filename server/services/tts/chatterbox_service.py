#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import re
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
        
        # Buffer for single-word sentences
        self._buffered_text = ""

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    def _reset_state(self):
        """Reset internal state variables."""
        self._cumulative_time = 0
        self._started = False
        self._previous_text = ""
        self._buffered_text = ""
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
            # End of turn - reset previous text and buffered text
            self._previous_text = ""
            self._buffered_text = ""

    def _preprocess_text(self, text: str) -> Optional[str]:
        """Preprocess text before sending to TTS.
        
        Handles:
        1. Single-word sentence buffering - buffers one-word sentences and combines with next multi-word sentence
        2. Python decorator conversion - converts @x.y to "the x dot y decorator"
        3. Function call conversion - converts func(x,y=z) to "func with the inputs x, y equals z"
        4. General dot notation - converts word.word2 to "word dot word2"
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Processed text ready for TTS, or None if text should be buffered
        """
        if not text.strip():
            return None
            
        # First handle all code pattern conversions
        processed_text = self._convert_code_patterns(text)
        
        # Check if this is a single word sentence (ignoring punctuation)
        words = re.findall(r'\b\w+\b', processed_text)
        
        if len(words) == 1:
            # Single word - buffer it
            if self._buffered_text:
                self._buffered_text += " " + processed_text
            else:
                self._buffered_text = processed_text
            logger.debug(f"{self}: Buffering single-word sentence: '{processed_text}'")
            return None
        else:
            # Multi-word sentence - combine with any buffered text
            if self._buffered_text:
                combined_text = self._buffered_text + " " + processed_text
                self._buffered_text = ""  # Clear the buffer
                logger.debug(f"{self}: Combining buffered text with new sentence: '{combined_text}'")
                return combined_text
            else:
                return processed_text
    
    def _convert_code_patterns(self, text: str) -> str:
        """Convert various code patterns to readable speech.
        
        Converts patterns like:
        - @x.y -> "the x dot y decorator"
        - @x -> "the x decorator"
        - func(x,y=z) -> "func with the inputs x, y equals z"
        - word.word2 -> "word dot word2"
        
        Args:
            text: Text that may contain code patterns
            
        Returns:
            Text with code patterns converted to readable format
        """
        # 1. Handle decorators first (most specific pattern)
        text = self._convert_decorators(text)
        
        # 2. Handle function calls
        text = self._convert_function_calls(text)
        
        # 3. Handle general dot notation (least specific, so last)
        text = self._convert_dot_notation(text)
        
        return text
    
    def _convert_decorators(self, text: str) -> str:
        """Convert Python decorators to readable speech.
        
        Converts patterns like:
        - @x.y -> "the x dot y decorator"
        - @x -> "the x decorator"
        
        Args:
            text: Text that may contain decorators
            
        Returns:
            Text with decorators converted to readable format
        """
        # Pattern for @module.function or @module.submodule.function
        dotted_decorator_pattern = r'@([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)'
        
        def replace_dotted_decorator(match):
            decorator_path = match.group(1)
            # Replace dots with "dot" for speech
            readable_path = decorator_path.replace('.', ' dot ')
            return f"the {readable_path} decorator"
        
        # Replace dotted decorators first
        text = re.sub(dotted_decorator_pattern, replace_dotted_decorator, text)
        
        # Pattern for simple @function decorators (that weren't caught by the dotted pattern)
        simple_decorator_pattern = r'@([a-zA-Z_][a-zA-Z0-9_]*)'
        
        def replace_simple_decorator(match):
            decorator_name = match.group(1)
            return f"the {decorator_name} decorator"
        
        # Replace remaining simple decorators
        text = re.sub(simple_decorator_pattern, replace_simple_decorator, text)
        
        return text
    
    def _convert_function_calls(self, text: str) -> str:
        """Convert function calls to readable speech.
        
        Converts patterns like:
        - func(x,y=z) -> "func with the inputs x, y equals z"
        - method(a, b=1, c="hello") -> "method with the inputs a, b equals 1, c equals hello"
        
        Args:
            text: Text that may contain function calls
            
        Returns:
            Text with function calls converted to readable format
        """
        # Pattern for function calls: word followed by parentheses with content
        function_call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)'
        
        def replace_function_call(match):
            function_name = match.group(1)
            args_str = match.group(2).strip()
            
            if not args_str:
                # No arguments
                return f"{function_name} with no inputs"
            
            # Parse arguments
            readable_args = self._parse_function_arguments(args_str)
            
            if readable_args:
                return f"{function_name} with the inputs {readable_args}"
            else:
                return f"{function_name} with inputs"
        
        return re.sub(function_call_pattern, replace_function_call, text)
    
    def _parse_function_arguments(self, args_str: str) -> str:
        """Parse function arguments into readable speech.
        
        Args:
            args_str: String containing function arguments like "x,y=z,a=1"
            
        Returns:
            Readable version like "x, y equals z, a equals 1"
        """
        if not args_str.strip():
            return ""
        
        # Split by commas, but be careful about nested structures
        # For simplicity, we'll do a basic split and handle simple cases
        args = []
        current_arg = ""
        paren_depth = 0
        
        for char in args_str:
            if char == '(' or char == '[' or char == '{':
                paren_depth += 1
                current_arg += char
            elif char == ')' or char == ']' or char == '}':
                paren_depth -= 1
                current_arg += char
            elif char == ',' and paren_depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        # Convert each argument
        readable_args = []
        for arg in args:
            arg = arg.strip()
            if '=' in arg and not any(quote in arg for quote in ['"', "'"]):
                # Simple keyword argument
                parts = arg.split('=', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                # Remove quotes from string values for speech
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                readable_args.append(f"{key} equals {value}")
            else:
                # Positional argument or complex expression
                readable_args.append(arg)
        
        return ", ".join(readable_args)
    
    def _convert_dot_notation(self, text: str) -> str:
        """Convert general dot notation to readable speech.
        
        Converts patterns like:
        - word.word2 -> "word dot word2"
        - object.method -> "object dot method"
        
        Args:
            text: Text that may contain dot notation
            
        Returns:
            Text with dot notation converted to readable format
        """
        # Pattern for word.word (but not decimals like 3.14)
        # Look for word characters followed by dot followed by word characters
        # Exclude cases where preceded or followed by digits
        dot_notation_pattern = r'(?<!\d)\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\d)'
        
        def replace_dot_notation(match):
            first_word = match.group(1)
            second_word = match.group(2)
            return f"{first_word} dot {second_word}"
        
        return re.sub(dot_notation_pattern, replace_dot_notation, text)

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
        logger.debug(f"{self}: Received TTS [{text}]")

        # Preprocess the text (handle single-word buffering and decorators)
        processed_text = self._preprocess_text(text)

        logger.debug(f"{self}: Generating pre-processed TTS [{processed_text}]")
        
        if processed_text is None:
            # Text was buffered, don't generate audio yet
            logger.debug(f"{self}: Text buffered, skipping TTS generation")
            return
        
        # handle Modal pronunciation
        # processed_text = processed_text.replace('modal', 'moadle')
        processed_text = processed_text.replace('Modal', 'modal')
        
        params = {
            "prompt": processed_text,
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

                await self.start_tts_usage_metrics(processed_text)

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
                    self._previous_text += " " + processed_text
                else:
                    self._previous_text = processed_text

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            await self.stop_ttfb_metrics()
            # Let the parent class handle TTSStoppedFrame
