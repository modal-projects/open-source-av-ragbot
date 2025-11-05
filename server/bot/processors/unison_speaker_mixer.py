from typing import Dict
from dataclasses import dataclass

import numpy as np
from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

@dataclass
class TTSSpeakerAudioRawFrame(TTSAudioRawFrame):
    speaker: str

class UnisonSpeakerMixer(FrameProcessor):
    """Audio mixer that combines incoming audio from multiple speakers.
    The expectation is that the content in each speaker's TTSSpeakerAudioRawFrames are the same, 
    and come in the same order.
    It uses the soundfile library to load files so it supports multiple formats.
    The audio files need to only have one channel (mono) and it needs to match the sample rate of the output transport.
    
    """

    def __init__(
        self,
        speakers: list[str],
        volume: float = None,
        **kwargs,
    ):
        """Initialize the UnisonSpeakerMixer.

        Args:
            volume: Mixing volume level (0.0 to 1.0). Defaults to 0.4.
            mixing: Whether mixing is initially enabled. Defaults to True.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        
        self._audio_buffers: Dict[str, list[bytes]] = {speaker: [] for speaker in speakers}
        self._volume = volume if volume is not None else 1.0 / len(speakers)
        logger.debug(f"Initialized UnisonSpeakerMixer with speakers: {speakers}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process mixer control frames to update settings or enable/disable mixing.

        Args:
            frame: The mixer control frame to process.
        """

        await super().process_frame(frame, direction)
        
        if isinstance(frame, TTSSpeakerAudioRawFrame):
            self._audio_buffers[frame.speaker].append(frame.audio)
           
            # Determine the *maximum* buffer length (pad with silence if needed)
            if all([len(buffer) > 0 for buffer in self._audio_buffers.values()]):
                
                # Pad all buffers to max_len with silence (zeros)
                audio_to_mix = [buffer.pop(0) for buffer in self._audio_buffers.values()]
                max_len = max(len(audio) for audio in audio_to_mix)
                
                padded_audio_to_mix = []
                for audio in audio_to_mix:
                    if len(audio) < max_len:
                        # Pad with zeros (silence, for PCM16)
                        pad_len = max_len - len(audio)
                        audio_padded = audio + (b"\x00" * pad_len)
                        padded_audio_to_mix.append(audio_padded)
                    elif len(audio) == max_len:
                        padded_audio_to_mix.append(audio)
                    else:
                        logger.info(f"Warning: audio stream {audio} has length {len(audio)} but expected {max_len}")
                
                mixed_audio = self._mix_streams(padded_audio_to_mix)                
                await self.push_frame(TTSAudioRawFrame(mixed_audio, frame.sample_rate, frame.num_channels))

        else:
            await self.push_frame(frame, direction)

    def _mix_streams(self, audio_streams: list[bytes]):
        """Mix raw audio frames with chunks of the same length from each speaker.

        Args:
            audio_streams: List of byte objects to mix. Each must be the same length and represent PCM16 mono audio.
            expected_length: Number of bytes each stream is expected to have.
        """
        
        # Convert buffers to int16 arrays
        audio_streams_np = [np.frombuffer(audio, dtype=np.int16) for audio in audio_streams]

        # Stack and sum (mix)
        stacked = np.stack(audio_streams_np, axis=0)  # shape: (num_streams, num_frames)
        summed = stacked.sum(axis=0)
        # Scale down to prevent clipping
        normalized = (summed * self._volume).clip(-32768, 32767).astype(np.int16)
        mixed_audio = normalized.tobytes()
        return mixed_audio