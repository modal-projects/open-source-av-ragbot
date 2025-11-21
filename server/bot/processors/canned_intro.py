from dataclasses import dataclass
from pipecat.frames.frames import (
    Frame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frame_processor import FrameDirection
import urllib.request

@dataclass
class CannedIntroTriggerFrame(Frame):
    audio_url: str = None

class CannedIntroPlayer(FrameProcessor):
    def __init__(self, audio_url: str = None, sample_rate: int = 16000, num_channels: int = 1):
        super().__init__()
        self._audio_url = audio_url
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._audio_bytes = urllib.request.urlopen(self._audio_url).read()
        if self._audio_bytes.startswith(b"RIFF"):
            self._audio_bytes = self._audio_bytes[44:]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, CannedIntroTriggerFrame):
            if frame.audio_url:
                audio_bytes = urllib.request.urlopen(frame.audio_url).read()
                if audio_bytes.startswith(b"RIFF"):
                    audio_bytes = audio_bytes[44:]
                else:
                    audio_bytes = self._audio_bytes
            await self.push_frame(OutputAudioRawFrame(audio_bytes, sample_rate=self._sample_rate, num_channels=self._num_channels))
        else:
            await self.push_frame(frame, direction)


