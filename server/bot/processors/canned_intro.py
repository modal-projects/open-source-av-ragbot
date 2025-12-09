from dataclasses import dataclass
from pipecat.frames.frames import (
    Frame,
    OutputAudioRawFrame,
    ErrorFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frame_processor import FrameDirection

@dataclass
class CannedIntroTriggerFrame(Frame):
    audio_path: str = None
    sample_rate: int = 16000
    num_channels: int = 1

class CannedIntroPlayer(FrameProcessor):
    def __init__(self, audio_path: str = None, sample_rate: int = 16000, num_channels: int = 1):
        super().__init__()
        self._audio_path = audio_path
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        if self._audio_path:
            self._audio_bytes = open(self._audio_path, "rb").read()
            if self._audio_bytes.startswith(b"RIFF"):
                self._audio_bytes = self._audio_bytes[44:]
        else:
            self._audio_bytes = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, CannedIntroTriggerFrame):
            audio_bytes = None
            sample_rate = None
            num_channels = None
            if frame.audio_path:
                audio_bytes = open(frame.audio_path, "rb").read()
                if audio_bytes.startswith(b"RIFF"):
                    audio_bytes = audio_bytes[44:]
                sample_rate = frame.sample_rate
                num_channels = frame.num_channels
            else:
                audio_bytes = self._audio_bytes
                sample_rate = self._sample_rate
                num_channels = self._num_channels
            if audio_bytes is None:
                await self.push_error(ErrorFrame(f"No audio bytes found for frame {frame}"))
            else:
                await self.push_frame(OutputAudioRawFrame(audio_bytes, sample_rate=sample_rate, num_channels=num_channels))
        else:
            await self.push_frame(frame, direction)


