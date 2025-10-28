import os

from PIL import Image
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def get_frames(substring_match: str):

    sprites = []
    script_dir = os.path.dirname(__file__)

    # Get all PNG files from the assets directory and sort them
    assets_dir = os.path.join(script_dir, "assets")
    for filename in sorted(os.listdir(assets_dir)):
        if substring_match in filename and filename.endswith('.png'):
            full_path = os.path.join(assets_dir, filename)
            with Image.open(full_path) as img:
                sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

    if len(sprites) == 1:
        return sprites[0]
    return SpriteFrame(sprites)


class MoeDalBotAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening), thinking, and talking states based on
    the bot's current status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False
        self._talking_frames = get_frames("talking")
        self._thinking_frames = get_frames("thinking")
        self._listening_frames = get_frames("listening")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(self._talking_frames)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(self._listening_frames)
            self._is_talking = False
        # Switch to thinking animation when user stops speaking
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.push_frame(self._thinking_frames)
            self._is_talking = False
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        