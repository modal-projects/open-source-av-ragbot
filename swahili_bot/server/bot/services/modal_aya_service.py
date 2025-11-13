"""
Pipecat wrapper for Modal Aya-101 LLM service
"""

import httpx
from loguru import logger
from typing import AsyncGenerator

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai import OpenAILLMService, OpenAILLMContext

from swahili_bot.server.bot.services.modal_services import ModalTunnelManager


class ModalAyaLLMService(OpenAILLMService):
    """Pipecat wrapper for Aya-101 LLM service via vLLM OpenAI API."""

    def __init__(
        self,
        *,
        modal_tunnel_manager: ModalTunnelManager,
        **kwargs,
    ):
        # We'll get the base_url from the tunnel manager
        self.modal_tunnel_manager = modal_tunnel_manager
        self._base_url = None

        # Initialize with a placeholder URL (will be updated)
        super().__init__(
            api_key="dummy",  # vLLM doesn't need a real API key
            base_url="http://placeholder",  # Will be replaced
            model="aya-101",
            **kwargs,
        )

        logger.info("ModalAyaLLMService initialized")

    async def _get_base_url(self):
        """Get the base URL from the tunnel manager."""
        if self._base_url is None:
            self._base_url = await self.modal_tunnel_manager.get_url()
            logger.info(f"Got Aya LLM base URL: {self._base_url}")

            # Update the client with the real URL
            self._client = self._create_client(
                api_key="dummy",
                base_url=self._base_url,
            )

        return self._base_url

    async def _process_context(self, context: OpenAILLMContext):
        """Process the context and generate LLM responses."""
        await self._get_base_url()  # Ensure we have the URL

        try:
            await self.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

            # Get messages from context
            messages = context.get_messages()

            logger.info(f"Sending {len(messages)} messages to Aya LLM")

            # Stream completion
            async for chunk in self._stream_chat_completions(messages):
                await self.push_frame(chunk, FrameDirection.DOWNSTREAM)

            await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        except Exception as e:
            logger.error(f"Error in Aya LLM processing: {e}")
            await self.push_error(ErrorFrame(f"LLM error: {e}"))

    async def _stream_chat_completions(
        self, messages: list
    ) -> AsyncGenerator[Frame, None]:
        """Stream chat completions from the Aya LLM."""
        try:
            # Use the OpenAI client to stream
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=512,  # Reasonable limit for conversation
                temperature=0.7,  # Balanced creativity
                stream=True,
            )

            async for chunk in response:
                if len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta

                if delta.content:
                    # Yield text frame
                    yield LLMTextFrame(text=delta.content)

        except Exception as e:
            logger.error(f"Error streaming from Aya LLM: {e}")
            raise
