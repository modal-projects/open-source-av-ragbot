from loguru import logger
import sys
import uuid
import asyncio
import time
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.frames.frames import StopFrame, CancelFrame
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from server.bot.processors.parser import ModalRagStreamingJsonParser
from server.bot.services.modal_services import ModalTunnelManager
import modal

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass


class ModalOpenAILLMService(OpenAILLMService):
    def __init__(
        self, 
        *args,
        modal_tunnel_manager: ModalTunnelManager = None,
        base_url: str = None,
        **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key" 
        
        self.modal_tunnel_manager = modal_tunnel_manager
        self.base_url = base_url
        if self.modal_tunnel_manager:
            logger.info(f"Using Modal Tunnels")
        if self.base_url:
            logger.info(f"Using URL: {self.base_url}")
        else:
            raise Exception("base_url must be provided")

        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        
        super().__init__(*args, base_url=base_url, **kwargs)
    
        # Create the JSON parser instance
        self.json_parser = ModalRagStreamingJsonParser(self)
        

    async def stop(self, frame: StopFrame):
        await super().stop(frame)
        self._cleanup()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._cleanup()

    def _cleanup(self):
        if self.modal_tunnel_manager:
            self.modal_tunnel_manager.close()

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):

        await self.start_ttfb_metrics()

        # Reset the JSON parser for this context
        self.json_parser.reset()

        messages = context.get_messages()
        edited_messages = []
        for msg in messages[:-1]:
          if msg['role'] in ["assistant", "system"]:
            edited_messages.append(msg)
        edited_messages.append(messages[-1])
        new_context = OpenAILLMContext(messages=edited_messages)

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions_specific_context(
            new_context
        )

        async for chunk in chunk_stream:
            
            if chunk.choices is None or len(chunk.choices) == 0:
                continue

            if not chunk.choices[0].delta:
                continue

            if chunk.choices[0].delta.content:
                await self.stop_ttfb_metrics()
                await self.json_parser.process_chunk(chunk.choices[0].delta.content)


