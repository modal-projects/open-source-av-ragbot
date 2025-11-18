from loguru import logger
import sys
import uuid
import asyncio
import time
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.frames.frames import StopFrame, CancelFrame, TTSSpeakFrame
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
        **kwargs
    ):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key" 
        
        self.modal_tunnel_manager = modal_tunnel_manager
        self.base_url = base_url
        if self.modal_tunnel_manager:
            logger.info(f"Using Modal Tunnels")
        if self.base_url:
            logger.info(f"Using URL: {self.base_url}")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
        else:
            self._connect_client_task = asyncio.create_task(self._delayed_create_client(**kwargs))

        super().__init__(*args, base_url=base_url, **kwargs)

        self._client = None
    
        # Create the JSON parser instance
        self.json_parser = ModalRagStreamingJsonParser(self)

    async def _get_url(self):
        if self.modal_tunnel_manager:
            print(f"Getting URL from modal tunnel manager")
            url = await self.modal_tunnel_manager.get_url()
            print(f"Got URL from tunnel manager: {url}")
            if not url.endswith("/v1"):
                url = f"{url}/v1"
            self.base_url = url
        return self.base_url

    async def _delayed_create_client(self, **kwargs):
        print(f"Delayed creating client task started...")
        self.base_url = await self._get_url()
        print(f"Got Base URL from _get_url: {self.base_url}")
            
        print(f"Creating client with base URL: {self.base_url}")
        self._client = self.create_client(
            base_url=self.base_url,
            **kwargs
        )
            

    # @classmethod
    # async def from_tunnel_manager(cls, modal_tunnel_manager: ModalTunnelManager, **kwargs):

    #     if not kwargs.get("base_url", None):
    #         base_url = await modal_tunnel_manager.get_url()
            
    #         return cls(
    #             modal_tunnel_manager=modal_tunnel_manager,
    #             base_url=base_url,
    #             **kwargs
    #         )
    #     else:
    #         return cls(
    #             modal_tunnel_manager=modal_tunnel_manager,
    #             **kwargs
    #         )

    async def stop(self, frame: StopFrame):
        await super().stop(frame)
        await self._cleanup()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._cleanup()

    async def _cleanup(self):
        if self.modal_tunnel_manager:
            await self.modal_tunnel_manager.close()

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):

        if not self._connect_client_task.done():
            await self.push_frame(TTSSpeakFrame("My apologies, I'm still setting up a few things. I'll respond as soon as I'm ready."))
            return


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


