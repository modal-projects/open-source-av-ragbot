from loguru import logger
import sys
import uuid
import asyncio
import time
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from server.bot.processors.parser import ModalRagStreamingJsonParser
import modal

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass


class ModalVLLMService(OpenAILLMService):
    def __init__(
        self, 
        *args,
        app_name: str = None,
        cls_name: str = None,
        **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key" 
        
        self._app_name = app_name
        self._cls_name = cls_name
        self.call_id = None
        self.modal_client_id = str(uuid.uuid4())
        self.registry_dict = modal.Dict.from_name(f"{self.modal_client_id}-websocket-client-registry", create_if_missing=True)
        self.registry_dict.put(self.modal_client_id, True)
        if self._app_name and self._cls_name:
            logger.info(f"Spawning service for {self._app_name}.{self._cls_name} with client id {self.modal_client_id}")
            modal_service = modal.Cls.from_name(self._app_name, self._cls_name)()
            self.call_id = modal_service.register_client.spawn(self.registry_dict, self.modal_client_id)
            kwargs["base_url"] = self._get_base_url()
        elif not kwargs.get("base_url"):
            raise Exception("Either app_name and cls_name or base_url must be provided")
        else:
            logger.info(f"Using base URL: { kwargs.get("base_url") }")

        
        if not kwargs["base_url"].endswith("/v1"):
            kwargs["base_url"] += "/v1"
        super().__init__(*args, **kwargs)
    
        # Create the JSON parser instance
        self.json_parser = ModalRagStreamingJsonParser(self)

    def _get_base_url(self):
        retries = 360 # 3 minutes
        base_url = None
        while base_url is None and retries > 0:
            retries -= 1
            base_url = self.registry_dict.get("url")
            time.sleep(0.500)
        if base_url is None:
            raise Exception("Failed to get base URL")
        return base_url

    # async def start(self, frame: StartFrame):
    #     await super().start(frame)
    #     retries = 240 # 2 minutes
    #     while self._websocket_url is None and retries > 0:
    #         retries -= 1
    #         self._websocket_url = await self.registry_dict.get.aio("url")
    #         await asyncio.sleep(0.500)
    #     if self._websocket_url is None:
    #         raise Exception("Failed to get websocket URL")

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


