from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from server.bot.processors.parser import ModalRagStreamingJsonParser
import modal


class ModalVLLMService(OpenAILLMService):
    def __init__(self, *args, **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key"        
        if not kwargs.get("base_url"):
            vllm_dict = modal.Dict.from_name("vllm-dict", create_if_missing=True)
            kwargs["base_url"] = vllm_dict.get("vllm_url")
        if not kwargs["base_url"].endswith("/v1"):
            kwargs["base_url"] += "/v1"
        super().__init__(*args, **kwargs)
        
        # Create the JSON parser instance
        self.json_parser = ModalRagStreamingJsonParser(self)

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


