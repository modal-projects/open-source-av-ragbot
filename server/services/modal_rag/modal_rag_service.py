
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.metrics.metrics import LLMTokenUsage

from server.services.modal_rag.parser import ModalRagStreamingJsonParser

class ModalRagLLMService(OpenAILLMService):
    def __init__(self, *args, **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key"
        super().__init__(*args, **kwargs)
        
        # Create the JSON parser instance
        self.json_parser = ModalRagStreamingJsonParser(self)

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):

        # Reset the JSON parser for this context
        self.json_parser.reset()

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions(
            context
        )

        async for chunk in chunk_stream:
            
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                await self.start_llm_usage_metrics(tokens)

            if chunk.choices is None or len(chunk.choices) == 0:
                continue

            await self.stop_ttfb_metrics()

            if not chunk.choices[0].delta:
                continue

            if chunk.choices[0].delta.content:
                # Process the content through our streaming JSON parser
                await self.json_parser.process_chunk(chunk.choices[0].delta.content)