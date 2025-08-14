
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.metrics.metrics import LLMTokenUsage

from server.services.modal_rag.parser import ModalRagStreamingJsonParser


class ModalRagLLMService(OpenAILLMService):
    def __init__(self, *args, **kwargs):
        from server.services.modal_rag.vllm_rag_server import get_rag_server_url
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key"
        if not kwargs.get("base_url"):
            kwargs["base_url"] = get_rag_server_url() + "/v1"
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
            
            if chunk.choices is None or len(chunk.choices) == 0:
                continue

            if not chunk.choices[0].delta:
                continue

            if chunk.choices[0].delta.content:
                await self.stop_ttfb_metrics()

                # Process the content through our streaming JSON parser
                await self.json_parser.process_chunk(chunk.choices[0].delta.content)