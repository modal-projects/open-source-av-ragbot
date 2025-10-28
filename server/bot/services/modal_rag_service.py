from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
# from openai import DefaultAioHttpClient

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.metrics.metrics import LLMTokenUsage

from server.bot.processors.parser import ModalRagStreamingJsonParser
import modal


class ModalRagLLMService(OpenAILLMService):
    def __init__(self, *args, **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key"
        func = modal.Function.from_name("vllm-service", "serve")
        llm_url = func.get_web_url() + "/v1"
        if not kwargs.get("base_url"):
            kwargs["base_url"] = llm_url
        super().__init__(*args, **kwargs)
        
        # Create the JSON parser instance
        self.json_parser = ModalRagStreamingJsonParser(self)

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):
        import time
        # Reset the JSON parser for this context
        self.json_parser.reset()

        await self.start_ttfb_metrics()
        waiting_for_first_chunk = True
        # print(f"🚀 Starting time: {time.perf_counter()}")
        
        start_time = time.perf_counter()
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
                print("received empty chunk")
                continue

            if not chunk.choices[0].delta:
                print("received chunk with no delta")
                continue

            if chunk.choices[0].delta.content:
                if waiting_for_first_chunk:
                    await self.stop_ttfb_metrics()
                    waiting_for_first_chunk = False
                    print("received first chunk")
                # print(f"🚀 Stopping time: {time.perf_counter()}")
                # print(f"🚀 Content: {chunk.choices[0].delta.content}")
                # print(f"🚀 Time taken: {time.perf_counter() - start_time:.2f} seconds")
                # start_time = time.perf_counter()
                # Process the content through our streaming JSON parser
                await self.json_parser.process_chunk(chunk.choices[0].delta.content)
        
        print(f"🚀 Total time taken: {time.perf_counter() - start_time:.2f} seconds")


