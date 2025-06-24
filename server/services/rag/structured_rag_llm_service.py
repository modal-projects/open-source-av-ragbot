import json

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.services.llm_service import FunctionCallFromLLM

class StructuredRAGLLMService(OpenAILLMService):
    def __init__(self, *args, **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key"
        super().__init__(*args, **kwargs)

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

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

            if chunk.choices[0].delta.tool_calls:
                # We're streaming the LLM response to enable the fastest response times.
                # For text, we just yield each chunk as we receive it and count on consumers
                # to do whatever coalescing they need (eg. to pass full sentences to TTS)
                #
                # If the LLM is a function call, we'll do some coalescing here.
                # If the response contains a function name, we'll yield a frame to tell consumers
                # that they can start preparing to call the function with that name.
                # We accumulate all the arguments for the rest of the streamed response, then when
                # the response is done, we package up all the arguments and the function name and
                # yield a frame containing the function name and the arguments.

                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.index != func_idx:
                    functions_list.append(function_name)
                    arguments_list.append(arguments)
                    tool_id_list.append(tool_call_id)
                    function_name = ""
                    arguments = ""
                    tool_call_id = ""
                    func_idx += 1
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    tool_call_id = tool_call.id
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments
                    arguments += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                if chunk.choices[0].delta.content.startswith("<struct>"):
                    structured_data = json.loads(chunk.choices[0].delta.content.replace("<struct>", "").replace("</struct>", ""))
                    if structured_data.get("code_blocks", None):
                        await self.push_frame(RTVIServerMessageFrame(
                            data={
                                "type": "code_blocks",
                                "payload": structured_data["code_blocks"]
                            }
                        ))
                    if structured_data.get("links", None):
                        await self.push_frame(RTVIServerMessageFrame(
                            data={
                                "type": "links",
                                "payload": structured_data["links"]
                            }
                        ))
                else:
                    await self.push_frame(LLMTextFrame(chunk.choices[0].delta.content))

            # When gpt-4o-audio / gpt-4o-mini-audio is used for llm or stt+llm
            # we need to get LLMTextFrame for the transcript
            elif hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio.get(
                "transcript"
            ):
                await self.push_frame(LLMTextFrame(chunk.choices[0].delta.audio["transcript"]))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            function_calls = []

            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                arguments = json.loads(arguments)
                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            await self.run_function_calls(function_calls)