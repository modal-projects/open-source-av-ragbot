import json
import enum
from abc import ABC, abstractmethod
from typing import Optional, List

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.services.llm_service import FunctionCallFromLLM


class ParseState(enum.Enum):
    WAITING_FOR_OBJECT_START = "waiting_for_object_start"
    WAITING_FOR_KEY = "waiting_for_key"
    IN_KEY = "in_key"
    WAITING_FOR_COLON = "waiting_for_colon"
    WAITING_FOR_VALUE = "waiting_for_value"
    IN_STRING_VALUE = "in_string_value"
    IN_ARRAY_VALUE = "in_array_value"
    WAITING_FOR_COMMA_OR_END = "waiting_for_comma_or_end"
    COMPLETE = "complete"


class StreamingJSONParser(ABC):
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset parser state for a new JSON response."""
        self.state = ParseState.WAITING_FOR_OBJECT_START
        self.current_key = ""
        self.current_value = ""
        self.brace_depth = 0
        self.bracket_depth = 0
        self.in_string = False
        self.escape_next = False
        self.current_field = None  # "spoke_response", "code_blocks", or "links"
        
        # Buffers for different field types
        self.spoke_response_buffer = ""
        self.code_blocks_buffer = ""
        self.links_buffer = ""
        
        # Flags to track completion
        self.spoke_response_streaming = False
        self.spoke_response_complete = False
        self.code_blocks_complete = False
        self.links_complete = False
    
    async def process_chunk(self, content: str) -> None:
        """Process a chunk of streaming content."""
        for char in content:
            await self._process_char(char)
    
    async def _process_char(self, char: str) -> None:
        """Process a single character."""
        
        # Handle string escaping
        if self.escape_next:
            self.current_value += char
            self.escape_next = False
            
            # If we're streaming spoke_response, send the escaped character
            if self.spoke_response_streaming and self.current_field == "spoke_response":
                await self.handle_spoke_response_chunk(char)
            
            return
        
        if char == '\\' and self.in_string:
            self.current_value += char
            self.escape_next = True
            
            # If we're streaming spoke_response, send the backslash
            if self.spoke_response_streaming and self.current_field == "spoke_response":
                await self.handle_spoke_response_chunk(char)
            
            return
        
        # State machine logic
        if self.state == ParseState.WAITING_FOR_OBJECT_START:
            if char == '{':
                self.brace_depth = 1
                self.state = ParseState.WAITING_FOR_KEY
        
        elif self.state == ParseState.WAITING_FOR_KEY:
            if char == '"':
                self.in_string = True
                self.current_key = ""
                self.state = ParseState.IN_KEY
            elif char.isspace():
                pass  # Skip whitespace
        
        elif self.state == ParseState.IN_KEY:
            if char == '"' and not self.escape_next:
                self.in_string = False
                self.current_field = self.current_key
                self.state = ParseState.WAITING_FOR_COLON
            else:
                self.current_key += char
        
        elif self.state == ParseState.WAITING_FOR_COLON:
            if char == ':':
                self.state = ParseState.WAITING_FOR_VALUE
            elif char.isspace():
                pass  # Skip whitespace
        
        elif self.state == ParseState.WAITING_FOR_VALUE:
            if char == '"':
                self.in_string = True
                self.current_value = ""
                self.state = ParseState.IN_STRING_VALUE
                
                # Start streaming for spoke_response
                if self.current_field == "spoke_response":
                    self.spoke_response_streaming = True
                    
            elif char == '[':
                self.bracket_depth = 1
                self.current_value = char
                self.state = ParseState.IN_ARRAY_VALUE
            elif char.isspace():
                pass  # Skip whitespace
        
        elif self.state == ParseState.IN_STRING_VALUE:
            if char == '"' and not self.escape_next:
                self.in_string = False
                await self._handle_string_value_complete()
                self.state = ParseState.WAITING_FOR_COMMA_OR_END
            else:
                self.current_value += char
                
                # Stream spoke_response content as it comes in
                if self.spoke_response_streaming and self.current_field == "spoke_response":
                    await self.handle_spoke_response_chunk(char)
        
        elif self.state == ParseState.IN_ARRAY_VALUE:
            self.current_value += char
            
            if char == '[':
                self.bracket_depth += 1
            elif char == ']':
                self.bracket_depth -= 1
                
                if self.bracket_depth == 0:
                    # Array is complete
                    await self._handle_array_value_complete()
                    self.state = ParseState.WAITING_FOR_COMMA_OR_END
        
        elif self.state == ParseState.WAITING_FOR_COMMA_OR_END:
            if char == ',':
                self.state = ParseState.WAITING_FOR_KEY
            elif char == '}':
                self.brace_depth -= 1
                if self.brace_depth == 0:
                    self.state = ParseState.COMPLETE
                    await self._handle_parsing_complete()
            elif char.isspace():
                pass  # Skip whitespace
    
    async def _handle_string_value_complete(self):
        """Handle completion of a string value."""
        if self.current_field == "spoke_response":
            self.spoke_response_buffer = self.current_value
            self.spoke_response_streaming = False
            self.spoke_response_complete = True
            await self.handle_spoke_response_complete(self.spoke_response_buffer)
    
    async def _handle_array_value_complete(self):
        """Handle completion of an array value."""
        try:
            # Parse the JSON array
            array_data = json.loads(self.current_value)
            
            if self.current_field == "code_blocks":
                self.code_blocks_buffer = array_data
                self.code_blocks_complete = True
                await self.handle_code_blocks_complete(self.code_blocks_buffer)
            elif self.current_field == "links":
                self.links_buffer = array_data
                self.links_complete = True
                await self.handle_links_complete(self.links_buffer)
        except json.JSONDecodeError as e:
            print(f"Error parsing array for field {self.current_field}: {e}")
    
    async def _handle_parsing_complete(self):
        """Handle completion of the entire JSON object."""
        await self.handle_parsing_complete()

    # Abstract handler methods to be implemented by subclasses
    @abstractmethod
    async def handle_spoke_response_chunk(self, chunk: str):
        """Handle a chunk of the spoke_response as it streams in."""
        pass
    
    @abstractmethod
    async def handle_spoke_response_complete(self, complete_response: str):
        """Handle completion of the spoke_response field."""
        pass
    
    @abstractmethod
    async def handle_code_blocks_complete(self, code_blocks: List[str]):
        """Handle completion of the code_blocks array."""
        pass
    
    @abstractmethod
    async def handle_links_complete(self, links: List[str]):
        """Handle completion of the links array."""
        pass
    
    @abstractmethod
    async def handle_parsing_complete(self):
        """Handle completion of the entire JSON parsing."""
        pass


class ModalRagStreamingJSONParser(StreamingJSONParser):
    """Concrete implementation of StreamingJSONParser for Modal RAG responses."""
    
    def __init__(self, service):
        """Initialize with a reference to the service that owns this parser."""
        super().__init__()
        self.service = service
    
    async def handle_spoke_response_chunk(self, chunk: str):
        """Handle a chunk of the spoke_response as it streams in."""
        # Stream the text immediately for TTS
        await self.service.push_frame(LLMTextFrame(chunk))
    
    async def handle_spoke_response_complete(self, complete_response: str):
        """Handle completion of the spoke_response field."""
        # TODO: Implement any final processing for the complete spoke_response
        # This is called when the entire spoke_response field has been received
        pass
    
    async def handle_code_blocks_complete(self, code_blocks: List[str]):
        """Handle completion of the code_blocks array."""
        # Send code blocks as structured data
        await self.service.push_frame(RTVIServerMessageFrame(
            data={
                "type": "code_blocks",
                "payload": code_blocks
            }
        ))
    
    async def handle_links_complete(self, links: List[str]):
        """Handle completion of the links array."""
        # Send links as structured data
        await self.service.push_frame(RTVIServerMessageFrame(
            data={
                "type": "links",
                "payload": links
            }
        ))
    
    async def handle_parsing_complete(self):
        """Handle completion of the entire JSON parsing."""
        # TODO: Implement any final cleanup or processing logic here
        # This is called when the entire JSON object has been parsed
        pass


class StructuredRAGLLMService(OpenAILLMService):
    def __init__(self, *args, **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key"
        super().__init__(*args, **kwargs)
        
        # Create the JSON parser instance
        self.json_parser = ModalRagStreamingJSONParser(self)

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

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
                # Process the content through our streaming JSON parser
                await self.json_parser.process_chunk(chunk.choices[0].delta.content)

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