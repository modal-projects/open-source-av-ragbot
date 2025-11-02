import enum
import time


from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import LLMTextFrame, Frame
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame


import json
from typing import List


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


class ModalRagStreamingJsonParser():
    """Streaming JSON parser for Modal RAG structured responses."""

    def __init__(self, service: FrameProcessor):
        """Initialize with a reference to the service that owns this parser."""
        self.service = service
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

        self._start_time = time.perf_counter()

    # async def process_frame(self, frame: Frame, direction: FrameDirection):
    #     await super().process_frame(frame, direction)

    #     if isinstance(frame, LLMTextFrame):
    #         await self.process_chunk(frame.text)

    #     # ALWAYS push all frames
    #     else:
    #         # SUPER IMPORTANT: always push every frame!
    #         await self.service.push_frame(frame, direction)

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

    # Handler methods for different types of content
    async def handle_spoke_response_chunk(self, chunk: str):
        """Handle a chunk of the spoke_response as it streams in."""
        # Stream the text immediately for TTS
        await self.service.push_frame(LLMTextFrame(chunk))
        if self._start_time is not None:
            # print(f"Spoke response chunk: {chunk}")
            # print(f"Time taken: {time.perf_counter() - self._start_time:.2f} seconds")
            self._start_time = time.perf_counter()

    async def handle_spoke_response_complete(self, complete_response: str):
        """Handle completion of the spoke_response field."""
        # TODO: Implement any final processing for the complete spoke_response
        # This is called when the entire spoke_response field has been received
        pass

    async def handle_code_blocks_complete(self, code_blocks: List[str]):
        """Handle completion of the code_blocks array."""

        # for code_block in code_blocks:
            #
        # Send code blocks as structured data
        await self.service.push_frame(RTVIServerMessageFrame(
            data={
                "type": "code_blocks",
                "payload": code_blocks
            }
        ))

    async def handle_links_complete(self, links: List[str]):
        """Handle completion of the links array."""
        import httpx

        # for each link, test if we get a 200 response - make this all async!
        good_links = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            for link in links:
                test_link = link
                success = False

                try:
                    response = await client.get(test_link)
                    if response.status_code == 200:
                        good_links.append(test_link)
                        continue
                    else:
                        print(f"Link {test_link} returned status code {response.status_code}")

                    # try changing link to start with https://modal.com/docs if it doesn't work
                    if test_link.startswith("https://modal.com") and not test_link.startswith("https://modal.com/docs/"):
                        test_link = test_link.replace("https://modal.com", "https://modal.com/docs")
                    if test_link.endswith(".html"):
                        test_link = test_link[:-5]

                    response = await client.get(test_link)
                    if response.status_code == 200:
                        good_links.append(test_link)
                        continue
                    else:
                        print(f"Link {test_link} returned status code {response.status_code}")
                except Exception as e:
                    print(f"Error testing link {test_link}: {e}")

        # Send links as structured data
        await self.service.push_frame(RTVIServerMessageFrame(
            data={
                "type": "links",
                "payload": good_links
            }
        ))

    async def handle_parsing_complete(self):
        """Handle completion of the entire JSON parsing."""
        # TODO: Implement any final cleanup or processing logic here
        # This is called when the entire JSON object has been parsed
        pass