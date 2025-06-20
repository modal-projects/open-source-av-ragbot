# Structured output model for TTS integration
from pydantic import BaseModel, Field


from typing import List


class ModalLLMOutput(BaseModel):
    """Structured output for Modal LLM responses, optimized for TTS."""
    answer_for_tts: str = Field(
        description="Clean, conversational answer suitable for text-to-speech, without code blocks or complex formatting"
    )
    code_blocks: List[str] = Field(
        default_factory=list,
        description="List of code snippets mentioned in the response"
    )
    links: List[str] = Field(
        default_factory=list,
        description="List of relevant URLs or documentation links"
    )