# Based on pipecat.utils.text.base_text_aggregator.BaseTextAggregator
# https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/utils/text/simple_text_aggregator.py

from typing import Optional
import re

from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


ENDOFSENTENCE_PATTERN_STR = r"""
    (?<![A-Z])       # Negative lookbehind: not preceded by an uppercase letter (e.g., "U.S.A.")
    (?<!\d\.\d)      # Not preceded by a decimal number (e.g., "3.14159")
    (?<!^\d\.)       # Not preceded by a numbered list item (e.g., "1. Let's start")
    (?<!\d\s[ap])    # Negative lookbehind: not preceded by time (e.g., "3:00 a.m.")
    (?<!Mr|Ms|Dr)    # Negative lookbehind: not preceded by Mr, Ms, Dr (combined bc. length is the same)
    (?<!Mrs)         # Negative lookbehind: not preceded by "Mrs"
    (?<!Prof)        # Negative lookbehind: not preceded by "Prof"
    (\.\s*\.\s*\.|[\.\?\!;])|   # Match a period, question mark, exclamation point, or semicolon
    (\。\s*\。\s*\。|[。？！；।])  # the full-width version (mainly used in East Asian languages such as Chinese, Hindi)
    $                # End of string
"""

ENDOFSENTENCE_PATTERN = re.compile(ENDOFSENTENCE_PATTERN_STR, re.VERBOSE)

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

NUMBER_PATTERN = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")

# New patterns for code text processing
DECORATOR_PATTERN = re.compile(r'@([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)')

FUNCTION_CALL_PATTERN = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\('
    r'([^)]*)'
    r'\)'
)

# Only match code-like identifiers (with underscores or CamelCase)
QUOTE_PATTERN = re.compile(r"[`']([a-zA-Z_][a-zA-Z0-9_]*(?:[A-Z][a-z]*)*)[`']")


def replace_match(text: str, match, old_char: str, new_char: str) -> str:
    """Replace a character within a match object in the text."""
    start, end = match.span()
    matched_text = text[start:end]
    replaced_text = matched_text.replace(old_char, new_char)
    return text[:start] + replaced_text + text[end:]


def count_words(text: str) -> int:
    """Count the number of words in the text."""
    return len(text.split())


def remove_quotes_around_code(text: str) -> str:
    """Remove single quotes and backticks around code-like words."""
    return QUOTE_PATTERN.sub(r'\1', text)


def transform_function_calls(text: str) -> str:
    """Transform function calls into natural language."""
    def replace_function_call(match):
        function_path = match.group(1)
        args_str = match.group(2).strip()
        
        # Determine if it's a method or function based on naming convention
        parts = function_path.split('.')
        if len(parts) > 1:
            # Be more conservative about what counts as an instance
            # Only treat very common instance-like names as methods
            first_part = parts[0]
            instance_names = {'obj', 'self', 'item', 'data', 'result', 'response', 'model', 'trainer', 'instance'}
            
            if first_part.lower() in instance_names or (
                first_part.islower() and 
                len(first_part) <= 5 and 
                '_' not in first_part and
                not first_part in {'app', 'torch', 'utils', 'modal', 'api', 'http', 'json', 'math'}
            ):
                # Likely an instance
                call_type = "method"
                readable_path = f"{parts[-1]} method"
            else:
                # Likely a module/class like 'torch.cuda', 'app.function', 'utils.process'
                call_type = "function"
                readable_path = " dot ".join(parts) + " function"
        else:
            call_type = "function"
            readable_path = function_path + " function"
        
        if not args_str:
            return f"the {readable_path}"
        
        # Parse arguments
        args = []
        kwargs = []
        
        # Simple argument parsing (handles basic cases)
        if args_str:
            arg_parts = [arg.strip() for arg in args_str.split(',')]
            for arg in arg_parts:
                if '=' in arg and not arg.startswith('"') and not arg.startswith("'"):
                    # It's a keyword argument
                    key, value = arg.split('=', 1)
                    kwargs.append(f"{key.strip()} equals {value.strip()}")
                else:
                    # It's a positional argument
                    args.append(arg)
        
        result = f"the {readable_path}"
        
        if args or kwargs:
            result += " with"
            if args:
                result += f" inputs {', '.join(args)}"
            if args and kwargs:
                result += " and"
            if kwargs:
                result += f" {', '.join(kwargs)}"
        
        return result
    
    return FUNCTION_CALL_PATTERN.sub(replace_function_call, text)


def transform_decorators(text: str) -> str:
    """Transform decorators into natural language."""
    def replace_decorator(match):
        decorator_path = match.group(1)
        parts = decorator_path.split('.')
        
        if len(parts) > 1:
            readable_path = " dot ".join(parts)
        else:
            readable_path = decorator_path
        
        # Check if "decorator" already follows this pattern
        # Look ahead in the text to see if "decorator" appears right after
        match_end = match.end()
        remaining_text = text[match_end:].lstrip()
        
        if remaining_text.startswith('decorator'):
            # Don't add "decorator" since it's already there
            return f"the {readable_path}"
        else:
            return f"the {readable_path} decorator"
    
    return DECORATOR_PATTERN.sub(replace_decorator, text)


def preprocess_text_for_speech(text: str) -> str:
    """Apply all text preprocessing for better speech synthesis."""
    # Apply transformations in reverse order as requested
    
    # c. Remove quotes around code words
    text = remove_quotes_around_code(text)
    
    # b. Transform function calls
    text = transform_function_calls(text)
    
    # a. Transform decorators
    text = transform_decorators(text)
    
    return text


def match_endofsentence(text: str) -> int:
    """Finds the position of the end of a sentence in the provided text string.

    This function processes the input text by replacing periods in email
    addresses, numbers, and decorator patterns with ampersands to prevent them 
    from being misidentified as sentence terminals. It then searches for the end 
    of a sentence using a specified regex pattern.

    Args:
        text (str): The input text in which to find the end of the sentence.

    Returns:
        int: The position of the end of the sentence if found, otherwise 0.

    """
    text = text.rstrip()

    # Replace email dots by ampersands so we can find the end of sentence. For
    # example, first.last@email.com becomes first&last@email&com.
    emails = list(EMAIL_PATTERN.finditer(text))
    for email_match in emails:
        text = replace_match(text, email_match, ".", "&")

    # Replace number dots by ampersands so we can find the end of sentence.
    numbers = list(NUMBER_PATTERN.finditer(text))
    for number_match in numbers:
        text = replace_match(text, number_match, ".", "&")

    # Replace decorator dots by ampersands to prevent splitting decorator patterns
    # For example, @app.function becomes @app&function
    decorators = list(DECORATOR_PATTERN.finditer(text))
    for decorator_match in decorators:
        text = replace_match(text, decorator_match, ".", "&")

    # Replace function call dots by ampersands to prevent splitting function patterns
    # For example, torch.cuda.is_available() becomes torch&cuda&is_available()
    functions = list(FUNCTION_CALL_PATTERN.finditer(text))
    for function_match in functions:
        text = replace_match(text, function_match, ".", "&")

    # Find all sentence endings and return the position of the last one
    matches = list(ENDOFSENTENCE_PATTERN.finditer(text))
    
    return matches[-1].end() if matches else 0


class ModalRagTextAggregator(BaseTextAggregator):
    """Enhanced text aggregator for Modal RAG responses.
    
    This aggregator processes text for better speech synthesis by:
    - Transforming code references into natural language
    - Ensuring minimum sentence length before completion
    - Detecting proper sentence boundaries
    """

    def __init__(self):
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    async def aggregate(self, text: str) -> Optional[str]:
        result: Optional[str] = None

        self._text += text

        # Find sentence boundaries in original text
        eos_end_marker = match_endofsentence(self._text)
        if eos_end_marker:
            potential_sentence = self._text[:eos_end_marker]
            
            # Apply preprocessing to the potential sentence
            processed_sentence = preprocess_text_for_speech(potential_sentence)
            
            # Check minimum word count on the processed sentence
            if count_words(processed_sentence) > 3:
                result = processed_sentence
                # Remove the processed portion from internal text
                self._text = self._text[eos_end_marker:]
            # If sentence is too short, keep accumulating
        
        return result

    async def handle_interruption(self):
        self._text = ""

    async def reset(self):
        self._text = ""