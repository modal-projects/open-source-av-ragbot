#!/usr/bin/env python3
"""
Test script for ChatterboxTTSService text preprocessing functions.
Run this to test the preprocessing logic without launching the full service.
"""

import sys
import os
from unittest.mock import Mock

# Add the server directory to the path so we can import the service
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from services.tts.chatterbox_service import ChatterboxTTSService


def create_test_service():
    """Create a ChatterboxTTSService instance with mocked dependencies for testing."""
    mock_session = Mock()
    service = ChatterboxTTSService(
        aiohttp_session=mock_session,
        base_url="http://test",
        sample_rate=24000
    )
    return service


def test_decorator_conversion():
    """Test Python decorator conversion."""
    print("=== Testing Decorator Conversion ===")
    service = create_test_service()
    
    test_cases = [
        ("@app.route", "the app dot route decorator"),
        ("@modal.web_endpoint", "the modal dot web endpoint decorator"),
        ("@property", "the property decorator"),
        ("@app.route('/api')", "the app dot route decorator('/api')"),
        ("Use @functools.lru_cache here", "Use the functools dot lru cache decorator here"),
    ]
    
    for input_text, expected in test_cases:
        result = service._convert_decorators(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")


def test_function_calls():
    """Test function call conversion."""
    print("\n=== Testing Function Call Conversion ===")
    service = create_test_service()
    
    test_cases = [
        ("print(x, y=5)", "print with the inputs x, y equals 5"),
        ("connect()", "connect with no inputs"),
        ("func(a, b='hello', c=True)", "func with the inputs a, b equals hello, c equals True"),
        ("database.connect(host='localhost', port=5432)", "database.connect with the inputs host equals localhost, port equals 5432"),
        ("nested(func(x), y=2)", "nested with the inputs func(x), y equals 2"),
    ]
    
    for input_text, expected in test_cases:
        result = service._convert_function_calls(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")


def test_dot_notation():
    """Test general dot notation conversion."""
    print("\n=== Testing Dot Notation Conversion ===")
    service = create_test_service()
    
    test_cases = [
        ("app.listen", "app dot listen"),
        ("user.name", "user dot name"),
        ("object.method", "object dot method"),
        ("3.14159", "3.14159"),  # Should preserve decimals
        ("version is 1.2.3", "version is 1.2.3"),  # Should preserve version numbers
        ("call api.endpoint now", "call api dot endpoint now"),
    ]
    
    for input_text, expected in test_cases:
        result = service._convert_dot_notation(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")


def test_full_preprocessing():
    """Test the complete preprocessing pipeline."""
    print("\n=== Testing Full Preprocessing Pipeline ===")
    service = create_test_service()
    
    test_cases = [
        # Single word buffering
        ("Hello!", None),  # Should be buffered
        ("This is a sentence.", "Hello! This is a sentence."),  # Should combine with buffer
        
        # Combined patterns
        ("@app.route('/api')", "the app dot route decorator('/api')"),
        ("Call database.connect(host='localhost')", "Call database dot connect with the inputs host equals localhost"),
        ("Use @property decorator and call func(x=1)", "Use the property decorator and call func with the inputs x equals 1"),
    ]
    
    # Reset service state for clean testing
    service._reset_state()
    
    for input_text, expected in test_cases:
        result = service._preprocess_text(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")


def test_argument_parsing():
    """Test function argument parsing."""
    print("\n=== Testing Function Argument Parsing ===")
    service = create_test_service()
    
    test_cases = [
        ("x, y=5", "x, y equals 5"),
        ("a, b='hello', c=True", "a, b equals hello, c equals True"),
        ("", ""),
        ("just_one_arg", "just_one_arg"),
        ("nested(func()), x=2", "nested(func()), x equals 2"),
    ]
    
    for input_text, expected in test_cases:
        result = service._parse_function_arguments(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")


def test_edge_cases():
    """Test edge cases and complex scenarios."""
    print("\n=== Testing Edge Cases ===")
    service = create_test_service()
    
    test_cases = [
        # Empty and whitespace
        ("", None),
        ("   ", None),
        
        # Mixed patterns
        ("@app.route decorator for api.endpoint()", "the app dot route decorator for api dot endpoint with no inputs"),
        ("Sure!", None),  # Single word should buffer
        ("Okay!", None),  # Another single word should be added to buffer
        ("Let's call func(x=1)", "Sure! Okay! Let's call func with the inputs x equals 1"),  # Should combine all buffered
    ]
    
    # Reset for clean testing
    service._reset_state()
    
    for input_text, expected in test_cases:
        result = service._preprocess_text(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")


def main():
    """Run all tests."""
    print("Testing ChatterboxTTSService Text Preprocessing")
    print("=" * 50)
    
    test_decorator_conversion()
    test_function_calls()
    test_dot_notation()
    test_argument_parsing()
    test_full_preprocessing()
    test_edge_cases()
    
    print("\n" + "=" * 50)
    print("Testing complete! Check the ✅/❌ marks above for results.")


if __name__ == "__main__":
    main() 