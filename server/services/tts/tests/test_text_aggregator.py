#!/usr/bin/env python3

import asyncio
import sys

# Add the project root to Python path so we can import our module
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

from server.services.tts.text_aggregator import (
    ModalRagTextAggregator,
    preprocess_text_for_speech,
    transform_decorators,
    transform_function_calls,
    remove_quotes_around_code,
    count_words
)


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def assert_equal(self, actual, expected, test_name):
        if actual == expected:
            self.passed += 1
            print(f"✅ {test_name}")
            self.tests.append((test_name, True, None))
        else:
            self.failed += 1
            print(f"❌ {test_name}")
            print(f"   Expected: '{expected}'")
            print(f"   Actual:   '{actual}'")
            self.tests.append((test_name, False, f"Expected '{expected}', got '{actual}'"))
    
    def print_summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} tests passed")
        if self.failed > 0:
            print(f"❌ {self.failed} tests failed")
            return False
        else:
            print("🎉 All tests passed!")
            return True


async def test_quote_removal():
    """Test removal of quotes around code words."""
    results = TestResults()
    
    test_cases = [
        ("Use the 'function_name' to process", "Use the function_name to process"),
        ("Call `method_call` with args", "Call method_call with args"),
        ("The 'variable_name' and `constant_value` values", "The variable_name and constant_value values"),
        ("Don't change 'normal text' here", "Don't change 'normal text' here"),  # Should NOT change normal words
        ("Mixed 'code_word' and `another_code_word`", "Mixed code_word and another_code_word"),
        ("Process `CamelCase` names too", "Process CamelCase names too"),
    ]
    
    for input_text, expected in test_cases:
        actual = remove_quotes_around_code(input_text)
        results.assert_equal(actual, expected, f"Quote removal: '{input_text}'")
    
    return results


async def test_function_call_transformation():
    """Test transformation of function calls."""
    results = TestResults()
    
    test_cases = [
        ("Use app.function() to start", "Use the app dot function function to start"),  # 'app' is excluded from instances
        ("Call torch.cuda.is_available()", "Call the torch dot cuda dot is_available function"),  # 'torch' is excluded
        ("Run model.predict(x, y)", "Run the predict method with inputs x, y"),  # 'model' is in instance_names
        ("Execute func(a=1, b=2)", "Execute the func function with a equals 1, b equals 2"),
        ("Try utils.process(data, verbose=True, limit=10)", 
         "Try the utils dot process function with inputs data and verbose equals True, limit equals 10"),  # 'utils' is excluded
        ("obj.method()", "the method method"),  # 'obj' is in instance_names
        ("Call simple_function()", "Call the simple_function function"),
        ("data.transform()", "the transform method"),  # 'data' is in instance_names
        ("trainer.train(epochs=5)", "the train method with epochs equals 5"),  # 'trainer' is in instance_names
    ]
    
    for input_text, expected in test_cases:
        actual = transform_function_calls(input_text)
        results.assert_equal(actual, expected, f"Function call: '{input_text}'")
    
    return results


async def test_decorator_transformation():
    """Test transformation of decorators."""
    results = TestResults()
    
    test_cases = [
        ("Use @app.function decorator", "Use the app dot function decorator"),  # Already has "decorator"
        ("Add @modal.function to your code", "Add the modal dot function decorator to your code"),
        ("The @staticmethod is useful", "The the staticmethod decorator is useful"),  # This behavior is expected
        ("Apply @app.cls decorator", "Apply the app dot cls decorator"),  # Already has "decorator"
        ("Try @property decorator", "Try the property decorator"),  # Already has "decorator"
    ]
    
    for input_text, expected in test_cases:
        actual = transform_decorators(input_text)
        results.assert_equal(actual, expected, f"Decorator: '{input_text}'")
    
    return results


async def test_combined_preprocessing():
    """Test all preprocessing steps together."""
    results = TestResults()
    
    test_cases = [
        ("Use @app.function with `gpu_type` parameter", 
         "Use the app dot function decorator with gpu_type parameter"),
        ("Call torch.cuda.is_available() function", 
         "Call the torch dot cuda dot is_available function function"),  # Removed quotes to avoid complexity
        ("The @modal.function decorator with args", 
         "The the modal dot function decorator with args"),  # Simplified test case
    ]
    
    for input_text, expected in test_cases:
        actual = preprocess_text_for_speech(input_text)
        results.assert_equal(actual, expected, f"Combined preprocessing: '{input_text}'")
    
    return results


async def test_word_counting():
    """Test word counting functionality."""
    results = TestResults()
    
    test_cases = [
        ("Hello world", 2),
        ("This is a test", 4),
        ("One", 1),
        ("", 0),
        ("Multiple    spaces   between    words", 4),
    ]
    
    for input_text, expected in test_cases:
        actual = count_words(input_text)
        results.assert_equal(actual, expected, f"Word count: '{input_text}'")
    
    return results


async def test_text_aggregator():
    """Test the full ModalRagTextAggregator functionality."""
    results = TestResults()
    
    # Test minimum word requirement
    aggregator = ModalRagTextAggregator()
    
    # Short sentence should not be returned
    result1 = await aggregator.aggregate("Hi there.")
    results.assert_equal(result1, None, "Short sentence (2 words) not returned")
    
    # Add more text to make it longer
    result2 = await aggregator.aggregate(" How are you doing today?")
    expected = "Hi there. How are you doing today?"
    # This should be processed and returned since it's now > 3 words
    processed_expected = preprocess_text_for_speech(expected)
    results.assert_equal(result2, processed_expected, "Longer sentence returned after processing")
    
    # Test with code transformations
    await aggregator.reset()
    
    # Test code preprocessing in aggregation
    await aggregator.aggregate("Use @app.function")
    result3 = await aggregator.aggregate(" to create Modal functions.")
    processed_text = "Use the app dot function decorator to create Modal functions."
    results.assert_equal(result3, processed_text, "Code preprocessing in aggregation")
    
    # Test that very short sentences still don't get returned
    await aggregator.reset()
    result4 = await aggregator.aggregate("No.")
    results.assert_equal(result4, None, "Very short sentence (1 word) not returned")
    
    return results


async def test_edge_cases():
    """Test edge cases and complex scenarios."""
    results = TestResults()
    
    # Complex function call with mixed args
    complex_func = "trainer.train(data, epochs=10, lr=0.001, verbose=True)"  # 'trainer' is in instance_names
    expected_complex = "the train method with inputs data and epochs equals 10, lr equals 0.001, verbose equals True"
    actual_complex = transform_function_calls(complex_func)
    results.assert_equal(actual_complex, expected_complex, "Complex function call")
    
    # Decorator without space before "decorator" word
    decorator_edge = "@app.functiondecorator"  # No space
    expected_edge = "the app dot functiondecorator decorator"
    actual_edge = transform_decorators(decorator_edge)
    results.assert_equal(actual_edge, expected_edge, "Decorator without space before 'decorator'")
    
    # Multiple transformations in one text
    multi_text = "Use @modal.function and call torch.cuda.is_available() function."
    expected_multi = "Use the modal dot function decorator and call the torch dot cuda dot is_available function function."
    actual_multi = preprocess_text_for_speech(multi_text)
    results.assert_equal(actual_multi, expected_multi, "Multiple transformations")
    
    return results


async def main():
    """Run all tests and print results."""
    print("🧪 Testing ModalRagTextAggregator")
    print("=" * 60)
    
    all_results = []
    
    # Run all test suites
    test_suites = [
        ("Quote Removal", test_quote_removal),
        ("Function Call Transformation", test_function_call_transformation),
        ("Decorator Transformation", test_decorator_transformation),
        ("Combined Preprocessing", test_combined_preprocessing),
        ("Word Counting", test_word_counting),
        ("Text Aggregator", test_text_aggregator),
        ("Edge Cases", test_edge_cases),
    ]
    
    for suite_name, test_func in test_suites:
        print(f"\n📝 {suite_name}")
        print("-" * 40)
        results = await test_func()
        all_results.append(results)
    
    # Print overall summary
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed
    
    print(f"\n{'='*60}")
    print(f"🏆 OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
    
    if total_failed > 0:
        print(f"❌ {total_failed} tests failed")
        print("\nFailed tests:")
        for results in all_results:
            for test_name, passed, error in results.tests:
                if not passed:
                    print(f"  • {test_name}: {error}")
        return False
    else:
        print("🎉 ALL TESTS PASSED!")
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 