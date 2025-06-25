import asyncio
import json
from server.services.rag.structured_rag_llm_service import StreamingJSONParser


class MockHandler:
    def __init__(self):
        self.spoke_response_chunks = []
        self.spoke_response_complete_called = False
        self.spoke_response_complete_content = ""
        self.code_blocks_complete_called = False
        self.code_blocks_content = []
        self.links_complete_called = False
        self.links_content = []
        self.parsing_complete_called = False

    async def handle_spoke_response_chunk(self, chunk: str):
        self.spoke_response_chunks.append(chunk)
        print(f"📝 Spoke chunk: '{chunk}'", end='', flush=True)

    async def handle_spoke_response_complete(self, complete_response: str):
        self.spoke_response_complete_called = True
        self.spoke_response_complete_content = complete_response
        print(f"\n✅ Spoke response complete: '{complete_response}'")

    async def handle_code_blocks_complete(self, code_blocks):
        self.code_blocks_complete_called = True
        self.code_blocks_content = code_blocks
        print(f"🔧 Code blocks complete: {len(code_blocks)} blocks")
        for i, block in enumerate(code_blocks):
            print(f"   Block {i+1}: {block[:50]}...")

    async def handle_links_complete(self, links):
        self.links_complete_called = True
        self.links_content = links
        print(f"🔗 Links complete: {len(links)} links")
        for i, link in enumerate(links):
            print(f"   Link {i+1}: {link}")

    async def handle_parsing_complete(self):
        self.parsing_complete_called = True
        print("🎉 Parsing complete!")


async def test_streaming_parser():
    # The JSON response from your debug logs
    test_json_response = """{
    "spoke_response": "To create a Modal function with GPU support, you need to use the modal function decorator and specify the GPU type. Here's a simple example:",
    "code_blocks": [
        "from modal import app, Image\\n\\n@app.function(gpu='A100', image=Image.debian_slim().pip_install('torch'))\\ndef run_gpu_function():\\n    import torch\\n    print(torch.cuda.is_available())"
    ],
    "links": [
        "https://docs.modal.com/guide/functions.html",
        "https://docs.modal.com/guide/images.html",
        "https://docs.modal.com/guide/pricing.html"
    ]
}"""

    # Create parser and handler
    parser = StreamingJSONParser()
    handler = MockHandler()

    print("🚀 Testing StreamingJSONParser...")
    print("=" * 60)
    print(f"📄 Input JSON ({len(test_json_response)} chars):")
    print(test_json_response)
    print("=" * 60)
    print("🌊 Streaming simulation:")

    # Simulate streaming by processing character by character
    for i, char in enumerate(test_json_response):
        await parser.process_chunk(char, handler)
        
        # Add some visual feedback for major transitions
        if char == '{':
            print(f"\n[{i:3d}] 🚪 Object start")
        elif char == ':' and parser.current_field:
            print(f"\n[{i:3d}] 🔑 Field '{parser.current_field}' -> ", end='')
        elif char == '[':
            print(f"\n[{i:3d}] 📋 Array start for '{parser.current_field}'")
        elif char == ']' and parser.bracket_depth == 0:
            print(f"\n[{i:3d}] 📋 Array end for '{parser.current_field}'")

    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print("=" * 60)

    # Verify results
    success = True
    
    # Check spoke_response streaming
    spoke_chunks_text = ''.join(handler.spoke_response_chunks)
    expected_spoke = "To create a Modal function with GPU support, you need to use the modal function decorator and specify the GPU type. Here's a simple example:"
    
    print(f"✅ Spoke response chunks: {len(handler.spoke_response_chunks)} chars streamed")
    print(f"   Expected: '{expected_spoke}'")
    print(f"   Got:      '{spoke_chunks_text}'")
    print(f"   Match: {spoke_chunks_text == expected_spoke}")
    
    if spoke_chunks_text != expected_spoke:
        success = False
        print("❌ Spoke response streaming failed!")
    
    print(f"✅ Spoke response complete called: {handler.spoke_response_complete_called}")
    print(f"   Content: '{handler.spoke_response_complete_content}'")
    
    if not handler.spoke_response_complete_called or handler.spoke_response_complete_content != expected_spoke:
        success = False
        print("❌ Spoke response completion failed!")

    # Check code_blocks
    print(f"✅ Code blocks complete called: {handler.code_blocks_complete_called}")
    print(f"   Number of blocks: {len(handler.code_blocks_content)}")
    
    if handler.code_blocks_complete_called and len(handler.code_blocks_content) == 1:
        code_block = handler.code_blocks_content[0]
        print(f"   Block content: {code_block[:100]}...")
        expected_code = "from modal import app, Image\n\n@app.function(gpu='A100', image=Image.debian_slim().pip_install('torch'))\ndef run_gpu_function():\n    import torch\n    print(torch.cuda.is_available())"
        if code_block == expected_code:
            print("   ✅ Code block matches expected content")
        else:
            success = False
            print("   ❌ Code block doesn't match expected content")
            print(f"   Expected: {expected_code}")
            print(f"   Got:      {code_block}")
    else:
        success = False
        print("❌ Code blocks parsing failed!")

    # Check links
    print(f"✅ Links complete called: {handler.links_complete_called}")
    print(f"   Number of links: {len(handler.links_content)}")
    
    expected_links = [
        "https://docs.modal.com/guide/functions.html",
        "https://docs.modal.com/guide/images.html", 
        "https://docs.modal.com/guide/pricing.html"
    ]
    
    if handler.links_complete_called and handler.links_content == expected_links:
        print("   ✅ Links match expected content")
    else:
        success = False
        print("   ❌ Links don't match expected content")
        print(f"   Expected: {expected_links}")
        print(f"   Got:      {handler.links_content}")

    # Check overall completion
    print(f"✅ Parsing complete called: {handler.parsing_complete_called}")
    
    if not handler.parsing_complete_called:
        success = False
        print("❌ Parsing completion not called!")

    print("=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! StreamingJSONParser is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the output above for details.")
    
    print("=" * 60)
    
    # Test with chunk-based streaming (like real streaming)
    print("\n🔄 Testing chunk-based streaming...")
    parser2 = StreamingJSONParser()
    handler2 = MockHandler()
    
    # Simulate the actual chunks from your debug logs
    chunks = [
        '{\n    ',
        ' "',
        'sp',
        'oke',
        '_response',
        '":',
        ' "',
        'To',
        ' create',
        ' a',
        ' Modal',
        ' function',
        ' with',
        ' GPU',
        ' support',
        ',',
        ' you',
        ' need',
        ' to',
        ' use',
        ' the',
        ' modal',
        ' function',
        ' decorator',
        ' and',
        ' specify',
        ' the',
        ' GPU',
        ' type',
        '.',
        ' Here',
        "'s",
        ' a',
        ' simple',
        ' example',
        ':",\n    ',
        ' "',
        'code',
        '_blocks',
        '":',
        ' [\n        ',
        ' "',
        'from',
        ' modal',
        ' import',
        ' app',
        ',',
        ' Image',
        '\\n',
        '\\n',
        '@app',
        '.function',
        '(g',
        'pu',
        "='",
        'A',
        '1',
        '0',
        '0',
        "',",
        ' image',
        '=',
        'Image',
        '.debian',
        '_s',
        'lim',
        '().',
        'pip',
        '_install',
        "('",
        'torch',
        "'))",
        '\\',
        'ndef',
        ' run',
        '_gpu',
        '_function',
        '():',
        '\\',
        'n',
        '    ',
        ' import',
        ' torch',
        '\\n',
        '    ',
        ' print',
        '(torch',
        '.cuda',
        '.is',
        '_available',
        '())',
        '"\n    ',
        ' ],\n    ',
        ' "',
        'links',
        '":',
        ' [\n        ',
        ' "',
        'https',
        '://',
        'docs',
        '.modal',
        '.com',
        '/g',
        'uide',
        '/functions',
        '.html',
        '",\n        ',
        ' "',
        'https',
        '://',
        'docs',
        '.modal',
        '.com',
        '/g',
        'uide',
        '/images',
        '.html',
        '",\n        ',
        ' "',
        'https',
        '://',
        'docs',
        '.modal',
        '.com',
        '/g',
        'uide',
        '/pr',
        'icing',
        '.html',
        '"\n    ',
        ' ]\n',
        '}'
    ]
    
    print(f"📦 Processing {len(chunks)} chunks (like real streaming)...")
    
    for i, chunk in enumerate(chunks):
        await parser2.process_chunk(chunk, handler2)
        if i < 10:  # Show first few chunks
            print(f"[{i:2d}] '{chunk}' -> State: {parser2.state.value}")
    
    # Verify chunk-based streaming gives same results
    chunk_spoke_text = ''.join(handler2.spoke_response_chunks)
    chunk_success = (
        chunk_spoke_text == expected_spoke and
        handler2.code_blocks_content == handler.code_blocks_content and
        handler2.links_content == handler.links_content and
        handler2.parsing_complete_called
    )
    
    print(f"🔄 Chunk-based streaming: {'✅ PASSED' if chunk_success else '❌ FAILED'}")
    
    return success and chunk_success


if __name__ == "__main__":
    result = asyncio.run(test_streaming_parser())
    exit(0 if result else 1) 