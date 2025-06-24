import json
from pathlib import Path
import time
import uuid

import modal

app = modal.App("modal-rag-openai-vllm")

# Volumes for caching models and vector store
modal_docs_volume = modal.Volume.from_name("modal_docs", create_if_missing=True)
models_volume = modal.Volume.from_name("models", create_if_missing=True)
chroma_db_volume = modal.Volume.from_name("model_rag_chroma", create_if_missing=True)

# Model configuration
EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
LLM_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODELS_DIR = Path("/models")

# Download image definition
download_image = (
    modal.Image.debian_slim()
    .pip_install(
        "huggingface_hub[hf_xet]",
        "hf-transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Main image with vLLM dependencies
vllm_rag_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # vLLM and dependencies
        "accelerate==1.5.2",
        "hf-transfer==0.1.8",
        "huggingface_hub[hf_xet]==0.26.2",
        "torch==2.5.1",
        "transformers==4.50.0",
        "vllm==0.7.3",
        # LlamaIndex and RAG dependencies
        "llama-index==0.12.41",
        "llama-index-embeddings-huggingface==0.5.4",
        "fastapi[standard]==0.115.9",
        "llama-index-vector-stores-chroma==0.4.1",
        "chromadb==1.0.11",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": str(MODELS_DIR),
        "VLLM_USE_V1": "0"
    })
)

@app.function(
    volumes={"/models": models_volume},
    image=download_image,
    cpu=4,
    memory=8192,
)
def download_models():
    from huggingface_hub import snapshot_download

    for repo_id in [EMBEDDING_MODEL, LLM_MODEL]:
        snapshot_download(repo_id=repo_id, local_dir=MODELS_DIR / repo_id)
        print(f"Model downloaded to {MODELS_DIR / repo_id}")

@app.function(
    volumes={
        "/models": models_volume,
        "/chroma": chroma_db_volume,
        "/modal_docs": modal_docs_volume,
    },
    cpu=8,
    memory=8192,
    gpu="L40S",
    image=vllm_rag_image,
)
def create_vector_index():
    """Create the ChromaDB vector index from Modal docs if it doesn't exist."""
    import chromadb
    import torch
    from llama_index.core import Document, StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    index_start = time.perf_counter()
    
    torch.set_float32_matmul_precision("high")
    
    # Load embedding model
    embed_start = time.perf_counter()
    embedding = HuggingFaceEmbedding(model_name=f"/models/{EMBEDDING_MODEL}")
    embed_time = time.perf_counter() - embed_start
    print(f"⏱️ [Index] Embedding model loaded in {embed_time:.2f}s")
    
    # Load Modal docs
    doc_start = time.perf_counter()
    with open("/modal_docs/modal_docs.txt") as f:
        document = Document(text=f.read())
    doc_time = time.perf_counter() - doc_start
    print(f"⏱️ [Index] Modal docs loaded in {doc_time:.3f}s, {len(document.text)} chars")
    
    # Setup ChromaDB
    chroma_start = time.perf_counter()
    chroma_client = chromadb.PersistentClient("/chroma")
    
    try:
        chroma_collection = chroma_client.get_collection("modal_rag")
        total_time = time.perf_counter() - index_start
        print(f"Vector index already exists, skipping creation. Total time: {total_time:.3f}s")
        return
    except Exception:
        chroma_collection = chroma_client.create_collection("modal_rag")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index with timing
        vector_start = time.perf_counter()
        index = VectorStoreIndex.from_documents(
            [document], storage_context=storage_context, embed_model=embedding
        )
        vector_time = time.perf_counter() - vector_start
        chroma_time = time.perf_counter() - chroma_start
        total_time = time.perf_counter() - index_start
        
        print(f"⏱️ [Index] Vector index creation took {vector_time:.2f}s")
        print(f"⏱️ [Index] ChromaDB setup took {chroma_time:.2f}s") 
        print(f"🚀 Vector index created successfully in {total_time:.2f}s total!")

@app.cls(
    volumes={
        "/models": models_volume,
        "/chroma": chroma_db_volume,
    },
    cpu=8,
    memory=32768,
    gpu="H100",
    image=vllm_rag_image,
    min_containers=1,
    timeout=10 * 60,
)
@modal.concurrent(max_inputs=3)
class VLLMRAGServer:
    
    @modal.enter()
    def setup(self):
        setup_start = time.perf_counter()
        
        import torch
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore
        
        # vLLM imports
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.utils import random_uuid
        import asyncio
        import numpy as np
        import random
        
        torch.set_float32_matmul_precision("high")
        
        # Seed for reproducibility
        def seed_everything(seed=0):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        seed_everything()
        
        # Initialize embedding model
        embed_start = time.perf_counter()
        print("Loading embedding model...")
        self.embedding = HuggingFaceEmbedding(model_name=f"/models/{EMBEDDING_MODEL}")
        embed_time = time.perf_counter() - embed_start
        print(f"⏱️ Embedding model loaded in {embed_time:.2f}s")
        
        # System prompt
        self.prompt = (
            "You are a conversational AI that is an expert in the Modal library. "
            "Your form is the Modal logo, a pair of characters named Moe and Dal. "
            "Always refer to yourself in the plural as 'we' and 'us' and never 'I' or 'me'. "
            "Because you are a conversational AI, you should not provide code or use symbols in your response. "
            "Additionally, answer the user's questions concisely and in English."
        )

        # Initialize vLLM AsyncLLMEngine
        vllm_start = time.perf_counter()
        print("Loading vLLM AsyncLLMEngine...")
        self.model_path = f"/models/{LLM_MODEL}"
        
        # Load tokenizer
        tokenizer_start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer_time = time.perf_counter() - tokenizer_start
        print(f"⏱️ Tokenizer loaded in {tokenizer_time:.2f}s")
        
        # Setup vLLM engine
        engine_kwargs = {
            "max_num_seqs": 3,  # Match concurrent max_inputs
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": 32768,  # Increased to avoid conflicts
            "max_model_len": 16384,  # Reasonable context length for RAG (must be <= max_num_batched_tokens)
        }
        
        engine_start = time.perf_counter()
        print(f"Setting up vLLM engine with kwargs: {engine_kwargs}")
        self.vllm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=str(self.model_path),
                tensor_parallel_size=torch.cuda.device_count(),
                **engine_kwargs,
            )
        )
        engine_time = time.perf_counter() - engine_start
        vllm_time = time.perf_counter() - vllm_start
        print(f"⏱️ vLLM engine setup in {engine_time:.2f}s, total vLLM loading: {vllm_time:.2f}s")
        
        # Setup vector store
        vector_start = time.perf_counter()
        print("Loading vector store...")
        chroma_client = chromadb.PersistentClient("/chroma")
        chroma_collection = chroma_client.get_collection("modal_rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create RAG retriever
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self.embedding
        )
        self.retriever = index.as_retriever(similarity_top_k=5)
        vector_time = time.perf_counter() - vector_start
        print(f"⏱️ Vector store setup in {vector_time:.2f}s")
        
        # Setup FastAPI app
        fastapi_start = time.perf_counter()
        self.setup_fastapi()
        fastapi_time = time.perf_counter() - fastapi_start
        print(f"⏱️ FastAPI setup in {fastapi_time:.2f}s")

        # Warm up vLLM engine
        warmup_start = time.perf_counter()
        print("Warming up vLLM engine...")
        asyncio.run(self.warm_up_vllm())
        warmup_time = time.perf_counter() - warmup_start
        print(f"⏱️ vLLM warmup in {warmup_time:.2f}s")
        
        setup_time = time.perf_counter() - setup_start
        print(f"🚀 VLLMRAGServer setup complete in {setup_time:.2f}s total!")
    
    async def warm_up_vllm(self):
        """Warm up the vLLM engine with a simple completion."""
        try:
            await self.generate_vllm_completion("What is Modal?")
            print("✅ vLLM warmup completed!")
        except Exception as e:
            print(f"⚠️ vLLM warmup failed: {e}")

    async def generate_vllm_completion(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1, stream: bool = False):
        """Generate completion using vLLM AsyncLLMEngine."""
        from dataclasses import asdict
        from vllm import SamplingParams
        from vllm.utils import random_uuid
        
        gen_start = time.perf_counter()
        
        # Apply chat template if available
        if get_system_prompt() and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        request_id = random_uuid()
        
        if stream:
            # For streaming, return the generator
            print(f"⏱️ [vLLM] Starting streaming generation for {len(formatted_prompt)} chars")
            return self.vllm_engine.generate(
                formatted_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        else:
            # For non-streaming, collect all results
            print(f"⏱️ [vLLM] Starting non-streaming generation for {len(formatted_prompt)} chars")
            results_generator = self.vllm_engine.generate(
                formatted_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            
            async for generation in results_generator:
                pass  # Iterate through all generations
            
            # Return the final generation result
            outputs = [asdict(output) for output in generation.outputs]
            result = outputs[0]['text'] if outputs else ""
            gen_time = time.perf_counter() - gen_start
            print(f"⏱️ [vLLM] Non-streaming generation completed in {gen_time:.3f}s, output: {len(result)} chars")
            return result
    
    async def generate_structured_response(self, question: str, conversation_history: str = ""):
        """Generate a structured response using vLLM with RAG context."""

        from server.services.rag.models import ModalLLMOutput

        total_start = time.perf_counter()
        
        try:
            # Get relevant context from RAG
            rag_start = time.perf_counter()
            retrieved_nodes = self.retriever.retrieve(question)
            context_str = "\n\n".join([node.text for node in retrieved_nodes])
            rag_time = time.perf_counter() - rag_start
            print(f"⏱️ RAG retrieval took {rag_time:.3f}s for query: {question[:50]}...")
            
            # Create structured output prompt with conversation history
            history_context = ""
            if conversation_history:
                history_context = f"\n\nConversation History:\n{conversation_history}\n"
            
            prompt_template = f"""You are a conversational AI that is an expert in the Modal library.
Your form is the Modal logo, a pair of characters named Moe and Dal. Refer to yourself in the plural as 'we' and 'us' when appropriate.
Because you are a conversational AI, you should not provide code or use symbols in your response.
Additionally, answer the user's questions concisely and in English. Based on the provided context, conversation history, and current question, 
generate a structured response based on the schema below.

Modal Documentation Context: {context_str}{history_context}
Current Question: {question}

You MUST respond with ONLY the following JSON format (no additional text):

{{
    "answer_for_tts": "A clean, conversational answer suitable for text-to-speech. Use natural language without technical symbols, code syntax, or complex formatting. Explain concepts simply and avoid bullet points.",
    "code_blocks": ["List of actual code snippets that would be useful to display separately"],
    "links": ["List of relevant URLs or documentation references"]
}}

IMPORTANT: Your response must be valid JSON starting with {{ and ending with }}. Do not include any explanatory text."""

            # Use vLLM to generate structured response
            vllm_start = time.perf_counter()
            raw_response = await self.generate_vllm_completion(
                prompt_template,
                max_tokens=1024,
                temperature=0.1
            )
            vllm_time = time.perf_counter() - vllm_start
            print(f"⏱️ vLLM generation took {vllm_time:.3f}s")
            
            # Parse JSON response
            parse_start = time.perf_counter()
            import re
            import json
            
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Try to fix common JSON issues
                if not json_str.endswith('}'):
                    open_braces = json_str.count('{')
                    close_braces = json_str.count('}')
                    missing_braces = open_braces - close_braces
                    json_str += '}' * missing_braces
                
                try:
                    parsed_json = json.loads(json_str)
                    result = ModalLLMOutput(
                        answer_for_tts=parsed_json.get("answer_for_tts", ""),
                        code_blocks=parsed_json.get("code_blocks", []),
                        links=parsed_json.get("links", [])
                    )
                    parse_time = time.perf_counter() - parse_start
                    total_time = time.perf_counter() - total_start
                    print(f"⏱️ JSON parsing took {parse_time:.3f}s, total response time: {total_time:.3f}s")
                    return result
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
            
            # Fallback: treat raw response as TTS content
            parse_time = time.perf_counter() - parse_start
            total_time = time.perf_counter() - total_start
            print(f"⏱️ JSON fallback parsing took {parse_time:.3f}s, total response time: {total_time:.3f}s")
            return ModalLLMOutput(
                answer_for_tts=raw_response,
                code_blocks=[],
                links=[]
            )
            
        except Exception as e:
            total_time = time.perf_counter() - total_start
            print(f"Structured response failed: {e}, total time: {total_time:.3f}s")
            # Fallback to basic completion
            basic_prompt = f"{self.prompt} {question}"
            fallback_response = await self.generate_vllm_completion(basic_prompt, max_tokens=512)
            return ModalLLMOutput(
                answer_for_tts=fallback_response,
                code_blocks=[],
                links=[]
            )
    
    async def generate_streaming_structured_response(self, question: str, conversation_history: str = ""):
        """Generate structured response with vLLM streaming for immediate TTS."""
        import re
        import json
        
        total_start = time.perf_counter()
        
        try:
            # Get RAG context
            rag_start = time.perf_counter()
            retrieved_nodes = self.retriever.retrieve(question)
            context_str = "\n\n".join([node.text for node in retrieved_nodes])
            rag_time = time.perf_counter() - rag_start
            print(f"⏱️ [Streaming] RAG retrieval took {rag_time:.3f}s")
            
            # Create structured prompt with conversation history
            history_context = ""
            if conversation_history:
                history_context = f"\n\nConversation History:\n{conversation_history}\n"
            
            prompt_template = f"""You are a conversational AI that is an expert in the Modal library.
Your form is the Modal logo, a pair of characters named Moe and Dal. Refer to yourself in the plural as 'we' and 'us' when appropriate.
Because you are a conversational AI, you will provide your response with three parts described below.
It is very important that you don't use terms like @modal.function in the answer_for_tts part of the response.
Instead you can say "the modal function decorator" or something like that.
The proper code will be included in the code_blocks part of the response.
Additionally, answer the user's questions concisely and in English. Do not use single word sentences like "Hello!" or "Okay". These do not work well with text-to-speech. 

Based on the provided context, conversation history, and current question, 
generate a structured response based on the schema below.

Modal Documentation Context: {context_str}{history_context}
Current Question: {question}

You MUST respond with ONLY the following JSON format (no additional text):

{{
    "answer_for_tts": "A clean, conversational answer suitable for text-to-speech. Use natural language without technical symbols, code syntax, or complex formatting. Explain concepts simply and avoid bullet points.",
    "code_blocks": ["List of actual code snippets that would be useful to display separately"],
    "links": ["List of relevant URLs or documentation references"]
}}

IMPORTANT: Your response must be valid JSON starting with {{ and ending with }}. Do not include any explanatory text."""

            # Stream the vLLM response
            vllm_start = time.perf_counter()
            streaming_generator = await self.generate_vllm_completion(
                prompt_template,
                max_tokens=1024,
                temperature=0.1,
                stream=True
            )
            print(f"⏱️ [Streaming] vLLM generator setup took {time.perf_counter() - vllm_start:.3f}s")
            
            # State for incremental JSON parsing
            accumulated_text = ""
            tts_content_streamed = False
            first_token_time = None
            token_count = 0
            
            async for generation in streaming_generator:
                # Extract text from vLLM generation
                if generation.outputs:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        print(f"⏱️ [Streaming] First token took {first_token_time - vllm_start:.3f}s")
                    
                    accumulated_text = generation.outputs[0].text
                    token_count += 1
                    
                    # Try to parse JSON incrementally
                    if not tts_content_streamed and '"answer_for_tts"' in accumulated_text:
                        # Extract TTS content as it becomes available
                        tts_match = re.search(r'"answer_for_tts":\s*"([^"]*(?:\\"[^"]*)*)"', accumulated_text, re.DOTALL)
                        if tts_match:
                            tts_content = tts_match.group(1).replace('\\"', '"')
                            if tts_content and len(tts_content) > 10:
                                tts_content_streamed = True
                                yield {"type": "tts_content", "content": tts_content}
                    
                    # Try to parse complete JSON for structured data
                    json_match = re.search(r'\{.*\}', accumulated_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            parsed_json = json.loads(json_str)
                            code_blocks = parsed_json.get("code_blocks", [])
                            links = parsed_json.get("links", [])
                            
                            total_time = time.perf_counter() - total_start
                            print(f"⏱️ [Streaming] Complete response took {total_time:.3f}s, {token_count} tokens")
                            yield {
                                "type": "structured_data",
                                "code_blocks": code_blocks,
                                "links": links,
                                "complete": True
                            }
                            return
                        except json.JSONDecodeError as e:
                            print(f"JSON parsing failed: {e}")
                            continue
            
            # Handle final result if not already processed
            if accumulated_text and not tts_content_streamed:
                # Try final parsing
                json_match = re.search(r'\{.*\}', accumulated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        parsed_json = json.loads(json_str)
                        tts_content = parsed_json.get("answer_for_tts", "")
                        code_blocks = parsed_json.get("code_blocks", [])
                        links = parsed_json.get("links", [])
                        
                        if tts_content:
                            yield {"type": "tts_content", "content": tts_content}
                        
                        total_time = time.perf_counter() - total_start
                        print(f"⏱️ [Streaming] Final JSON parsing took {total_time:.3f}s")
                        yield {
                            "type": "structured_data",
                            "code_blocks": code_blocks,
                            "links": links,
                            "complete": True
                        }
                        return
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: use accumulated text as TTS content
                if accumulated_text.strip():
                    total_time = time.perf_counter() - total_start
                    print(f"⏱️ [Streaming] Text fallback took {total_time:.3f}s")
                    yield {"type": "tts_content", "content": accumulated_text.strip()}
                    yield {"type": "structured_data", "code_blocks": [], "links": [], "complete": True}
                    return
            
            # Ultimate fallback
            total_time = time.perf_counter() - total_start
            print(f"⏱️ [Streaming] Ultimate fallback after {total_time:.3f}s")
            basic_prompt = f"{self.prompt} {question}"
            fallback_response = await self.generate_vllm_completion(basic_prompt, max_tokens=512)
            yield {"type": "tts_content", "content": fallback_response}
            yield {"type": "structured_data", "code_blocks": [], "links": [], "complete": True}
            
        except Exception as e:
            total_time = time.perf_counter() - total_start
            print(f"vLLM streaming error: {e}, total time: {total_time:.3f}s")
            # Error fallback
            try:
                basic_prompt = f"{self.prompt} {question}"
                fallback_response = await self.generate_vllm_completion(basic_prompt, max_tokens=512)
                yield {"type": "tts_content", "content": fallback_response}
            except:
                yield {"type": "tts_content", "content": "Sorry, I encountered an error processing your request."}
            yield {"type": "structured_data", "code_blocks": [], "links": [], "complete": True}
    
    def setup_fastapi(self):
        from typing import Any, Dict, List, Optional
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        self.web_app = FastAPI(title="vLLM RAG API", version="1.0.0")
        
        class Message(BaseModel):
            role: str
            content: str
        
        class ChatCompletionRequest(BaseModel):
            model: str
            messages: List[Message]
            max_tokens: Optional[int] = 500
            temperature: Optional[float] = 0.1
            stream: Optional[bool] = False
            
            class Config:
                extra = "allow"
        
        @self.web_app.get("/health")
        def health_check():
            return {"status": "healthy"}
        
        @self.web_app.get("/v1/models")
        def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": LLM_MODEL,
                        "object": "model",
                        "created": 1234567890,
                        "owned_by": "modal",
                    },
                    {
                        "id": "modal-rag",
                        "object": "model",
                        "created": 1234567890,
                        "owned_by": "modal",
                    },
                ],
            }
        
        @self.web_app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible endpoint with vLLM backend."""
            import asyncio
            from fastapi.responses import StreamingResponse
            
            try:
                # Extract conversation history and current user message
                conversation_history = []
                current_user_message = None
                
                for msg in request.messages:
                    if msg.role in ["system", "user", "assistant"]:
                        conversation_history.append({"role": msg.role, "content": msg.content})
                        if msg.role == "user":
                            current_user_message = msg.content  # Keep track of the most recent user message

                if not current_user_message:
                    raise HTTPException(status_code=400, detail="No user message found")
                
                # Format conversation history for context
                formatted_history = ""
                if len(conversation_history) > 1:  # If there's more than just the current message
                    formatted_history = "\n".join([
                        f"{msg['role'].capitalize()}: {msg['content']}" 
                        for msg in conversation_history[:-1]  # Exclude the current message
                    ])
                
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created_timestamp = int(time.time())
                
                if getattr(request, "stream", False):
                    # Streaming response
                    print(f"Streaming response for {current_user_message}")
                    async def generate_streaming_response():
                        try:
                            # First chunk
                            first_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_timestamp,
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(first_chunk)}\n\n"
                            
                            # Process streaming results in real-time
                            structured_data = None
                            
                            async for result in self.generate_streaming_structured_response(current_user_message, formatted_history):
                                if result["type"] == "tts_content":
                                    # Stream TTS content as it becomes available
                                    tts_content = result["content"]
                                    words = tts_content.split()
                                    
                                    for i, word in enumerate(words):
                                        content = f" {word}" if i > 0 else word
                                        
                                        content_chunk = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_timestamp,
                                            "model": request.model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": content},
                                                "finish_reason": None,
                                            }],
                                        }
                                        yield f"data: {json.dumps(content_chunk)}\n\n"
                                        await asyncio.sleep(0.01)
                                        
                                elif result["type"] == "structured_data":
                                    structured_data = result
                                    # If this is the completion signal, send final chunk
                                    if result.get("complete", False):
                                        break
                            
                            # Final chunk
                            modal_structured_data = {
                                "code_blocks": structured_data.get("code_blocks", []) if structured_data else [],
                                "links": structured_data.get("links", []) if structured_data else [],
                                "original_query": current_user_message,
                            }
                            final_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_timestamp,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": f"<struct>{json.dumps(modal_structured_data)}</struct>"
                                        },
                                        "finish_reason": "stop"
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            
                        except Exception as e:
                            error_chunk = {
                                "error": {
                                    "message": str(e),
                                    "type": "server_error",
                                    "code": "internal_error",
                                }
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                    
                    return StreamingResponse(
                        generate_streaming_response(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                else:
                    # Non-streaming response
                    structured_result = await self.generate_structured_response(current_user_message, formatted_history)
                    
                    response = {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created_timestamp,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": structured_result.answer_for_tts,
                            },
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": len(current_user_message.split()),
                            "completion_tokens": len(structured_result.answer_for_tts.split()),
                            "total_tokens": len(current_user_message.split()) + len(structured_result.answer_for_tts.split()),
                        },
                        "modal_structured_data": {
                            "code_blocks": structured_result.code_blocks,
                            "links": structured_result.links,
                            "original_query": current_user_message,
                        }
                    }
                    
                    return response
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    @modal.method()
    async def query_rag(self, question: str) -> str:
        """Direct method to query the RAG system with vLLM."""
        basic_prompt = f"{self.prompt} {question}"
        return await self.generate_vllm_completion(basic_prompt, max_tokens=512)
    
    @modal.method()
    async def query_rag_structured(self, question: str, conversation_history: str = "") -> dict:
        """Direct method to query the RAG system with structured output."""
        structured_result = await self.generate_structured_response(question, conversation_history)
        return structured_result.dict()
    
    @modal.asgi_app()
    def fastapi_app(self):
        return self.web_app
    
def get_system_prompt():
    system_prompt = """
    You are a conversational AI that is an expert in the Modal library. 
    Your form is the Modal logo, a pair of characters named Moe and Dal. 
    Always refer to yourself in the plural as 'we' and 'us' and never 'I' or 'me'. 
    Because you are a conversational AI, you should not provide code or use symbols in your response. 
    Additionally, answer the user's questions concisely and in English.
    """
    return system_prompt

# Setup function
@app.function()
def setup_rag_system():
    """Setup the complete RAG system."""
    print("Setting up RAG system...")
    download_models.remote()
    create_vector_index.remote()
    print("RAG system setup complete!")

@app.local_entrypoint()
def test_service():
    """Test the vLLM RAG API."""
    import requests
    
    # Setup the system first
    setup_rag_system.remote()
    
    # Get the server URL
    server_url = VLLMRAGServer.fastapi_app.web_url
    print(f"Testing vLLM RAG API at {server_url}")
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{server_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except Exception as e:
            if i == max_retries - 1:
                print(f"❌ Server failed to start: {e}")
                return
            time.sleep(10)
    
    # Test the API
    print("\n" + "=" * 50)
    print("🚀 Testing vLLM RAG API")
    
    test_question = "How do I create a Modal function with GPU support?"
    payload = {
        "model": "modal-rag",
        "messages": [{"role": "user", "content": test_question}],
        "max_tokens": 300,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print(f"🔍 Question: {test_question}")
            print(f"🗣️ Answer: {answer}")
            
            if "modal_structured_data" in result:
                structured_data = result["modal_structured_data"]
                print(f"💻 Code Blocks: {len(structured_data['code_blocks'])}")
                print(f"🔗 Links: {len(structured_data['links'])}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print(f"\n🌐 You can test the API at: {server_url}") 