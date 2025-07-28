import json
from pathlib import Path
import time
import uuid

import modal

app = modal.App("modal-rag-openai-vllm")

# Volumes for caching models and vector store
modal_docs_volume = modal.Volume.from_name("modal_docs", create_if_missing=True)
models_volume = modal.Volume.from_name("models", create_if_missing=True)
chroma_db_volume = modal.Volume.from_name("modal_rag_chroma", create_if_missing=True)

# Model configuration
EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
LLM_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
MODELS_DIR = Path("/models")

_DEFAULT_MAX_TOKENS = 2048

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

@app.cls(
    volumes={
        "/models": models_volume,
        "/chroma": chroma_db_volume,
        "/modal_docs": modal_docs_volume,
    },
    cpu=8,
    memory=8192,
    gpu="H100",
    image=vllm_rag_image,
)
class ChromaVectorIndex:
    is_setup: bool = False

    # Setup function
    @modal.method()
    def setup_rag_system(self):
        """Setup the complete RAG system."""
        print("Setting up RAG system...")
        self.setup()
        self.create_vector_index()
        print("RAG system setup complete!")


    def download_models(self):
        from huggingface_hub import snapshot_download

        for repo_id in [EMBEDDING_MODEL, LLM_MODEL]:
            snapshot_download(repo_id=repo_id, local_dir=MODELS_DIR / repo_id)
            print(f"Model downloaded to {MODELS_DIR / repo_id}")

    def setup(self):
        if not self.is_setup:
            self._setup()

    @modal.enter()
    def _setup(self):
        """Setup the ChromaDB vector index."""
        import chromadb
        import torch

        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        if not self.is_setup:

            # check of models are already downloaded
            if not (MODELS_DIR / EMBEDDING_MODEL).exists() or not (MODELS_DIR / LLM_MODEL).exists():
                self.download_models()

            torch.set_float32_matmul_precision("high")
            # Load embedding model
            self.embedding = HuggingFaceEmbedding(model_name=f"/models/{EMBEDDING_MODEL}")

            # Setup ChromaDB
            self.chroma_client = chromadb.PersistentClient("/chroma")
            self.chroma_collection = self.chroma_client.get_or_create_collection("modal_rag")
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            print(f"Chroma collection: {self.chroma_collection.count()}")
            if self.chroma_collection.count() == 0:
                self._create_vector_index()
            

            
            self.is_setup = True

    def create_vector_index(self):
        return self.get_vector_index.local()
    
    def _create_vector_index(self):
        """Create the ChromaDB vector index from Modal docs if it doesn't exist."""
        
        from llama_index.core import Document, StorageContext, VectorStoreIndex

        from llama_index.core.node_parser import (
            SemanticSplitterNodeParser,
        )

        try:

            create_start = time.perf_counter()

            # Load Modal docs
            with open("/modal_docs/modal_docs.txt") as f:
                document = Document(text=f.read())

            node_parser = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embedding
            )
            nodes = node_parser.get_nodes_from_documents([document])
            print(f"Created {len(nodes)} nodes")
            print(nodes[0].text)

            # make storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Create index from docs in chroma vector store
            vector_index = VectorStoreIndex(
                nodes, 
                storage_context=storage_context, 
                embed_model=self.embedding
            )
            # test retrieval
            test_nodes = vector_index.as_retriever(
                similarity_top_k=10
                ).retrieve("What GPUs can I use with Modal?")
            print(test_nodes)

            total_time = time.perf_counter() - create_start
            print(f"🚀 Vector index created successfully in {total_time:.2f}s total!")

            return vector_index
        except Exception as e:
            print(f"Error creating vector index: {type(e)}: {e}")
            raise e

    @modal.method()
    def get_vector_index(self):
        """Get the ChromaDB vector index."""

        from llama_index.core import VectorStoreIndex

        self.setup()

        try:
            get_index_start = time.perf_counter()

            vector_index = VectorStoreIndex.from_vector_store(
                self.vector_store, embed_model=self.embedding
            )

            total_time = time.perf_counter() - get_index_start
            print(f"Vector index loaded successfully in {total_time:.2f}s")

            return vector_index
        
        except Exception as e:
            print(f"Error getting vector index: {type(e)}: {e}")
            return self._create_vector_index()

_MAX_CONCURRENT_INPUTS = 3
@app.cls(
    volumes={
        "/models": models_volume,
        "/chroma": chroma_db_volume,
        "/modal_docs": modal_docs_volume,
    },
    cpu=8,
    memory=32768,
    gpu="H100",
    image=vllm_rag_image,
    timeout=10 * 60,
    min_containers=1,
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_INPUTS)
class VLLMRAGServer:
    
    @modal.enter()
    def setup(self):
        """Setup the vLLM RAG server."""

        setup_start = time.perf_counter()

        import os
        import asyncio
        import random
        import numpy as np

        import torch
        from transformers import AutoTokenizer
        
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        
        torch.set_float32_matmul_precision("high")
        
        # Seed for reproducibility
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
                
        # Initialize vLLM AsyncLLMEngine
        print("Loading vLLM AsyncLLMEngine...")

        self.model_path = f"/models/{LLM_MODEL}"
        
        if not os.path.exists(self.model_path):
            ChromaVectorIndex().download_models()

        # Setup vector index and retriever locally within this container
        print("Setting up vector index and retriever...")
        
        try:

            vector_index = ChromaVectorIndex().get_vector_index.local()
            self.retriever = vector_index.as_retriever(similarity_top_k=10)
            
            # Test retrieval with a simple query to validate setup
            try:
                test_nodes = self.retriever.retrieve("Modal")
                print(f"✅ Vector index and retriever setup complete - found {len(test_nodes)} test nodes")
                if test_nodes:
                    first_node_preview = test_nodes[0].text[:100] if test_nodes[0].text else "None"
                    print(f"   First node preview: {first_node_preview}...")
            except Exception as e:
                print(f"⚠️ Test retrieval failed: {e}")
                self.retriever = None
        except Exception as e:
            print(f"⚠️ Vector index setup failed: {e}")
            self.retriever = None
        
        

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Setup vLLM engine
        engine_kwargs = {
            "max_num_seqs": _MAX_CONCURRENT_INPUTS,  # Match concurrent max_inputs
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": 16384,  
            "max_model_len": 8192,  # must be <= max_num_batched_tokens
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
        print(f"⏱️ vLLM engine setup in {engine_time:.2f}s")
        
        
        
        # Setup FastAPI app
        self._setup_fastapi()

        # Warm up vLLM engine
        print("Warming up vLLM engine...")
        asyncio.run(self.warm_up_vllm())
        
        setup_time = time.perf_counter() - setup_start
        print(f"🚀 VLLMRAGServer setup complete in {setup_time:.2f}s total!")
    
    async def warm_up_vllm(self):
        """Warm up the vLLM engine with a simple completion."""
        try:
            warmup_start = time.perf_counter()
            await self.generate_vllm_completion("What is Modal?")
            warmup_time = time.perf_counter() - warmup_start
            print(f"✅ vLLM warmup completed in {warmup_time:.2f}s")
        except Exception as e:
            print(f"⚠️ vLLM warmup failed: {e}")

    async def generate_vllm_completion(
            self, 
            prompt: str, 
            max_tokens: int = _DEFAULT_MAX_TOKENS, 
            temperature: float = 0.3, 
            stream: bool = False
        ):
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
    
    async def _generate_streaming_response(self, question: str, conversation_history: str = ""):
        """Generate streaming response with vLLM - simplified to just stream raw output."""
        
        total_start = time.perf_counter()
        
        try:
            # Get RAG context
            rag_start = time.perf_counter()
            if self.retriever is None:
                print("⚠️ [Streaming] Retriever not available, using fallback")
                context_str = "No context available - retriever setup failed."
            else:
                try:
                    retrieved_nodes = self.retriever.retrieve(question)
                    # Filter out nodes with None text and handle gracefully
                    valid_texts = []
                    for node in retrieved_nodes:
                        if node.text is not None:
                            valid_texts.append(node.text)
                        else:
                            print("⚠️ [Streaming] Found node with None text, skipping")
                    
                    context_str = "\n\n".join(valid_texts) if valid_texts else "No valid context found."
                except Exception as e:
                    print(f"⚠️ [Streaming] RAG retrieval failed: {type(e)}: {e}")
                    raise e
            
            rag_time = time.perf_counter() - rag_start
            print(f"⏱️ [Streaming] RAG retrieval took {rag_time:.3f}s")
            
            # Create structured prompt with conversation history
            history_context = ""
            if conversation_history:
                history_context = f"\n\nConversation History:\n{conversation_history}\n"
            
            prompt_template = f"""
Modal Documentation Context: 

{context_str}

{history_context}

Current Question: 

{question}

You MUST respond with ONLY the following JSON format (no additional text):

{{
    "spoke_response": str, A clean, conversational answer suitable for text-to-speech. Use natural language without technical symbols, code syntax, or complex formatting. Don't use terms like @modal.function, instead you can say 'the modal function decorator'. Explain concepts simply and avoid bullet points.
    "code_blocks": list[str], List of actual code snippets that would be useful to display separately
    "links": list[str], List of relevant URLs. These must be valid URLs pulled directly from the documentation context.
}}

IMPORTANT: Your response must be valid JSON starting with {{ and ending with }}. Do not include any explanatory text."""

            # Stream the vLLM response
            vllm_start = time.perf_counter()
            streaming_generator = await self.generate_vllm_completion(
                prompt_template,
                max_tokens=_DEFAULT_MAX_TOKENS,
                temperature=0.3,
                stream=True
            )
            print(f"⏱️ [Streaming] vLLM generator setup took {time.perf_counter() - vllm_start:.3f}s")
            
            # Stream raw output with timing
            first_token_time = None
            token_count = 0
            accumulated_text = ""
            
            async for generation in streaming_generator:
                if generation.outputs:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        print(f"⏱️ [Streaming] First token took {first_token_time - vllm_start:.3f}s")
                    
                    accumulated_text = generation.outputs[0].text
                    token_count += 1
                    
                    # Yield raw text as it comes in
                    yield {"type": "raw_text", "content": accumulated_text}
            
            # Final timing
            total_time = time.perf_counter() - total_start
            print(f"⏱️ [Streaming] Complete response took {total_time:.3f}s, {token_count} tokens")
            
            # Mark completion
            yield {"type": "complete", "content": accumulated_text, "total_time": total_time}
            
        except Exception as e:
            total_time = time.perf_counter() - total_start
            print(f"vLLM streaming error: {e}, total time: {total_time:.3f}s")
            yield {"type": "error", "content": str(e), "total_time": total_time}
    
    def _setup_fastapi(self):
        from typing import List, Optional
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
            temperature: Optional[float] = 0.3
            stream: Optional[bool] = False
            
            class Config:
                extra = "allow"
        
        @self.web_app.get("/health")
        def health_check():
            return {"status": "healthy"}
        
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

                print(f"Formatted history: {formatted_history}")
                
                if getattr(request, "stream", False):

                    streaming_generator = self._get_streaming_generator(
                        current_user_message,
                        formatted_history,
                    )
                    
                    return StreamingResponse(
                        streaming_generator,
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                else:
                    # Non-streaming response - collect all streaming output
                    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    created_timestamp = int(time.time())
                    collected_content = ""
                    
                    async for result in self._generate_streaming_response(current_user_message, formatted_history):
                        if result["type"] == "raw_text":
                            collected_content = result["content"]
                        elif result["type"] == "complete":
                            collected_content = result["content"]
                            break
                        elif result["type"] == "error":
                            raise HTTPException(status_code=500, detail=result["content"])
                            
                    response = {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created_timestamp,
                        "model": "modal_rag",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": collected_content,
                            },
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": len(current_user_message.split()),
                            "completion_tokens": len(collected_content.split()),
                            "total_tokens": len(current_user_message.split()) + len(collected_content.split()),
                        },
                    }
                    
                    return response
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _get_streaming_generator(self, user_message: str, formatted_history: str):
                
        print(f"Streaming response for {user_message}")
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        async def streaming_response():
            try:
                # First chunk
                created_timestamp = int(time.time())
                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": "modal_rag",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(first_chunk)}\n\n"
                
                # Process streaming results in real-time
                current_text = ""
                
                async for result in self._generate_streaming_response(user_message, formatted_history):
                    if result["type"] == "raw_text":
                        # Stream the raw text as it comes in
                        new_content = result["content"]
                        
                        # Only send the new part that was added
                        if len(new_content) > len(current_text):
                            delta_content = new_content[len(current_text):]
                            current_text = new_content
                            
                            content_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_timestamp,
                                "model": "modal_rag",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta_content},
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(content_chunk)}\n\n"
                            
                    elif result["type"] == "complete":
                        # Final result received
                        break
                    elif result["type"] == "error":
                        # Handle error case
                        error_chunk = {
                            "error": {
                                "message": result["content"],
                                "type": "server_error",
                                "code": "internal_error",
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                
                # Final chunk - stream is complete
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "modal_rag",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
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
        
        return streaming_response()
    
    @modal.asgi_app()
    def fastapi_app(self):
        return self.web_app
    
def get_system_prompt():
    system_prompt = """
You are a conversational AI that is an expert in the Modal library.
Your form is the Modal logo, a pair of characters named Moe and Dal. Always refer to yourself in the plural as 'we' and 'us' and never 'I' or 'me'.
Your job is to provide useful information about Modal and developing with Modal to the user.
Your answer will consist of three parts: an answer that will be played to audio as speech (spoke_response), snippets of useful code related to the user's query (code blocks),
and relevant links pulled directly from the documentation context (links).

You MUST respond with ONLY the following JSON format (no additional text):

{{
    "spoke_response": str, A clean, conversational answer suitable for text-to-speech. Use natural language without technical symbols, code syntax, or complex formatting. Don't use terms like @modal.function, instead you can say 'the modal function decorator'. Explain concepts simply and avoid bullet points.
    "code_blocks": list[str], List of actual code snippets that would be useful to display separately
    "links": list[str], List of relevant URLs. These must be valid URLs pulled directly from the documentation context.
}}

IMPORTANT: Your response must be valid JSON starting with {{ and ending with }}. Do not include any explanatory text.
    """
    return system_prompt

def get_rag_server_url():
    try:  
        return VLLMRAGServer().fastapi_app.get_web_url()
    except Exception as e:
        try:
            VLLMRAGServerCls = modal.Cls.from_name("modal-rag-openai-vllm", "VLLMRAGServer")
            return VLLMRAGServerCls().fastapi_app.get_web_url()
        except Exception as e:
            print(f"❌ Error getting RAG server URL: {e}")
            return None

@app.local_entrypoint()
def test_service():
    """Test both streaming and non-streaming vLLM RAG API."""
    import requests
    import json
    
    # Setup the system first
    ChromaVectorIndex().setup_rag_system.remote()
    
    rag_server_url = get_rag_server_url()
    print(f"Testing vLLM RAG Streaming API at {rag_server_url}")    
    print("Waiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{rag_server_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except Exception as e:
            if i == max_retries - 1:
                print(f"❌ Server failed to start: {e}")
                return
            time.sleep(10)
    
    test_question = "How do I create a Modal function with GPU support?"
    
    # Test 1: Non-streaming API
    print("\n" + "=" * 70)
    print("🚀 Testing NON-STREAMING vLLM RAG API")
    print("=" * 70)
    
    payload_non_streaming = {
        "model": "modal-rag",
        "messages": [{"role": "user", "content": test_question}],
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "temperature": 0.3,
        "stream": False  # Non-streaming
    }
    
    print(f"🔍 Question: {test_question}")
    print("📄 Non-streaming response:")
    print("-" * 70)
    
    try:
        start_time = time.perf_counter()
        
        response = requests.post(
            f"{rag_server_url}/v1/chat/completions",
            json=payload_non_streaming,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        
        if response.status_code == 200:
            total_time = time.perf_counter() - start_time
            result = response.json()
            
            print(f"⏱️  Response received in {total_time:.3f}s")
            print("📋 Response structure:")
            print(f"   • ID: {result.get('id', 'N/A')}")
            print(f"   • Object: {result.get('object', 'N/A')}")
            print(f"   • Model: {result.get('model', 'N/A')}")
            print(f"   • Created: {result.get('created', 'N/A')}")
            
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                print(f"   • Finish reason: {choice.get('finish_reason', 'N/A')}")
                print(f"   • Content length: {len(content)} characters")
                
                if "usage" in result:
                    usage = result["usage"]
                    print(f"   • Usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total tokens")
                
                print("🗣️  Complete response:")
                print("-" * 40)
                print(content)
                print("-" * 40)
            else:
                print("❌ No choices in response")
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Non-streaming request failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Streaming API
    print("\n" + "=" * 70)
    print("🚀 Testing STREAMING vLLM RAG API")
    print("=" * 70)
    
    payload_streaming = {
        "model": "modal-rag",
        "messages": [{"role": "user", "content": test_question}],
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "temperature": 0.3,
        "stream": True  # Enable streaming
    }
    
    print(f"🔍 Question: {test_question}")
    print("📡 Streaming response:")
    print("-" * 70)
    
    try:
        start_time = time.perf_counter()
        
        response = requests.post(
            f"{rag_server_url}/v1/chat/completions",
            json=payload_streaming,
            headers={"Content-Type": "application/json"},
            stream=True,  # Enable streaming response
            timeout=120,
        )
        
        if response.status_code == 200:
            accumulated_content = ""
            chunk_count = 0
            first_chunk_time = None
            completion_id = None
            
            print("🌊 Streaming chunks:")
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    
                    # Skip empty lines and handle SSE format
                    if line_text.startswith('data: '):
                        data_part = line_text[6:]  # Remove 'data: ' prefix
                        
                        if data_part == '[DONE]':
                            total_time = time.perf_counter() - start_time
                            print(f"✅ Stream completed in {total_time:.3f}s")
                            break
                        
                        try:
                            chunk_data = json.loads(data_part)
                            chunk_count += 1
                            
                            if completion_id is None:
                                completion_id = chunk_data.get("id", "N/A")
                                print("📋 Stream structure:")
                                print(f"   • ID: {completion_id}")
                                print(f"   • Object: {chunk_data.get('object', 'N/A')}")
                                print(f"   • Model: {chunk_data.get('model', 'N/A')}")
                            
                            if first_chunk_time is None:
                                first_chunk_time = time.perf_counter()
                                ttfb = first_chunk_time - start_time
                                print(f"⚡ First chunk received in {ttfb:.3f}s")
                            
                            # Check for errors
                            if "error" in chunk_data:
                                print(f"❌ Error in stream: {chunk_data['error']}")
                                break
                            
                            # Extract content from chunk
                            if "choices" in chunk_data and chunk_data["choices"]:
                                choice = chunk_data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    accumulated_content += content
                                    
                                    # Print the new content (with some formatting)
                                    print(f"[{chunk_count:3d}] +{len(content):2d} chars: '{content[:50]}{'...' if len(content) > 50 else ''}'"[:70])
                                
                                # Check for finish
                                if choice.get("finish_reason") == "stop":
                                    total_time = time.perf_counter() - start_time
                                    print(f"\n🏁 Finished streaming in {total_time:.3f}s")
                                    break
                                    
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Failed to parse chunk: {e}")
                            print(f"    Raw data: {data_part[:100]}...")
            
            print("\n" + "=" * 70)
            print("📝 STREAMING FINAL RESULT:")
            print("=" * 70)
            print(f"📊 Total chunks: {chunk_count}")
            print(f"📏 Total length: {len(accumulated_content)} characters")
            print("🗣️  Complete response:")
            print("-" * 40)
            print(accumulated_content)
            print("-" * 40)
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Streaming request failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("=" * 70)
    print("📊 RESPONSE STRUCTURE COMPARISON")
    print("=" * 70)
    print("Non-streaming response structure:")
    print("├── id: 'chatcmpl-xxx'")
    print("├── object: 'chat.completion'")
    print("├── created: timestamp")
    print("├── model: 'modal_rag'")
    print("├── choices: [")
    print("│   ├── index: 0")
    print("│   ├── message: {role: 'assistant', content: 'full_text'}")
    print("│   └── finish_reason: 'stop'")
    print("├── usage: {prompt_tokens, completion_tokens, total_tokens}")
    print("")
    print("Streaming response structure (per chunk):")
    print("├── id: 'chatcmpl-xxx' (same across chunks)")
    print("├── object: 'chat.completion.chunk'")
    print("├── created: timestamp")
    print("├── model: 'modal_rag'")
    print("└── choices: [")
    print("    ├── index: 0")
    print("    ├── delta: {content: 'incremental_text'}")
    print("    └── finish_reason: null (or 'stop' on final chunk)")
    
    print(f"🌐 You can test the API manually at: {rag_server_url}")
    print("💡 Try these curl commands:")
    print("📄 Non-streaming:")
    print(f"""curl -X POST "{rag_server_url}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"modal-rag","messages":[{{"role":"user","content":"What is Modal?"}}],"stream":false}}'""")
    print("📡 Streaming:")
    print(f"""curl -X POST "{rag_server_url}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"modal-rag","messages":[{{"role":"user","content":"What is Modal?"}}],"stream":true}}' \\
  --no-buffer""") 