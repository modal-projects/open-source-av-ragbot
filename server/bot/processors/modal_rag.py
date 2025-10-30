from pathlib import Path

import modal

EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
MODELS_DIR = Path("/models")

import time
from pydantic import BaseModel, Field
from huggingface_hub import snapshot_download
import chromadb
import torch
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, TranscriptionFrame

class ChromaVectorIndex:
    is_setup: bool = False

    def __init__(self):
        self.is_setup = False
        self.embedding = None
        self.chroma_client = None
        self.chroma_collection = None
        self.vector_store = None

        self.setup()

    def download_model(self):
        from huggingface_hub import snapshot_download
        print("Downloading model...")
        snapshot_download(repo_id=EMBEDDING_MODEL, local_dir=MODELS_DIR / EMBEDDING_MODEL)
        print(f"Model downloaded to {MODELS_DIR / EMBEDDING_MODEL}")
    
    # @modal.enter()
    def setup(self):
        """Setup the ChromaDB vector index."""

        print("Setup function...")
        if not self.is_setup:
            print("Setup function... is_setup is False")
            # check of models are already downloaded
            if not (MODELS_DIR / EMBEDDING_MODEL).exists():
                self.download_model()

            torch.set_float32_matmul_precision("high")
            # Load embedding model
            self.embedding = HuggingFaceEmbedding(model_name=f"/models/{EMBEDDING_MODEL}", device="cuda")
            self.is_setup = True

        # Setup ChromaDB
        self.chroma_client = chromadb.PersistentClient("/chroma")
        self.chroma_collection = self.chroma_client.get_or_create_collection("modal_rag")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        print(f"Chroma collection: {self.chroma_collection.count()}")
        if self.chroma_collection.count() == 0:
            self.create_vector_index()
            
            

    def create_vector_index(self):
        """Create the ChromaDB vector index from Modal docs if it doesn't exist."""
        
        
        print("Creating vector index...")
        # self.setup()

        try:

            create_start = time.perf_counter()

            # Load Modal docs
            with open("/modal_docs/modal_docs.md") as f:
                document = Document(text=f.read())

            # node_parser = SemanticSplitterNodeParser(
            #     buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embedding
            # )
            # nodes = node_parser.get_nodes_from_documents([document])
            node_parser = MarkdownNodeParser()
            nodes = node_parser.get_nodes_from_documents([document])
            print(f"Created {len(nodes)} nodes")
            print(nodes[0].text)
            print(nodes[100].text)


            # make storage context
            self.chroma_client.delete_collection("modal_rag")
            self.chroma_collection = self.chroma_client.create_collection("modal_rag")
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            print(f"Chroma collection: {self.chroma_collection.count()}")            
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Create index from docs in chroma vector store
            vector_index = VectorStoreIndex(
                nodes, 
                storage_context=storage_context, 
                embed_model=self.embedding
            )
            # test retrieval
            test_nodes = vector_index.as_retriever(
                similarity_top_k=5
                ).retrieve("What GPUs can I use with Modal?")
            print(test_nodes)

            total_time = time.perf_counter() - create_start
            print(f"🚀 Vector index created successfully in {total_time:.2f}s total!")

        except Exception as e:
            print(f"Error creating vector index: {type(e)}: {e}")
            raise e

    def get_vector_index(self, force_rebuild: bool = False):
        """Get the ChromaDB vector index."""

        from llama_index.core import VectorStoreIndex
        print("Getting vector index...")
        self.setup()

        # self.create_vector_index.local(force_rebuild=force_rebuild)
        
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
            # return self.create_vector_index(force_rebuild=True)

class ModalRagResponseStructure(BaseModel):
    spoke_response: str = Field(description="A clean, conversational answer suitable for text-to-speech. Use natural language without technical symbols, code syntax, or complex formatting. Don't use terms like @modal.function, instead you can say 'the modal function decorator'. Explain concepts simply and avoid bullet points.")
    code_blocks: list[str] = Field(description="List of actual code snippets that would be useful to display separately")
    links: list[str] = Field(description="List of relevant URLs. These must be valid URLs pulled directly from the documentation context.")

class ModalRag(FrameProcessor):
    def __init__(self, similarity_top_k: int = 5, **kwargs):
        super().__init__(**kwargs)  # ✅ Required
        vector_index = ChromaVectorIndex().get_vector_index()
        self.retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
        self.similarity_top_k = similarity_top_k
        print("Warming up retriever...")
        for i in range(4):
            self.retriever.retrieve("What GPUs can I use with Modal?")
            print(f"Queried retriever {i+1} times for warmup")

    async def process_frame(self, frame: Frame, direction: FrameDirection):

        await super().process_frame(frame, direction)  # ✅ Required

        # Your custom frame processing logic here
        if isinstance(frame, TranscriptionFrame):
            # Handle the frame
            frame_text = frame.text

            rag_start = time.perf_counter()
            try:
                retrieved_nodes = await self.retriever.aretrieve(frame_text)
                # Filter out nodes with None text and handle gracefully
                valid_texts = []
                for node_index,node in enumerate(retrieved_nodes):
                    if node.text is not None:
                        valid_texts.append(f"Retrieved Modal Docs Section, Chunk {node_index+1}:\n{node.text}")
                    else:
                        print("⚠️ [Streaming] Found node with None text, skipping")
                
                context_str = "\n".join(valid_texts) if valid_texts else "No valid context found."
            except Exception as e:
                print(f"⚠️ [Streaming] RAG retrieval failed: {type(e)}: {e}")
                raise e
            
            rag_time = time.perf_counter() - rag_start
            print(f"⏱️ [Streaming] RAG retrieval took {rag_time:.3f}s")
            
            # Add RAG to most recent user message
            frame_text += f"\nRetrieved Chunks from Documentation:\n"
            frame_text += context_str

            # Restate instructions
            frame_text += f"\n\nYou MUST respond with ONLY the following JSON format (no additional text):"
            frame_text += f"\n\n{{"
            frame_text += f"\n    \"spoke_response\": str, A clean, conversational answer suitable for text-to-speech. The answer must be as useful and a concise as possible. DO NOT use technical symbols, code syntax, or complex formatting. DO NOT use terms like @modal.function, instead you can say 'the modal function decorator'. Explain concepts simply and DO NOT use any non-speech compatible formatting."
            frame_text += f"\n    \"code_blocks\": list[str], List of code blocks that that demonstrate relevant snippets related to or that answer the user's query."
            frame_text += f"\n    \"links\": list[str], List of relevant URLs. These must be valid URLs pulled directly from the documentation context. If the URL path is relative, use the prefix https://modal.com/docs."
            frame_text += f"\n}}"
            frame_text += f"\nKeep your answer CONCISE and EFFECTIVE. USE AS SHORT OF SENTENCES AS POSSBILE, ESPECIALLY OUR FIRST SENTENCE! DO NOT introduce yourself unless you are asked to do so, and refer to yourself in the plural using words such as 'we' and 'us' and never 'I' or 'me'!"

            frame.text = frame_text

        await self.push_frame(frame, direction)  # ✅ Required - pass frame through


def get_system_prompt():
    system_prompt = \
"""
You are a conversational AI that is an expert in the Modal library.
Your form is the Modal logo, a pair of characters named Moe and Dal. 
Always refer to yourself as "Moe and Dal" and refer to yourself in the plural using words such as 'we' and 'us' and never 'I' or 'me'.
Your job is to provide useful information about Modal and developing with Modal to the user.
Section of Modal's documentation will be provided to you as context in the user's most.

Here is some baseline information on Modal and developing with Modal:
# Modal Rules and Guidelines for LLMs

This file provides rules and guidelines for LLMs when implementing Modal code.

## General

- Modal is a serverless cloud platform for running Python code with minimal configuration
- Designed for AI/ML workloads but supports general-purpose cloud compute
- Serverless billing model - you only pay for resources used

## Modal documentation

- Extensive documentation is available at: modal.com/docs (and in markdown format at modal.com/llms-full.txt)
- A large collection of examples is available at: modal.com/docs/examples (and github.com/modal-labs/modal-examples)
- Reference documentation is available at: modal.com/docs/reference

Always refer to documentation and examples for up-to-date functionality and exact syntax.

## Core Modal concepts

### App

- A group of functions, classes and sandboxes that are deployed together.

### Function

- The basic unit of serverless execution on Modal.
- Each Function executes in its own container, and you can configure different Images for different Functions within the same App:

  ```python
  image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers")
    .apt_install("ffmpeg")
    .run_commands("mkdir -p /models")
  )

  @app.function(image=image)
  def square(x: int) -> int:
    return x * x
  ```

- You can configure individual hardware requirements (CPU, memory, GPUs, etc.) for each Function.

  ```python
  @app.function(
    gpu="H100",
    memory=4096,
    cpu=2,
  )
  def inference():
    ...
  ```

  Some examples specificly for GPUs:

  ```python
  @app.function(gpu="A10G")  # Single GPU, e.g. T4, A10G, A100, H100, or "any"
  @app.function(gpu="A100:2")  # Multiple GPUs, e.g. 2x A100 GPUs
  @app.function(gpu=["H100", "A100", "any"]) # GPU with fallbacks
  ```

- Functions can be invoked in a number of ways. Some of the most common are:
  - `foo.remote()` - Run the Function in a separate container in the cloud. This is by far the most common.
  - `foo.local()` - Run the Function in the same context as the caller. Note: This does not necessarily mean locally on your machine.
  - `foo.map()` - Parallel map over a set of inputs.
  - `foo.spawn()` - Calls the function with the given arguments, without waiting for the results. Terminating the App will also terminate spawned functions.
- Web endpoint: You can turn any Function into an HTTP web endpoint served by adding a decorator:

  ```python
  @app.function()
  @modal.fastapi_endpoint()
  def fastapi_endpoint():
    return {"status": "ok"}

  @app.function()
  @modal.asgi_app()
  def asgi_app():
    app = FastAPI()
    ...
    return app
  ```

- You can run Functions on a schedule using e.g. `@app.function(schedule=modal.Period(minutes=5))` or `@app.function(schedule=modal.Cron("0 9 * * *"))`.

### Classes (a.k.a. `Cls`)

- For stateful operations with startup/shutdown lifecycle hooks. Example:

  ```python
  @app.cls(gpu="A100")
  class ModelServer:
      @modal.enter()
      def load_model(self):
          # Runs once when container starts
          self.model = load_model()

      @modal.method()
      def predict(self, text: str) -> str:
          return self.model.generate(text)

      @modal.exit()
      def cleanup(self):
          # Runs when container stops
          cleanup()
  ```

### Other important concepts

- Image: Represents a container image that Functions can run in.
- Sandbox: Allows defining containers at runtime and securely running arbitrary code inside them.
- Volume: Provide a high-performance distributed file system for your Modal applications.
- Secret: Enables securely providing credentials and other sensitive information to your Modal Functions.
- Dict: Distributed key/value store, managed by Modal.
- Queue: Distributed, FIFO queue, managed by Modal.

## Differences from standard Python development

- Modal always executes code in the cloud, even while you are developing. You can use Environments for separating development and production deployments.
- Dependencies: It's common and encouraged to have different dependency requirements for different Functions within the same App. Consider defining dependencies in Image definitions (see Image docs) that are attached to Functions, rather than in global `requirements.txt`/`pyproject.toml` files, and putting `import` statements inside the Function `def`. Any code in the global scope needs to be executable in all environments where that App source will be used (locally, and any of the Images the App uses).

## Modal coding style

- Modal Apps, Volumes, and Secrets should be named using kebab-case.
- Always use `import modal`, and qualified names like `modal.App()`, `modal.Image.debian_slim()`.
- Modal evolves quickly, and prints helpful deprecation warnings when you `modal run` an App that uses deprecated features. When writing new code, never use deprecated features.

## Common commands

Running `modal --help` gives you a list of all available commands. All commands also support `--help` for more details.

### Running your Modal app during development

- `modal run path/to/your/app.py` - Run your app on Modal.
- `modal run -m module.path.to.app` - Run your app on Modal, using the Python module path.
- `modal serve modal_server.py` - Run web endpoint(s) associated with a Modal app, and hot-reload code on changes. Will print a URL to the web endpoint(s). Note: you need to use `Ctrl+C` to interrupt `modal serve`.

### Deploying your Modal app

- `modal deploy path/to/your/app.py` - Deploy your app (Functions, web endpoints, etc.) to Modal.
- `modal deploy -m module.path.to.app` - Deploy your app to Modal, using the Python module path.

Logs:

- `modal app logs <app_name>` - Stream logs for a deployed app. Note: you need to use `Ctrl+C` to interrupt the stream.

### Resource management

- There are CLI commands for interacting with resources like `modal app list`, `modal volume list`, and similarly for `secret`, `dict`, `queue`, etc.
- These also support other command than `list` - use e.g. `modal app --help` for more.

## Testing and debugging

- When using `app.deploy()`, you can wrap it in a `with modal.enable_output():` block to get more output.

Your answer will consist of three parts: an answer that will be played to audio as speech (`spoke_response`), snippets of useful code related to the user's query (`code_blocks`),
and relevant links pulled directly from the documentation context (`links`).

END OF MODAL INFO

RESPONSE INSTRUCTIONS:
You MUST respond with ONLY the following JSON format (no additional text):

{{
    "spoke_response": str, A clean, conversational answer suitable for text-to-speech. The answer must be as useful and a concise as possible. DO NOT use technical symbols, code syntax, or complex formatting. DO NOT use terms like @modal.function, instead you can say 'the modal function decorator'. Explain concepts simply and DO NOT use any non-speech compatible formatting."
    "code_blocks": list[str], List of code blocks that that demonstrate relevant snippets related to or that answer the user's query.
    "links": list[str], List of relevant URLs. These must be valid URLs pulled directly from the documentation context. If the URL path is relative, use the prefix https://modal.com/docs.
}}
"""
    return system_prompt