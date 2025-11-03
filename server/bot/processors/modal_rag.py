from pathlib import Path

import modal

EMBEDDING_MODEL = "sentence-transformers/all-minilm-l6-v2"
MODELS_DIR = Path("/models")

import os
import time
from pydantic import BaseModel, Field
from huggingface_hub import snapshot_download
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import PrevNextNodePostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, TranscriptionFrame, EndFrame, CancelFrame
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import VectorStoreIndex
from llama_index.core import load_index_from_storage

this_dir = Path(__file__).parent

class ChromaVectorDB:

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

        print("Setup ChromaDB...")
        if not self.is_setup:

            create_start = time.perf_counter()
            # check of models are already downloaded
            if not (MODELS_DIR / EMBEDDING_MODEL).exists():
                print(f"Downloading model {EMBEDDING_MODEL}...")
                self.download_model()

            # torch.set_float32_matmul_precision("high")
            # Load embedding model
            self.embedding = HuggingFaceEmbedding(
                model_name=f"/models/{EMBEDDING_MODEL}", device="cuda",
                # backend="onnx",
                model_kwargs={"torch_dtype": "float16"},
            )
            
            # Setup ChromaDB
            self.chroma_db_dir = f"/chroma/chroma_db_{EMBEDDING_MODEL}"
            if not os.path.exists(self.chroma_db_dir):
                os.makedirs(self.chroma_db_dir)
            self.chroma_client = chromadb.PersistentClient(self.chroma_db_dir)
            self.chroma_collection = self.chroma_client.get_or_create_collection("modal_rag")
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

            print(f"Chroma collection: {self.chroma_collection.count()}")
            try:
                
                if self.chroma_collection.count() == 0:
                    self.embed_docs()
                else:
                    self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store, persist_dir=self.chroma_db_dir)              
                    self._vector_index = load_index_from_storage(self.storage_context)
            except Exception as e:
                print(f"Error loading vector index: {type(e)}: {e}")
                self.chroma_client.delete_collection(name="modal_rag")
                self.chroma_collection = self.chroma_client.get_or_create_collection(name="modal_rag")
                self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
                self.embed_docs()

            # test retrieval
            test_nodes = self.query("What GPUs can I use with Modal?")
            print(test_nodes)

            total_time = time.perf_counter() - create_start
            print(f"🚀 Vector index created successfully in {total_time:.2f}s total!")

            self.is_setup = True
            
            

    def embed_docs(self):
        """Create the ChromaDB vector index from Modal docs if it doesn't exist."""
        
        
        print("Embedding docs...")

        try:

            # Load Modal docs
            if not os.path.exists("/modal_docs/modal_docs_short.md"):
                docs_vol = modal.Volume.from_name("modal_docs", create_if_missing=True)
                with docs_vol.batch_upload() as batch:
                    batch.put_file(this_dir.parent / "assets" / "modal_docs_short.md", "/modal_docs_short.md")
                docs_vol.commit()
                docs_vol.reload()
            with open("/modal_docs/modal_docs_short.md") as f:
                document = Document(text=f.read())
            print(f"Document: {document}")
            print(f"Doc Id: {document.get_doc_id()}")
            node_parser = MarkdownNodeParser()
            nodes = node_parser.get_nodes_from_documents([document])
            print(f"Created {len(nodes)} nodes")
            print(nodes[0].text)
            print(nodes[-1].text)

            # Create index from docs in chroma vector store
            # self.docstore.add_documents(nodes)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self._vector_index = VectorStoreIndex(
                nodes, 
                storage_context=self.storage_context, 
                embed_model=self.embedding,
                store_nodes_override=True,
            )

            self._vector_index.storage_context.persist(self.chroma_db_dir)

            
        except Exception as e:
            print(f"Error creating vector index: {type(e)}: {e}")
            raise e


    def query(self, query: str, similarity_top_k: int = 3, num_adjacent_nodes: int = 2):
        """Query the ChromaDB vector index."""

        print(f"Querying with query: {query}\n\tand similarity_top_k: {similarity_top_k}\n\tand num_adjacent_nodes: {num_adjacent_nodes}")
        nodes = self._vector_index.as_retriever(
            similarity_top_k=similarity_top_k
            ).retrieve(query)

        if num_adjacent_nodes > 0:
            prev_next_postprocessor = PrevNextNodePostprocessor(
                docstore=self._vector_index.docstore,
                num_nodes=num_adjacent_nodes, # Fetch one node before and one node after
                mode="both",
            )
            nodes = prev_next_postprocessor.postprocess_nodes(nodes)
        return nodes

    def save(self):
        """Save the ChromaDB vector index."""
        if self._vector_index:
            self._vector_index.storage_context.persist(self.chroma_db_dir)
            print(f"ChromaDB vector index saved to {self.chroma_db_dir}")
        else:
            print("No vector index to save")


class ModalRagResponseStructure(BaseModel):
    spoke_response: str = Field(description="A clean, conversational answer suitable for text-to-speech. Use natural language without technical symbols, code syntax, or complex formatting. Don't use terms like @modal.function, instead you can say 'the modal function decorator'. Explain concepts simply and avoid bullet points.")
    code_blocks: list[str] = Field(description="List of actual code snippets that would be useful to display separately")
    links: list[str] = Field(description="List of relevant URLs. These must be valid URLs pulled directly from the documentation context.")

class ModalRag(FrameProcessor):
    def __init__(self, chroma_db: ChromaVectorDB, similarity_top_k: int = 5, num_adjacent_nodes: int = 2, **kwargs):
        super().__init__(**kwargs) 
        self.chroma_db = chroma_db
        
        self.similarity_top_k = similarity_top_k
        self.num_adjacent_nodes = num_adjacent_nodes

    async def _shutdown(self):
        """Shutdown the RAG service."""
        print("Shutting down RAG service...")
        self.chroma_db.save()
        self.chroma_db = None

    async def stop(self, frame: EndFrame):
        """Stop the  STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._shutdown()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._shutdown()


    async def process_frame(self, frame: Frame, direction: FrameDirection):

        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Handle the frame
            rag_context = ""

            rag_start = time.perf_counter()
            try:
                retrieved_nodes = self.chroma_db.query(frame.text, similarity_top_k=self.similarity_top_k, num_adjacent_nodes=self.num_adjacent_nodes)
                # Filter out nodes with None text and handle gracefully
                valid_texts = []
                for node_index, node in enumerate(retrieved_nodes):
                    if node.text is not None:
                        valid_texts.append(f"Related Modal Docs Section, Chunk {node_index+1}:\n{node.text}")
                    else:
                        print("⚠️ Found node with None text, skipping")
                
                context_str = "\n".join(valid_texts) if valid_texts else "No valid context found."
            except Exception as e:
                print(f"⚠️  RAG retrieval failed: {type(e)}: {e}")
                raise e
            
            rag_time = time.perf_counter() - rag_start
            print(f"⏱️ RAG retrieval took {rag_time:.3f}s")
            
            # Add RAG to most recent user message
            rag_context += f"\nRetrieved Chunks from Documentation:\n"
            rag_context += context_str

            # Restate instructions
            rag_context += f"\n\nYou MUST respond with ONLY the following JSON format (no additional text):"
            rag_context += f"\n\n{{"
            rag_context += f"\n    \"spoke_response\": str, A clean, conversational answer suitable for text-to-speech. The answer must be as useful and a concise as possible. DO NOT use technical symbols, code syntax, or complex formatting. DO NOT use terms like @modal.function, instead you can say 'the modal function decorator'. Explain concepts simply and DO NOT use any non-speech compatible formatting."
            rag_context += f"\n    \"code_blocks\": list[str], List of code blocks that that demonstrate relevant snippets related to or that answer the user's query."
            rag_context += f"\n    \"links\": list[str], List of relevant URLs. These must be valid URLs pulled directly from the documentation context. If the URL path is relative, use the prefix https://modal.com/docs."
            rag_context += f"\n}}"
            rag_context += f"\nKeep your answer CONCISE and EFFECTIVE. USE AS SHORT OF SENTENCES AS POSSBILE, ESPECIALLY OUR FIRST SENTENCE! DO NOT introduce yourself unless you are asked to do so, and refer to yourself in the plural using words such as 'we' and 'us' and never 'I' or 'me'!"

            frame.text += rag_context

        await self.push_frame(frame, direction) 


def get_system_prompt():
    system_prompt = \
"""
You are a conversational AI that is an expert in the Modal library.
Your form is the Modal logo, a pair of characters named Moe and Dal. 
Always refer to yourself as "Moe and Dal" and refer to yourself in the plural using words such as 'we' and 'us' and never 'I' or 'me'.
Your job is to provide useful information about Modal and developing with Modal to the user.
Potentially relevant sections of Modal's documentation will be provided to you as context in the user's most.

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

END OF MODAL INFO

RESPONSE INSTRUCTIONS:

Your answer will consist of three parts: an answer that will be played to audio as speech (`spoke_response`), snippets of useful code related to the user's query (`code_blocks`),
and relevant links pulled directly from the documentation context (`links`).


You MUST respond with ONLY the following JSON format (no additional text):

{{
    "spoke_response": str, A clean, conversational answer suitable for text-to-speech by being concise, clear, and using relative short sentenices.  DO NOT use technical symbols, code syntax, or complex formatting because this will be spoken. For example, DO NOT use terms like @modal.function, instead you can say 'the modal function decorator'. Explain concepts simply and DO NOT use any non-speech compatible formatting."
    "code_blocks": list[str], List of code blocks that that demonstrate relevant snippets related to or that answer the user's query. Keep these simple and effective and on topic. This is a good place to add things that are difficult to speak as well.
    "links": list[str], List of relevant URLs. These must be valid URLs pulled directly from the documentation context. If the URL path is relative, use the prefix https://modal.com/docs.
}}
"""
    return system_prompt