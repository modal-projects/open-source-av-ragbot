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
        for _ in range(4):
            self.retriever.retrieve("What GPUs can I use with Modal?")

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