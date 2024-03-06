import llama_index
import chromadb
from importlib.metadata import version

print(f"LlamaIndex version: {version('llama_index')}")
print(f"Chroma version: {version('chromadb')}")

# Load API key from .env file

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Define embedding model and LLM

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.embed_model = OpenAIEmbedding()

# Load the index with some example data

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./data/paul_graham_essay.txt"],
).load_data()

# Chunk documents into nodes

from llama_index.core.node_parser import SentenceWindowNodeParser

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Extract nodes from documents
nodes = node_parser.get_nodes_from_documents(documents)

# Build the index

client = chromadb.EphemeralClient()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    device="cuda",
)

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

index_name = "MyExternalContent"

# Construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=client.create_collection(name=index_name),
)

# Set up the storage for the embeddings
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Setup the index
# build VectorStoreIndex that takes care of chunking documents
# and encoding chunks to embeddings for future retrieval
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)

# Setup the query engine

from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# The target key defaults to `window` to match the node_parser's default
postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window",
)

from llama_index.core.postprocessor import SentenceTransformerRerank

# Define reranker model
rerank = SentenceTransformerRerank(
    top_n = 2, 
    model = "BAAI/bge-reranker-base",
    device = "cuda",
)

query_engine = index.as_query_engine( 
	similarity_top_k = 6,
    vector_store_query_mode="hybrid", 
    alpha=0.5,
    node_postprocessors = [postproc, rerank],
)

# Run a query against the naive RAG implementation

response = query_engine.query(
    "What happened at InterLeaf?",
)
print(response)

window = response.source_nodes[0].node.metadata["window"]
sentence = response.source_nodes[0].node.metadata["original_text"]

print(f"Window: {window}")
print("------------------")
print(f"Original Sentence: {sentence}")