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
    input_files=["./data/paul_graham_essay.txt"]
).load_data()

# Chunk documents into nodes

from llama_index.core.node_parser import SimpleNodeParser

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
nodes = node_parser.get_nodes_from_documents(documents)

# Build the index

import chromadb

index_name = "MyExternalContent"

client = chromadb.EphemeralClient()
collection = client.create_collection(name=index_name)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    device="cuda",
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

# Setup the query engine

query_engine = index.as_query_engine()

# Run a query against the naive RAG implementation

response = query_engine.query(
    "What happened at InterLeaf?",
)
print(response)