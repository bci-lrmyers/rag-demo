from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

print("Starting VulcanGPT...")

# Language model parameters.

MODEL_PATH = "./models/llama-2-13b-chat.Q5_K_M.gguf" # TheBloke/Llama-2-13B-chat-GGUF
MODEL_TEMP = 0.2
MODEL_CONTEXT_SIZE = 4096
MODEL_GPU_LAYERS = 41 # Change this value based on your model and your GPU VRAM pool.
MODEL_BATCH_SIZE = 512  # Should be between 1 and MODEL_CONTEXT_SIZE, consider the amount of VRAM in your GPU.
MODEL_MAX_TOKENS = 4096

# Embedding model parameters.

EMBEDDING_MODEL = 'embaas/sentence-transformers-e5-large-v2'
EMBEDDING_ARGS = {'device': 'cuda'}
EMBEDDINT_ENCODE_ARGS = {'normalize_embeddings': False}

# Vector database parameters.

VECTOR_DB_PATH = 'vectorstore/db_faiss'

# Construct the prompt template.

prompt = PromptTemplate(
    template="""
        Use the following pieces of information to answer the user’s question.
        If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful and Caring answer: """,
    input_variables=['context', 'question']
)

# Load the language model.

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=MODEL_TEMP,
    n_ctx=MODEL_CONTEXT_SIZE,
    max_tokens=MODEL_MAX_TOKENS,
    n_gpu_layers=MODEL_GPU_LAYERS,
    n_batch=MODEL_BATCH_SIZE,
    verbose=False,
)

# Construct the embedding model.

embeddings = HuggingFaceEmbeddings(
   model_name=EMBEDDING_MODEL,
   model_kwargs=EMBEDDING_ARGS,
   encode_kwargs=EMBEDDINT_ENCODE_ARGS,
   multi_process=False
)

# Load the vector database.

db = FAISS.load_local(VECTOR_DB_PATH, embeddings)

# Build the chain.

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

print()
print("Welcome to VulcanGPT.")
print()

while True:
    user_input = input("you: ").strip()
    if user_input == "exit":
        break
    if user_input == "":
        continue

    response = chain.invoke(user_input)

    print("VulcanGPT: ", response["result"])
    print()
    print("Sources: ", response["source_documents"])
    print()
    print()

