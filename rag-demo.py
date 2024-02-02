from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain_community.llms import LlamaCpp
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.base import BaseCallbackHandler

DB_FAISS_PATH = 'vectorstore/db_faiss'
LLM_MODEL = 'TheBloke/Llama-2-70B-Chat-GGUF'
# EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# EMBEDDING_ARGS = {'device': 'cpu'}
# EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDING_MODEL = 'embaas/sentence-transformers-e5-large-v2'
EMBEDDING_ARGS = {'device': 'cuda'}
ENCODE_ARGS = {'normalize_embeddings': False}

n_gpu_layers = 41  # Change this value based on your model and your GPU VRAM pool.
# n_gpu_layers = 81  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# demo_prompt_template = """Use the following pieces of information to answer the user’s question.
# If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
# 
# Context: {context}
# Question: {question}
# 
# Only return the helpful answer below and nothing else.
# Helpful and Caring answer:
# """
demo_prompt_template = """Use the following pieces of information to answer the user’s question.
If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.

Context: {context}
Question: {question}

Answer:
"""

# Prompt template for QA retrieval for each vectorstore
def custom_prompt():
    prompt = PromptTemplate(
            template=demo_prompt_template,
            input_variables=['context', 'question'])
    return prompt


#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt})
    return qa_chain


# class MyCustomSyncHandler(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         stream = cl.user_session.get("stream")
#         stream["tokens"] += token
#         # print(stream["tokens"])


# Load the locally downloaded model here
def load_llm():
    # llm = CTransformers(
    #         model = LLM_MODEL,
    #         model_type="llama",
    #         max_new_tokens = 512,
    #         temperature = 0.5,
    #         lib='cuda')

    # Callbacks support token-wise streaming
    # callback_manager = CallbackManager([MyCustomSyncHandler()])

    #   model_path="./models/13b_Q5_K_M.gguf",
    #   model_path="./models/llama-2-70b-chat.Q4_K_M.gguf", # TheBloke/Llama-2-70B-Chat-GGUF
    #   model_path="./models/llama-2-13b-chat.Q5_K_M.gguf", # TheBloke/Llama-2-13B-chat-GGUF

    llm = LlamaCpp(
        model_path="./models/llama-2-13b-chat.Q5_K_M.gguf", # TheBloke/Llama-2-13B-chat-GGUF
        temperature=0.2,
        # model_kwargs={ "context_window":4096, "max_new_tokens":4096 },
        n_ctx=4096,
        max_tokens=4096,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        # callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    return llm


#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=EMBEDDING_ARGS,
            encode_kwargs=ENCODE_ARGS,
            multi_process=False)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    # gpu_res = FAISS.StandardGpuResources()
    # cpu_index_flat = FAISS.IndexFlatL2(db)
    # gpu_index_flat = FAISS.index_cpu_to_gpu(gpu_res, 0, cpu_index_flat)

    llm = load_llm()
    qa_prompt = custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting your gen AI bot!...")
    await msg.send()
    msg.content = "Welcome to VulcanGPT!. Ask your question here:"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    # stream = { "tokens":"" }
    # cl.user_session.set("stream", stream)

    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])

    # answer = stream["tokens"]
    answer = res["result"]
    sources = res["source_documents"]
    if sources:
        answer += f"\n\nSources: " + str(sources)
    else:
        answer += "\n\nNo sources found"
    await cl.Message(content=answer).send()


