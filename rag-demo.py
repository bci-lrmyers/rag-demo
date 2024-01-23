from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

print("hello")

DB_FAISS_PATH = 'vectorstore/db_faiss'

demo_prompt_template = """Use the following pieces of information to answer the user’s question.
If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful and Caring answer:
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


# Load the locally downloaded model here
def load_llm():
    llm = CTransformers(
            model = "meta/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens = 512,
            temperature = 0.5)
    return llm


#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
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
    msg.content = "Welcome to Demo Bot!. Ask your question here:"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["documents"]
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
    await cl.Message(content=answer).send()


