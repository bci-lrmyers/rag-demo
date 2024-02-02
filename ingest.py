from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'demodataPDFs/'
DATA_PATTERN = '*.pdf'

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# EMBEDDING_ARGS = {'device': 'cpu'}
# EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDING_MODEL = 'embaas/sentence-transformers-e5-large-v2'
EMBEDDING_ARGS = {'device': 'cuda'}
ENCODE_ARGS = {'normalize_embeddings': False}

DB_FAISS_PATH = 'vectorstore/db_faiss'


# Create vector database
def create_vector_db():
    loader = DirectoryLoader(
            DATA_PATH,
            glob=DATA_PATTERN,
            loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=EMBEDDING_ARGS,
            encode_kwargs=ENCODE_ARGS,
            multi_process=True)
    db = FAISS.from_documents(texts, embeddings)

    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()
