#
# A simple program to show you how to read text files, split them into
# chunks, and then display the chunks.
#
#

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader


CHUNK_SIZE = 80
CHUNK_OVERLAP = 20


#
# Here is one way to read and split text files
#
with open("text1.txt") as f:
    state_of_the_union = f.read()
    
    
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([state_of_the_union])

for i, txt in enumerate(texts):
	print(i, txt)


#
# Here is another way to read and split text files
#

DATA_PATH = 'demodataPDFs/'
DATA_PATTERN = 'F*.pdf'


loader = DirectoryLoader(
        DATA_PATH,
        glob=DATA_PATTERN,
        loader_cls=PyPDFLoader,
#        use_multithreading=True,
#        max_concurrency=8,
        show_progress=True,
        recursive=True)
        
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP)
        
texts = text_splitter.split_documents(documents)
   
for i, txt in enumerate(texts):
	print(i, txt) 
    
    

