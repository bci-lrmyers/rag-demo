# rag-demo

The following instructions loosely follow the text of the following article:
    * [DEMO: Langchain + RAG Demo on LlaMa-2–7b + Embeddings Model using Chainlit](https://medium.com/@madhur.prashant7/demo-langchain-rag-demo-on-llama-2-7b-embeddings-model-using-chainlit-559c10ce3fbf)
	
## Steps

1. Clone the repo.
    ```
	git clone git@github.com:bci-lrmyers/rag-demo.git
	cd rag-demo
	```
1. Create the anaconda environment:
    ```
	conda create --name rag-demo python=3.10
	conda activate rag-demo
	```
1. Install jupyter notebook:
    ```
    conda install jupyter
    ```
1. Install pytorch. Installation instructions copied
   from [here](https://pytorch.org/get-started/locally/).
    * For CPU only:
        ```
        conda install pytorch torchvision torchaudio cpuonly -c pytorch
        ```
    * For GPU:
        ```
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	    ```
	    * Test pytorch installation:
            ```
		    import torch
		    print(torch.cuda.is_available())
		    ```
1. Install the cuda version of llama.cpp python bindings. Installation instructions
   copied from [here](https://github.com/abetlen/llama-cpp-python).
    * For Windows:
        ```
	    set CMAKE_ARGS=-DLLAMA_CUBLAS=on
	    python -m pip install --verbose llama-cpp-python 
        ```
    * For Linux:
        ```
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" python -m pip install --verbose llama-cpp-python
	    ```
2. Install FAISS. FAISS is used as the vector store for the documents. The
   facebook instructions for installing FAISS are located [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).
   Install the GPU version.
    ```
	conda install -c conda-forge faiss-gpu
    ```
	* Test FAISS installation:
	    ```
		import faiss
		print(faiss.__version__)
		```
	If you get an undefined symbol cublasLt_for_cublas_HSS, then you need to modify your LD_LIBRARY_PATH
	to something like
	export LD_LIBRARY_PATH=/home/yourname/.conda/envs/rag-demo/lib:/$LD_LIBRARY_PATH
	Then go back and retest the installation.
	
3. Install LangChain and langchain-community:
    ```
	python -m pip install langchain langchain-community protobuf==3.19 google-auth
	python -m pip install pypdf sentence-transformers ctransformers
	```
4. Install chainlit for UI code. Instructions for installing chainlit are
   found [here](https://docs.chainlit.io/get-started/installation).
    ```
	python -m pip install chainlit protobuf==3.19
	```
	* Test the chainlit installation:
	```
	chainlit hello
	```
5. Create filesystem structure.
    ```
	mkdir demodataPDFs
	mkdir vectorstore
	mkdir vectorstore/db_faiss
	```
6. Place PDF files in the demodataPDFs directory then build the
   vector database. This may take some time depending on the number
   of PDF files.
    ```
	python ingest.py
	```
7. Launch the UI and ask questions.
    ```
	chainlit run rag-demo.py -w
	```
	the file run.bat also has this command inside of it.

## Encodings

Take a look at the [HuggingFace Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
for rankings of the sentence transformers. The [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
sentence transformer converts sentences to 384 dimension vector space and ranks 53rd on the
leader board. The [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
from HuggingFace transforms to 768 dimensional vector space and ranks 45th on the leader
board. Look [here](https://huggingface.co/sentence-transformers?sort_models=downloads#models)
for a complete list of the sentence tranformer models available at HuggingFace.

* ```EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'```
* ```EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'```

## Advanced RAG with llamaindex

* [Advanced Retrieval-Augmented Generation: From Theory to LlamaIndex Implementation](https://towardsdatascience.com/advanced-retrieval-augmented-generation-from-theory-to-llamaindex-implementation-4de1464a9930)

### Setup steps:

1. Install llamaindex.
    ```
    python -m pip install llama-index
    ```
1. Install the Weaviate vector database.
    ```
    python -m pip install weaviate-client llama-index-vector-stores-weaviate
    ```
1. Create a file with the name .env containing the following:
    ```
    OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"

1. Install the dotenv library
   ```
   python -m pip install python-dotenv
   ```

