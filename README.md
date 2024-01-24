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
1. Install FAISS. FAISS is used as the vector store for the documents. The
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
	
1. Install LangChain and langchain-community:
    ```
	python -m pip install langchain langchain-community
	python -m pip install pypdf sentence-transformers	ctransformers
	```
1. Install chainlit for UI code. Instructions for installing chainlit are
   found [here](https://docs.chainlit.io/get-started/installation).
    ```
	python -m pip install chainlit
	```
	* Test the chainlit installation:
	```
	chainlit hello
	```
1. Create filesystem structure.
    ```
	mkdir demodataPDFs
	mkdir vectorstore
	mkdir vectorstore/db_faiss
	```
1. Place PDF files in the demodataPDFs directory then build the
   vector database. This may take some time depending on the number
   of PDF files.
    ```
	python ingest.py
	```
1. Launch the UI and ask questions.
    ```
	chainlit run rag-demo.py -w
	```
	the file run.bat also has this command inside of it.	
	
