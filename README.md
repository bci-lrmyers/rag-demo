# rag-demo

The following instructions loosely follow the text of the following article:
    * [DEMO: Langchain + RAG Demo on LlaMa-2–7b + Embeddings Model using Chainlit](https://medium.com/@madhur.prashant7/demo-langchain-rag-demo-on-llama-2-7b-embeddings-model-using-chainlit-559c10ce3fbf)
	
## Steps

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
1. Install LangChain:
    ```
	pip install langchain
	```
1. Install chainlit for UI code. Instructions for installing chainlit are
   found [here](https://docs.chainlit.io/get-started/installation).
    ```
	pip install chainlit
	```
	* Test the chainlit installation:
	```
	chainlit hello
	```

