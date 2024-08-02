"""
This script creates a database of information gathered from local text files.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def saveToDB():
    # define what documents to load
    loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)

    # interpret information in the documents
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})

    # create and save the local database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss")
