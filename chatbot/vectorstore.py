
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Typesense
from langchain_community.document_loaders import Docx2txtLoader

from chatbot.embeddings import embeddings
from chatbot.config import (
    DOCUMENT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TYPESENSE_CONFIG,
)

def get_raw_docs(file_path=None):

    target_path = file_path if file_path else DOCUMENT_PATH
    file_extension = os.path.splitext(target_path)[1].lower()

    # format decider
    if file_extension == ".txt":
        loader = TextLoader(target_path, encoding="utf-8")
    elif file_extension == ".pdf":
        loader = PyPDFLoader(target_path)
    elif file_extension in [".docx", ".doc"]:
        loader = Docx2txtLoader(target_path)
    else:
        loader = TextLoader(target_path, encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = splitter.split_documents(documents)
    return docs

def create_vectorstore():
    
    docs = get_raw_docs()
    return index_docs_to_typesense(docs)

def index_file_to_typesense(file_path: str):
    docs = get_raw_docs(file_path)
    return index_docs_to_typesense(docs)

def index_docs_to_typesense(docs):
    vectorstore = Typesense.from_documents(
        docs,
        embeddings,
        typesense_client_params={
            "host": TYPESENSE_CONFIG["host"],
            "port": TYPESENSE_CONFIG["port"],
            "protocol": TYPESENSE_CONFIG["protocol"],
            "typesense_api_key": TYPESENSE_CONFIG["api_key"],
            "typesense_collection_name": TYPESENSE_CONFIG["collection_name"],
            "connection_timeout_seconds": 60
        },
    )
    return vectorstore