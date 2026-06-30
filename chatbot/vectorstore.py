"""
Vector Store module.

This module loads documents, splits them into chunks,
creates embeddings, and indexes them into Typesense.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Typesense

from chatbot.embeddings import embeddings
from chatbot.config import (
    DOCUMENT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TYPESENSE_CONFIG
)

# Load the document 
loader = TextLoader(DOCUMENT_PATH)
documents = loader.load()

# Chunking 
text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
docs = text_splitter.split_documents(documents)

# Create Typesense Vector Store
vectorstore = Typesense.from_documents(
    docs,
    embeddings,
    typesense_client_params={
        "host": TYPESENSE_CONFIG["host"],
        "port": TYPESENSE_CONFIG["port"],
        "protocol": TYPESENSE_CONFIG["protocol"],
        "typesense_api_key": TYPESENSE_CONFIG["api_key"],
        "typesense_collection_name": TYPESENSE_CONFIG["collection_name"],
    },
)