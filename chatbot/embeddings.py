"""
Embedding model initialization.

This module initializes the HuggingFace embedding model
used to convert documents and user queries into vector
representations for semantic search.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from chatbot.config import EMBEDDING_MODEL

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)