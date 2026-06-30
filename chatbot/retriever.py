"""
Retriever Module

Creates a retriever from the Typesense vector store.
"""

from chatbot.vectorstore import create_vectorstore
from chatbot.config import TOP_K


def create_retriever(TOP_K):
    vectorstore = create_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K} ## Numbr of chunks to retrieve 
    )

    return retriever