"""
Retriever Module

Creates a retriever from the Typesense vector store.
"""

from chatbot.vectorstore import create_vectorstore


def create_retriever(k=3):
    vectorstore = create_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k} ## Numbr of chunks to retrieve 
    )

    return retriever