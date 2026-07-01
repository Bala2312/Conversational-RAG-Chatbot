from chatbot.config import TOP_K
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from chatbot.vectorstore import create_vectorstore, get_raw_docs


def create_retriever(TOP_K):
    vectorstore = create_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K} ## Numbr of chunks to retrieve 
    )

    #BM25 RETREIVER
    raw_docs = get_raw_docs() 
    keyword_retriever = BM25Retriever.from_documents(raw_docs)
    keyword_retriever.k = TOP_K

    # Combine both vector and bm25 retreivers 
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever