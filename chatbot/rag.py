from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from chatbot.config import GROQ_API_KEY, MODEL_NAME
from chatbot.prompts import RAG_PROMPT


class RAGPipeline:

    def __init__(self, retriever):
        self.retriever = retriever

        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME,
            temperature=0,
        )

        self.chain = RAG_PROMPT | self.llm | StrOutputParser()

    def ask(self, question: str):

        docs = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        answer = self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return answer