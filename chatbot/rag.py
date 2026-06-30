from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from chatbot.config import GROQ_API_KEY, MODEL_NAME
from chatbot.prompts import contextualize_q_prompt, qa_prompt
from chatbot.memory import get_session_history

class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME,
            temperature=0,
        )

        # Create a retriever that understands chat history
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        #  the chain that answers the question using retrieved docs
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm, qa_prompt
        )

        # single RAG chain
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )

        # Wrap the chain in a history manager
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def ask(self, question: str, session_id: str = "default_session"):
        # We pass a session_id so the bot knows which conversation history to load
        response = self.conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        
        return response["answer"]