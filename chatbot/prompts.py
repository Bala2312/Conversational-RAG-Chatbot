

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#Prompt to rewrite the user's query using chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# answer the question using the retrieved context
qa_system_prompt = (
    "You are a helpful AI assistant.\n"
    "Use ONLY the context provided below to answer the user's question.\n"
    "If the answer is not found in the context, reply: 'I don't have enough information to answer that question.'\n"
    "Do not make up facts.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)