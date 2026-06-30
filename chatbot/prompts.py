

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
    "CRITICAL CODE & SYNTAX RULES:\n"
    "- When generating code snippets or syntax explanations, you must copy the exact structures provided in the context VERBATIM.\n"
    "- Do NOT guess, alter, or simplify syntax shorthand (e.g., do not pass raw strings into pipes unless explicitly shown in the text).\n"
    "- Ground all algorithmic explanations strictly in the text. Do not assume models need fine-tuning if the text implies a standard RAG workflow.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)