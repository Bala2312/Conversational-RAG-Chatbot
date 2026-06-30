"""
Prompt Template Module

Contains the prompt template used by the RAG chatbot.
"""

from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Use ONLY the context provided below to answer the user's question.

Instructions:
- Answer only from the provided context.
- If the answer is not found in the context, reply:
  "I don't have enough information to answer that question."
- Do not make up facts.
- Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
)