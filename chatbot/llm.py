"""
LLM initialization module.

This module creates and exports the Groq-hosted Llama model
used throughout the chatbot application.
"""

from langchain_groq import ChatGroq
from chatbot.config import (
    GROQ_API_KEY,
    MODEL_NAME,
    TEMPERATURE
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE
)