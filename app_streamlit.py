import streamlit as st
import uuid
from chatbot.retriever import create_retriever
from chatbot.rag import RAGPipeline

st.set_page_config(page_title="DocuMind RAG Chatbot", page_icon="🤖", layout="centered")

st.title("Conversational RAG Chatbot")
st.subheader("Your AAI Document Assistant")
st.write("Ask questionss baed on your indexed knowledge base.")
st.markdown("---")

# Cache the RAG pipeline instantiation
@st.cache_resource
def initialize_rag():
    with st.spinner("Connecting to knowledge base and initializing LLM..."):
        retriever = create_retriever()
        return RAGPipeline(retriever)

rag = initialize_rag()

# unique session states for the UI
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "Hello! I am ready. Ask me anything about your documents."}
    ]

# chat messages on the screen
for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#   new user input
if user_query := st.chat_input("Type your question here..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.ui_messages.append({"role": "user", "content": user_query})

    #  ragg repsonse 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = rag.ask(user_query, session_id=st.session_state.session_id)
                st.markdown(answer)
                
                # Save respondse
                st.session_state.ui_messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")