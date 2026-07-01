import streamlit as st
import uuid
import time
from chatbot.retriever import create_retriever
from chatbot.rag import RAGPipeline
from chatbot.config import TOP_K
import streamlit as st
import uuid
import tempfile
import os
from chatbot.vectorstore import index_file_to_typesense



st.set_page_config(page_title="DocuMind RAG Chatbot", layout="wide")

st.title("Conversational RAG Chatbot")
st.caption("Upload documents in the sidebar and chat with them in real-time.")

def display_source_citations(sources):
    if sources:
        with st.expander("Verified Source Citations"):
            for idx, doc in enumerate(sources):
                # Extract filename if available in the metadata path
                raw_path = doc.metadata.get('source', 'Unknown Document')
                filename = os.path.basename(raw_path)
                
                st.markdown(f"**Source Document {idx + 1}:** `{filename}`")
                st.caption(f"\"{doc.page_content.strip()}\"")
                if idx < len(sources) - 1:
                    st.markdown("---")

# sidebar for doc upload 
with st.sidebar:
    st.header("Document Management")
    st.write("Add new files to your vector knowledge base.")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["txt", "pdf", "docx"], 
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        if st.button("Process & Index Document", use_container_width=True):
            file_name = uploaded_file.name
            file_extension = os.path.splitext(file_name)[1].lower()
            
            with st.spinner(f"Splitting and indexing '{file_name}'..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    index_file_to_typesense(temp_file_path)
                    os.remove(temp_file_path)
                    
                    # Clear out old pipeline states so it initializes with the newly added chunks
                    if "rag_pipeline" in st.session_state:
                        del st.session_state["rag_pipeline"]
                    
                    st.success(f"Successfully added '{file_name}' to knowledge base!")
                except Exception as e:
                    st.error(f"Failed to process file: {e}")

#   new and unique session key or ids
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "Hello! I have loaded your knowledge base. Ask away, or upload a new file!"}
    ]

#lazy-load or rebuild the RAG pipeline inside session state
if "rag_pipeline" not in st.session_state:
    retriever = create_retriever(TOP_K)
    st.session_state.rag_pipeline = RAGPipeline(retriever)

rag = st.session_state.rag_pipeline

# Render historical interface items on-screen
for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            display_source_citations(message["sources"])

#Capture live user interaction input
if user_query := st.chat_input("Ask a question about your documents..."):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.ui_messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag.ask(user_query, session_id=st.session_state.session_id)
                answer = result["answer"]
                sources = result["sources"]
                st.markdown(answer)
                display_source_citations(sources)
                
                # Save data package into the historical state structure
                st.session_state.ui_messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"An error occurred while generating an answer: {e}")