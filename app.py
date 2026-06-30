from chatbot.retriever import create_retriever
from chatbot.rag import RAGPipeline
from chatbot.config import TOP_K

print("Creating retriever...")
retriever = create_retriever(TOP_K)
print("Retriever ready!")

rag = RAGPipeline(retriever)

# hardcoded session ID for the CLI loop
session_id = "user_cli_session"

while True:
    question = input("\nQuestion (type 'exit' to quit): ")
    
    if question.lower() == "exit":
        break
        
    # Pass the session_id to maintain memory
    answer = rag.ask(question, session_id=session_id)
    
    print("\nAnswer:\n")
    print(answer)