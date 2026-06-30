from chatbot.retriever import create_retriever
from chatbot.rag import RAGPipeline

print("Creating retriever...")

retriever = create_retriever()

print("Retriever ready!")

rag = RAGPipeline(retriever)

while True:

    question = input("\nQuestion (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    answer = rag.ask(question)

    print("\nAnswer:\n")
    print(answer)