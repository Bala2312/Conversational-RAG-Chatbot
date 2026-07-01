
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.3-70b-versatile"

TYPESENSE_CONFIG = {
    "host": os.getenv("TYPESENSE_HOST"),
    "port": os.getenv("TYPESENSE_PORT"),
    "protocol": os.getenv("TYPESENSE_PROTOCOL"),
    "api_key": os.getenv("TYPESENSE_API_KEY"),
    "collection_name": os.getenv("TYPESENSE_COLLECTION"),
}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TEMPERATURE = 0
DOCUMENT_PATH = "data/documents/test.txt"
