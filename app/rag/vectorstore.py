import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from typing import List

# Database Configurations
PG_USER = os.getenv("POSTGRES_USER")
PG_PASS = os.getenv("POSTGRES_PASSWORD")
PG_HOST = os.getenv("POSTGRES_HOST")
PG_PORT = os.getenv("POSTGRES_PORT")
PG_DB = os.getenv("POSTGRES_DB")

# Validate required variables
required_vars = {
    "POSTGRES_USER": PG_USER,
    "POSTGRES_PASSWORD": PG_PASS,
    "POSTGRES_HOST": PG_HOST,
    "POSTGRES_PORT": PG_PORT,
    "POSTGRES_DB": PG_DB,
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
}
missing = [k for k, v in required_vars.items() if not v]
if missing:
    raise ValueError(f"Missing required environment variables in .env: {', '.join(missing)}")

CONNECTION_STRING = f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
COLLECTION_NAME = "rag_documents"

# Configure Google Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
)

# Initialize and connect to VectorStore
vector_store = PGVector(
    connection=CONNECTION_STRING,
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    use_jsonb=True,
    pre_delete_collection=False
)

def add_documents_to_store(docs: List[Document]):
    """Persist documents with Gemini embeddings into pgvector."""
    vector_store.add_documents(docs)

def similarity_search(query: str, k: int = 4) -> List[Document]:
    """Perform semantic similarity search in the vector database."""
    return vector_store.similarity_search(query, k=k)
