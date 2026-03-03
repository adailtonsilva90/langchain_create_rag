import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from typing import List

# Configurações do Banco
PG_USER = os.getenv("POSTGRES_USER", "admin")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "admin123")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "ragdb")

CONNECTION_STRING = f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
COLLECTION_NAME = "rag_documents"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Configurando o modelo de Embeddings via Ollama (sugestões: nomic-embed-text ou mxbai-embed-large)
# O modelo precisa estar previamente baixado no servidor Ollama (`ollama pull nomic-embed-text`)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

# Inicializando e conectando ao VectorStore
vector_store = PGVector(
    connection=CONNECTION_STRING,
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    use_jsonb=True,
    pre_delete_collection=False
)

def add_documents_to_store(docs: List[Document]):
    """
    Persiste os documentos com seus embeddings no pgvector.
    """
    vector_store.add_documents(docs)

def similarity_search(query: str, k: int = 4) -> List[Document]:
    """
    Efetua a busca por similaridade semântica no banco vetorial.
    """
    return vector_store.similarity_search(query, k=k)
