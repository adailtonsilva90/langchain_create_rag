import os
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel, Field

from app.rag.loaders import extract_text_from_file
from app.rag.vectorstore import add_documents_to_store, embeddings

# -----------------
# State Schema
# -----------------
class DocumentState(TypedDict, total=False):
    file_bytes: bytes
    filename: str
    raw_text: str
    metadata: Dict[str, Any]
    chunks: List[Document]

# -----------------
# LLM Schema para a Análise Semântica
# -----------------
class DocumentAnalysis(BaseModel):
    category: str = Field(description="A categoria com uma a três palavras do documento (ex: 'Contrato', 'Artigo Científico', 'Nota Fiscal', etc)")
    summary: str = Field(description="Um resumo de 1 a 2 frases do que o documento se trata de forma fidedigna.")
    entities: List[str] = Field(description="Lista de entidades principais mencionadas no texto (empresas, pessoas, produtos).")

# -----------------
# Nodes
# -----------------

def extract_node(state: DocumentState) -> DocumentState:
    print(f"--> [Extract Node] Processando arquivo: {state['filename']}")
    raw_text = extract_text_from_file(state["file_bytes"], state["filename"])
    return {"raw_text": raw_text}

def analysis_node(state: DocumentState) -> DocumentState:
    print("--> [Analysis Node] Analisando semântica do texto...")
    text = state["raw_text"]
    if not text.strip():
        print("Aviso: Texto vazio.")
        return {"metadata": {}}
        
    # Usa apenas os primeiros 4000 caracteres para avaliação semântica para economizar tokens
    sample_text = text[:4000] 
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Em langchain-google-genai moderno para Pydantic parsing:
    structured_llm = llm.with_structured_output(DocumentAnalysis)
    
    try:
        analysis: DocumentAnalysis = structured_llm.invoke(
            f"Analise o seguinte documento e extraia os metadados pedidos:\n\n{sample_text}"
        )
        metadata = {
            "source": state["filename"],
            "category": analysis.category,
            "summary": analysis.summary,
            "entities": ", ".join(analysis.entities)
        }
    except Exception as e:
        print(f"Falha na Análise Semântica: {e}")
        metadata = {"source": state["filename"], "category": "Desconhecida"}
        
    return {"metadata": metadata}

def chunking_node(state: DocumentState) -> DocumentState:
    print("--> [Chunking Node] Criando chunks semânticos...")
    text = state["raw_text"]
    metadata = state.get("metadata", {"source": state["filename"]})
    
    if not text.strip():
        return {"chunks": []}

    # Utilizando o SemanticChunker do LangChain Experimental com as Embeddings locais do Ollama
    text_splitter = SemanticChunker(embeddings)
    
    # O SemanticChunker vai dividir as sentenças usando LLM embeddings e pareando-as se fizerem sentido
    docs = text_splitter.create_documents([text], metadatas=[metadata])
    print(f"Foram gerados {len(docs)} chunks semânticos.")
    
    return {"chunks": docs}

def insertion_node(state: DocumentState) -> DocumentState:
    print("--> [Insertion Node] Inserindo chunks no pgvector...")
    chunks = state.get("chunks", [])
    if chunks:
        add_documents_to_store(chunks)
        print("Inserção concluída.")
    else:
        print("Nenhum chunk para inserir.")
        
    return state # Retorna o state intacto ou com confirmação

# -----------------
# Graph Builder
# -----------------
workflow = StateGraph(DocumentState)

# Adiciona os nós
workflow.add_node("extract", extract_node)
workflow.add_node("analyze", analysis_node)
workflow.add_node("chunk", chunking_node)
workflow.add_node("insert", insertion_node)

# Define o fluxo
workflow.set_entry_point("extract")
workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", "chunk")
workflow.add_edge("chunk", "insert")
workflow.add_edge("insert", END)

# Compila o grafo
app_graph = workflow.compile()
