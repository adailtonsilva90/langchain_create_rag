import os
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel, Field

from app.rag.loaders import extract_text_from_file
from app.rag.vectorstore import add_documents_to_store, embeddings

# Model Configurations from Environment
GEMINI_ANALYSIS_MODEL = os.getenv("GEMINI_ANALYSIS_MODEL", "models/gemini-flash-latest")

# -----------------
# State Schema
# -----------------
class DocumentState(TypedDict, total=False):
    file_bytes: bytes
    filename: str
    raw_text: str
    metadata: Dict[str, Any]
    chunks: List[Document]

# LLM Schema for Semantic Analysis
class DocumentAnalysis(BaseModel):
    category: str = Field(description="The document category in one to three words (e.g., 'Contract', 'Scientific Article', 'Invoice', etc.)")
    summary: str = Field(description="A reliable 1-2 sentence summary of what the document is about.")
    entities: List[str] = Field(description="List of main entities mentioned in the text (companies, people, products).")

# -----------------
# Nodes
# -----------------

def extract_node(state: DocumentState) -> DocumentState:
    print(f"--> [Extract Node] Processing file: {state['filename']}")
    raw_text = extract_text_from_file(state["file_bytes"], state["filename"])
    return {"raw_text": raw_text}

def analysis_node(state: DocumentState) -> DocumentState:
    print("--> [Analysis Node] Analyzing text semantics...")
    text = state["raw_text"]
    if not text.strip():
        print("Warning: Empty text.")
        return {"metadata": {}}
        
    # Use only the first 4000 characters for semantic evaluation to save tokens
    sample_text = text[:4000] 
    
    # Using Gemini for semantic analysis (Configurable via GEMINI_ANALYSIS_MODEL)
    llm = ChatGoogleGenerativeAI(model=GEMINI_ANALYSIS_MODEL, temperature=0)
    
    # In modern langchain-google-genai for Pydantic parsing:
    structured_llm = llm.with_structured_output(DocumentAnalysis)
    
    try:
        analysis: DocumentAnalysis = structured_llm.invoke(
            f"Analyze the following document and extract the requested metadata:\n\n{sample_text}"
        )
        metadata = {
            "source": state["filename"],
            "category": analysis.category,
            "summary": analysis.summary,
            "entities": ", ".join(analysis.entities)
        }
    except Exception as e:
        print(f"Semantic Analysis Failed: {e}")
        metadata = {"source": state["filename"], "category": "Unknown"}
        
    return {"metadata": metadata}

def chunking_node(state: DocumentState) -> DocumentState:
    print("--> [Chunking Node] Creating semantic chunks...")
    text = state.get("raw_text", "")
    metadata = state.get("metadata", {"source": state.get("filename", "unknown")})
    
    if not text.strip():
        return {"chunks": []}

    try:
        # Using SemanticChunker from LangChain Experimental with local Ollama Embeddings
        text_splitter = SemanticChunker(embeddings)
        
        # SemanticChunker will divide sentences using LLM embeddings
        docs = text_splitter.create_documents([text], metadatas=[metadata])
        print(f"Generated {len(docs)} semantic chunks.")
        return {"chunks": docs}
    except Exception as e:
        print(f"Semantic Chunking Failed: {e}. Falling back to single chunk.")
        # Fallback to a single chunk if semantic chunking fails (e.g., Ollama unavailable)
        fallback_doc = Document(page_content=text, metadata=metadata)
        return {"chunks": [fallback_doc]}

def insertion_node(state: DocumentState) -> DocumentState:
    print("--> [Insertion Node] Inserting chunks into pgvector...")
    chunks = state.get("chunks", [])
    if not chunks:
        print("No chunks to insert.")
        return state
        
    try:
        add_documents_to_store(chunks)
        print("Insertion complete.")
    except Exception as e:
        print(f"Insertion Failed: {e}")
        # We don't crash here, so the user can at least get the extraction results
        
    return state

# -----------------
# Graph Builder
# -----------------
workflow = StateGraph(DocumentState)

# Adiciona os nós
workflow.add_node("extract", extract_node)
workflow.add_node("analyze", analysis_node)
workflow.add_node("chunk", chunking_node)
workflow.add_node("insert", insertion_node)

# Define the flow
workflow.set_entry_point("extract")
workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", "chunk")
workflow.add_edge("chunk", "insert")
workflow.add_edge("insert", END)

# Compile the graph
app_graph = workflow.compile()
