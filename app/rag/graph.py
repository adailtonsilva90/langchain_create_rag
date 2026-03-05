import os
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.rag.loaders import extract_semantic_content
from app.rag.vectorstore import add_documents_to_store

# -----------------
# State Schema
# -----------------
class DocumentState(TypedDict, total=False):
    file_bytes: bytes
    filename: str
    semantic_data: Dict[str, Any]  # {raw_text, sections: [{category, content}]}
    chunks: List[Document]

# -----------------
# Nodes
# -----------------

def extract_node(state: DocumentState) -> DocumentState:
    print(f"--> [Extract Node] Multimodal processing: {state['filename']}")
    semantic_data = extract_semantic_content(state["file_bytes"], state["filename"])
    return {"semantic_data": semantic_data}

def chunking_node(state: DocumentState) -> DocumentState:
    print("--> [Chunking Node] Creating chunks from semantic sections...")
    semantic_data = state.get("semantic_data", {})
    sections = semantic_data.get("sections", [])
    filename = state.get("filename", "unknown")
    
    if not sections:
        print("Warning: No semantic sections found. Falling back to raw text.")
        raw_text = semantic_data.get("raw_text", "")
        if not raw_text: return {"chunks": []}
        return {"chunks": [Document(page_content=raw_text, metadata={"source": filename, "category": "General"})]}

    docs = []
    for section in sections:
        docs.append(Document(
            page_content=section.get("content", ""),
            metadata={
                "source": filename,
                "category": section.get("category", "General")
            }
        ))
    
    print(f"Generated {len(docs)} semantic chunks.")
    return {"chunks": docs}

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
        
    return state

# -----------------
# Graph Builder
# -----------------
workflow = StateGraph(DocumentState)

workflow.add_node("extract", extract_node)
workflow.add_node("chunk", chunking_node)
workflow.add_node("insert", insertion_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "chunk")
workflow.add_edge("chunk", "insert")
workflow.add_edge("insert", END)

app_graph = workflow.compile()
