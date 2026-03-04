from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.graph import app_graph
from app.rag.vectorstore import vector_store

app = FastAPI(title="RAG API - Document Intelligence", version="1.0")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

# --- Endpoints ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload files (PDF, PNG, JPG, DOCX, PPTX).
    Starts the LangGraph workflow for extraction, semantic analysis, and chunking.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name.")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Initializing LangGraph state
    initial_state = {
        "file_bytes": content,
        "filename": file.filename,
        "raw_text": "",
        "metadata": {},
        "chunks": []
    }

    try:
        # Start processing pipeline
        final_state = app_graph.invoke(initial_state)
        print(f"DEBUG: final_state keys: {final_state.keys()}")
        generated_metadata = final_state.get("metadata", {})
        chunks = final_state.get("chunks", [])
        print(f"DEBUG: final_state chunks count: {len(chunks)}")
        total_chunks = len(chunks)
        
        return JSONResponse({
            "status": "success",
            "message": "File ingested successfully!",
            "filename": file.filename,
            "analyzed_metadata": generated_metadata,
            "chunks_inserted": total_chunks
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LangGraph pipeline: {e}")


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Endpoint for Question Answering (QA) via RAG on ingested documents.
    Uses Gemini to generate responses by querying the pgvector abstraction.
    """
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="The provided 'question' cannot be empty.")

    # 1. Retrieve semantically similar documents
    retriever = vector_store.as_retriever(search_kwargs={"k": request.top_k})
    
    # 2. Prepare the ChatModel (Configurable via GEMINI_QUERY_MODEL)
    query_model = os.getenv("GEMINI_QUERY_MODEL", "models/gemini-flash-latest")
    llm = ChatGoogleGenerativeAI(model=query_model, temperature=0)

    # 3. Build a specialized system prompt for RAG
    system_prompt = (
        "You are a helpful assistant focused on answering questions by analyzing the context provided below.\n"
        "Use ONLY the context to base your answer. If you don't know the answer, say there is no information.\n\n"
        "Found Context (With AI Analyzer Metadata):\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Creating the final RAG chain and invoking it
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        result = rag_chain.invoke({"input": question})
        answer = result["answer"]
        context_docs = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in result["context"]
        ]

        return {
            "question": question,
            "answer": answer,
            "context_sources": context_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query documents and generate response: {e}")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API running perfectly."}
