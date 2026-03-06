from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from app.rag.graph import app_graph
from app.rag.vectorstore import vector_store
from app.rag.loaders import extract_semantic_content, extract_raw_text

app = FastAPI(title="RAG API - Document Intelligence", version="1.1")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 10

# --- Endpoints ---

@app.post("/text-extraction")
async def test_extraction(file: UploadFile = File(...)):
    """
    Endpoint to test how Gemini extracts raw text from a document.
    """
    try:
        content = await file.read()
        raw_text = extract_raw_text(content, file.filename)
        return {"filename": file.filename, "raw_text": raw_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload files.
    Uses Gemini for multimodal extraction and semantic chunking.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name.")

    try:
        content = await file.read()
        initial_state = {
            "file_bytes": content,
            "filename": file.filename,
            "semantic_data": {},
            "chunks": []
        }
        
        final_state = app_graph.invoke(initial_state)
        total_chunks = len(final_state.get("chunks", []))
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_inserted": total_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Endpoint for RAG QA. Uses Gemini Embeddings and Gemini Pro/Flash for generation.
    """
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    retriever = vector_store.as_retriever(search_kwargs={"k": request.top_k})
    
    query_model = os.getenv("GEMINI_QUERY_MODEL", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model=query_model, temperature=0)

    system_prompt = (
        "Você é um assistente prestativo. Responda APENAS com base no contexto abaixo.\n"
        "Se não souber, diga que não encontrou a informação.\n\n"
        "Contexto:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        result = rag_chain.invoke({"input": question})
        return {
            "question": question,
            "answer": result["answer"],
            "sources": [{"content": d.page_content, "metadata": d.metadata} for d in result["context"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API simplified and running with Gemini."}
