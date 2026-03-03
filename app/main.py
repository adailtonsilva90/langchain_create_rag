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

# Carrega variáveis de ambiente
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="RAG API - Document Intelligence", version="1.0")

# --- Modelos Pydantic ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

# --- Endpoints ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint para realizar o envio de arquivos (PDF, PNG, JPG, DOCX, PPTX).
    Inicia o fluxo do LangGraph para extração, análise semântica e chunking.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Arquivo não contém nome.")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo: {e}")

    # Inicializando o state do LangGraph
    initial_state = {
        "file_bytes": content,
        "filename": file.filename,
        "raw_text": "",
        "metadata": {},
        "chunks": []
    }

    try:
        # Inicia o pipeline de processamento (pode ser demorado para arquivos longos/LLMs remotas)
        # O langgraph atual usa dicts mas alguns métodos podem retornar iterators.
        # Vamos assumir o processamento imperativo pra facilitar aqui:
        final_state = app_graph.invoke(initial_state)
        metadata_gerado = final_state.get("metadata", {})
        total_chunks = len(final_state.get("chunks", []))
        
        return JSONResponse({
            "status": "success",
            "message": "Arquivo ingerido com sucesso!",
            "filename": file.filename,
            "metadata_analisado": metadata_gerado,
            "chunks_inseridos": total_chunks
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante o LangGraph pipeline: {e}")


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Endpoint para realizar perguntas (Question Answering) via RAG nos documentos ingeridos.
    Utiliza o Gemini para gerar as respostas consultando a abstração do pgvector.
    """
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="A 'question' informada não pode ser vazia.")

    # 1. Recupera os documentos semanticamente similares
    retriever = vector_store.as_retriever(search_kwargs={"k": request.top_k})
    
    # 2. Prepara o ChatModel
    # Optamos pelo gemini-1.5-flash pela agilidade. 
    # Pode trocar por pro se o contexto exigido for mais profundo.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 3. Constroi o prompt de sistema especializado pra RAG
    system_prompt = (
        "Você é um assistente prestativo focado em responder perguntas analisando o contexto fornecido abaixo.\n"
        "Use APENAS o contexto para basear sua resposta. Se não souber responder, diga que não há informações.\n\n"
        "Contexto Encontrado (Com Metadados da IA Analisadora):\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Criando a Chain final de RAG e invocando 
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        resultado = rag_chain.invoke({"input": question})
        answer = resultado["answer"]
        context_docs = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in resultado["context"]
        ]

        return {
            "question": question,
            "answer": answer,
            "context_sources": context_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao consultar documentos e gerar resposta: {e}")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "API RAG rodando perfeitamente."}
