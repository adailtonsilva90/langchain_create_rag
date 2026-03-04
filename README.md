# RAG API Project

Este é um projeto de API em Python (via FastAPI) focado em **Retrieval-Augmented Generation (RAG)** com chunking semântico baseado em LangGraph, utilizando Gemini e pgvector.

## Pré-requisitos
- Python 3.10+
- Docker e Docker Compose (para o pgvector)
- [Ollama](https://ollama.com/) rodando localmente (para geração de embeddings usando modelos `nomic-embed-text` ou `mxbai-embed-large`).

## Como rodar localmente

1. **Configurar variáveis de ambiente**:
   Gere o arquivo `.`env baseado no `.env_template`:
   ```bash
   cp .env_template .env
   ```
   Abra o `.env` gerado e preencha as variáveis de ambiente necessárias com os seus dados locais (chaves de API, credenciais de banco e URLs de serviços). Siga os comentários disponíveis no próprio `.env_template` como guia.

2. **Subir o banco de dados (pgvector)**:
   ```bash
   docker-compose up -d
   ```

3. **Instalar Dependências**:
   Recomenda-se criar um ambiente virtual antes (Ex: `python -m venv .venv` e `source .venv/bin/activate`).
   ```bash
   pip install -r requirements.txt
   ```

4. **Rodar a API**:
   ```bash
   uvicorn app.main:app --reload
   ```

5. O Swagger da API estará disponível em: `http://localhost:8000/docs`.
