# RAG API Project

This is a Python API (FastAPI) focused on **Retrieval-Augmented Generation (RAG)** with semantic chunking based on LangGraph, using Google Gemini and pgvector.

## Prerequisites
- Python 3.10+
- Docker and Docker Compose

## How to Run Locally

1. **Configure Environment Variables**:
   Create a `.env` file based on the `.env_template`:
   ```bash
   cp .env_template .env
   ```
   Open the generated `.env` and fill in the required variables (Google API Key, database credentials, etc.).

2. **Start the Infrastructure**:
   This will start PostgreSQL (with pgvector), pgAdmin, and Ollama (with automatic model download):
   ```bash
   docker-compose up -d
   ```
   *Note: On the first run, the `ollama-pull-models` container will download the `nomic-embed-text` model. You can monitor progress with `docker logs -f rag_ollama_puller`.*

3. **Install Dependencies**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the API**:
   ```bash
   uvicorn app.main:app --reload
   ```

5. **API Documentation (Swagger)**:
   Available at: `http://localhost:8000/docs`.

## Database Management (pgAdmin)
Available at: `http://localhost:5050`
- **Login**: (Use the email/password defined in your `.env`)
- **Server Connection**:
  - **Host**: `pgvector`
  - **Port**: `5432`
  - **Username/Password**: (Use credentials from your `.env`)

## Endpoints
- `POST /upload`: Upload PDF, DOCX, or Images for semantic ingestion.
- `POST /query`: Query the ingested documents using natural language.
- `GET /`: Health check.

## Testing
To verify the installation, you can run a simple health check or use the Swagger UI to test the endpoints.
- **Manual Test**: Upload a document and check the terminal logs for the English progress messages:
  `--> [Extract Node] Processing file...`
