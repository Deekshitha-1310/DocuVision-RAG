# Offline Multi-Modal PDF Extraction & Dual-Track RAG

An enterprise-grade, **100% offline and air-gapped** Retrieval-Augmented Generation (RAG) system engineered for high-accuracy document extraction, layout-aware chunking, visual reasoning, and grounded conversational search over complex PDFs (scientific papers, technical specs, financial reports, and multi-column manuals).

---

## Architecture Overview

```
                                      ┌────────────────────────────────────────┐
                                      │              PDF Document              │
                                      └───────────────────┬────────────────────┘
                                                          │
                                     ┌────────────────────┴────────────────────┐
                                     │         Docling Document Parser         │
                                     │ (Layout Analysis, OCR, Table Structure) │
                                     └────────────────────┬────────────────────┘
                                                          │
                        ┌─────────────────────────────────┴─────────────────────────────────┐
                        ▼                                                                   ▼
       ┌─────────────────────────────────┐                                 ┌─────────────────────────────────┐
       │           Text Track            │                                 │          Vision Track           │
       │  (Hybrid Chunking & Hierarchy)  │                                 │ (ColPali Visual Embeddings)     │
       └────────────────┬────────────────┘                                 └────────────────┬────────────────┘
                        │                                                                   │
       ┌────────────────▼────────────────┐                                 ┌────────────────▼────────────────┐
       │       LocalDualTrackStore       │                                 │       Visual Vector Index       │
       │   Dense Vectors + Sparse BM25   │                                 │      Page & Patch Encodings     │
       └────────────────┬────────────────┘                                 └────────────────┬────────────────┘
                        │                                                                   │
                        └─────────────────────────────────┬─────────────────────────────────┘
                                                          │
                                             ┌────────────▼───────────┐
                                             │       User Query       │
                                             └────────────┬───────────┘
                                                          │
                                             ┌────────────▼───────────┐
                                             │    Hybrid Candidate    │
                                             │    Generation (RRF)    │
                                             └────────────┬───────────┘
                                                          │
                                             ┌────────────▼───────────┐
                                             │  Cross-Encoder Rerank  │
                                             │ (BAAI/bge-reranker)    │
                                             └────────────┬───────────┘
                                                          │
                                             ┌────────────▼───────────┐
                                             │   Confidence Routing   │
                                             │  & Context Compression │
                                             └────────────┬───────────┘
                                                          │
                                             ┌────────────▼───────────┐
                                             │   Ollama Local LLM     │
                                             │ (Token Stream via SSE) │
                                             └────────────┬───────────┘
                                                          │
                                             ┌────────────▼───────────┐
                                             │    React 19 + Vite     │
                                             │   Interactive Web UI   │
                                             └────────────────────────┘
```

---

## Key Features

- **100% Air-Gapped & Offline Execution**: Strict offline flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) ensure no network calls or telemetry are triggered during runtime.
- **Dual-Pipeline Processing**:
  - **Text Track**: Extracts hierarchical section headings, structured tables (Markdown), and context breadcrumbs using IBM Docling.
  - **Vision Track**: Employs ColPali vision-language multi-modal embeddings for complex visual layouts (diagrams, flowcharts, infographics).
- **Hybrid Search with Cross-Encoder Reranking**:
  - First-stage candidate generation blends **Dense Semantic Embeddings** (`all-MiniLM-L6-v2`) with **Sparse Keyword Matching** (`BM25`).
  - Second-stage reranking is performed by `BAAI/bge-reranker-large` for pinpoint relevance scoring.
- **Dynamic Confidence Scoring**:
  - Uses top cross-encoder relevance scores and the score gap between the 1st and 2nd candidate to classify certainty (`HIGH`, `MEDIUM`, `LOW`) rather than uncalibrated metric blending.
- **Agentic Verification with LangGraph**:
  - Vision workflows feature deterministic answer validation nodes and guardrail loops to eliminate hallucinated assertions.
- **Real-Time Token Streaming**:
  - FastAPI streaming responses powered by Server-Sent Events (SSE) deliver low-latency progressive rendering to the client.
- **Modern React 19 Frontend**:
  - Fast, responsive user interface with document library management, asset previews (extracted tables and figures), collapsible sidebars, and source chunk inspection.

---

## Tech Stack

| Domain | Technologies |
|---|---|
| **Backend Framework** | FastAPI, Uvicorn, Pydantic, Httpx |
| **Document Processing** | IBM Docling, PyMuPDF, pdfplumber |
| **Embeddings & Search** | SentenceTransformers (`all-MiniLM-L6-v2`), Rank-BM25, Cross-Encoder |
| **Vision & RAG Orchestration**| Byaldi (ColPali), LangChain, LangGraph |
| **Local LLM Runtime** | Ollama (`qwen2.5:3b`, `qwen2.5vl:3b`) |
| **Frontend** | React 19, Vite, Lucide-react, Vanilla CSS |

---

## Directory Structure

```
pdf_extraction_rag/
├── backend/
│   ├── api.py                      # FastAPI server with SSE streaming endpoints
│   ├── engines.py                  # UnifiedPipelineFactory orchestrator
│   ├── sessions.json               # Persisted document library state
│   └── rag_pipeline/
│       ├── config.py               # Hardware, path, and model configurations
│       ├── ingestion_docling.py    # Docling layout parsing & chunk extraction
│       ├── ingestion.py            # Vision pipeline document ingestion
│       ├── pipeline_orchestrator.py# Hybrid search, reranking & Ollama synthesis
│       ├── langgraph_framework.py  # LangGraph vision workflow & guardrails
│       └── vector_store.py         # Dual-track dense vector + BM25 local store
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main React chat and document library UI
│   │   ├── App.css                 # Interface styles and responsive layout
│   │   ├── api.js                  # SSE client and REST API integration
│   │   └── main.jsx                # Application root entrypoint
│   ├── package.json
│   └── vite.config.js
├── models/                         # Local offline model weights (git-ignored)
│   ├── all-MiniLM-L6-v2/           # Text embeddings
│   ├── colpali/                    # ColPali multi-modal weights
│   ├── docling_models/             # Docling layout and TableFormer models
│   ├── local_tokenizer/            # Local fallback tokenizer
│   └── reranker_model/             # BAAI/bge-reranker-large
├── processed_data/                 # Runtime extracted assets & vector stores (git-ignored)
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Prerequisites & Model Preparation

Because this application operates offline, model weights must be present in the `models/` folder.

### 1. Local Models Directory Layout

Ensure your `models/` directory is structured as follows:

```
models/
├── all-MiniLM-L6-v2/
├── colpali/
├── local_tokenizer/
├── reranker_model/
└── docling_models/
    └── model_artifacts/
        ├── layout/
        │   ├── config.json
        │   ├── model.safetensors
        │   └── preprocessor_config.json
        └── tableformer/
            ├── accurate/
            │   ├── tableformer_accurate.safetensors
            │   └── tm_config.json
            └── fast/
                ├── tableformer_fast.safetensors
                └── tm_config.json
```

### 2. Download Models (On an Internet-Connected Machine)

If setting up for the first time, download the weights from Hugging Face:

```bash
# 1. Docling TableFormer and Layout models
huggingface-cli download ds4sd/docling-models --local-dir "models/docling_models" --local-dir-use-symlinks False
huggingface-cli download ds4sd/docling-layout-heron --local-dir "models/docling_models/model_artifacts/layout" --local-dir-use-symlinks False

# 2. Text Tokenizer & Embeddings
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir "models/all-MiniLM-L6-v2"
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir "models/local_tokenizer"

# 3. Cross-Encoder Reranker
huggingface-cli download BAAI/bge-reranker-large --local-dir "models/reranker_model"

# 4. ColPali Vision Model (Optional for vision track)
huggingface-cli download vidore/colpaligemma-3b-pt-448-base --local-dir "models/colpali"
```

### 3. Ollama Models Setup

Ensure Ollama is running locally and pull the target models:

```bash
ollama serve

# In another terminal:
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
```

---

## Installation & Setup

### 1. Backend Setup

1. Open PowerShell / Terminal in the project root:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. *(Optional for GPU acceleration)* Install PyTorch with CUDA support:
   ```powershell
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

### 2. Frontend Setup

1. Navigate to `frontend` and install dependencies:
   ```powershell
   cd frontend
   npm install
   cd ..
   ```

---

## Running the Application

### 1. Launch the Backend API
```powershell
# From project root with venv active:
cd backend
uvicorn api:app --reload --port 8000
```
The FastAPI backend will start at `http://localhost:8000`.

### 2. Launch the React Web Interface
```powershell
# In a separate terminal:
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## How to Use

1. **Upload a PDF Document**:
   - Drag and drop or browse to select a PDF in the web UI.
   - Select your preferred pipeline mode: **Text Track** (Docling layout & tables) or **Vision Track** (ColPali visual).
   - Watch the live SSE progress bar as pages are parsed, tables are structured, and embeddings are indexed.
2. **Select an Active Document**:
   - Pick any previously processed document from the **Document Library** sidebar.
   - View document metrics: total pages, detected tables, extracted images, and indexed chunks.
3. **Ask Questions**:
   - Type queries in the chat box.
   - The system performs hybrid retrieval (Dense + BM25), reranks candidate chunks, and streams the answer token-by-token.
4. **Inspect Sources & Metadata**:
   - Click on the **Retrieved Sources** accordion to view the exact text chunks, similarity scores, page numbers, and preview extracted table/image assets.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/documents` | Returns all available indexed documents and metadata. |
| `POST` | `/upload` | Uploads a PDF and returns a Server-Sent Events (SSE) progress stream. |
| `POST` | `/query` | Runs hybrid retrieval and streams answer tokens via SSE. |
| `GET` | `/extracted_asset/{filename}` | Serves extracted tables and figure image assets. |
| `DELETE` | `/documents/{doc_stem}` | Removes a document and its stored session index. |

---

## License

This project is licensed under the MIT License.