import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your pipeline components
from rag_pipeline import config
from rag_pipeline.ingestion import ingest_document
from rag_pipeline.retrieval import TwoStageRetriever

# Empty lifespan since you want total control on demand
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean execution window wrapper."""
    yield  

# Initialize FastAPI App (Merged correctly with lifespan and CORS)
app = FastAPI(
    title="RAG Pipeline Server",
    description="Backend API for Document Ingestion and Two-Stage Retrieval",
    lifespan=lifespan
)

# Enable CORS so your HTML/JS frontend can talk to it locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global holder for the retriever model state
STATE: Dict[str, Any] = {
    "retriever": None,
    "current_doc_stem": None
}

# Define data models for API validation
class QueryRequest(BaseModel):
    query: str

class ChunkResponse(BaseModel):
    page_num: int
    chunk_text: str

class ChatResponse(BaseModel):
    answer: str
    matched_chunks: List[ChunkResponse]


@app.post("/api/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """
    Step 1 Core Logic: Accepts a PDF upload from the drag-and-drop UI,
    saves it to disk, checks for pre-existing index assets, and runs ingestion if needed.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Setup standard systemic directory paths based on your environment configurations
    upload_dir = Path(config.OUTPUT_DIR) / "uploaded_files"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    saved_pdf_path = upload_dir / file.filename
    doc_stem = saved_pdf_path.stem

    # Write the uploaded stream directly to local disk storage
    try:
        with saved_pdf_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {str(e)}")

    # Check for existing index files to bypass ingestion lag where applicable
    expected_emb = Path(config.OUTPUT_DIR) / f"{doc_stem}_colpali_embeddings.pt"
    expected_json = Path(config.OUTPUT_DIR) / f"{doc_stem}_production_chunks.json"

    print(f"Checking for existing index files for: {doc_stem}...")
    if not expected_emb.exists() or not expected_json.exists():
        print("Index not found. Running ingestion pipeline...")
        try:
            # Executes the core heavy chunking extraction strategy
            ingest_document(str(saved_pdf_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline ingestion failed: {str(e)}")
    else:
        print("Indexed assets already found on disk. Skipping ingestion step.")

    # Save tracking reference state for sequential /chat workflow queries
    STATE["current_doc_stem"] = doc_stem

    return {
        "status": "success",
        "message": f"Document '{file.filename}' processed successfully.",
        "doc_stem": doc_stem
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_query(payload: QueryRequest):
    """
    Steps 3 & 4 Core Logic: Executes search on the active target file embeddings,
    extracts contexts, and returns data blocks structured for downstream consumption.
    """
    if not STATE["current_doc_stem"]:
        raise HTTPException(status_code=400, detail="No active indexed document context found. Please upload a PDF first.")

    # LAZY INITIALIZATION: If the retriever isn't loaded yet, build it right now
    if STATE["retriever"] is None:
        print("\n[Lazy Init] First query received or manual trigger caught. Loading ColPali weights into memory...")
        try:
            STATE["retriever"] = TwoStageRetriever()
            print("[Lazy Init] Retriever engine built successfully!")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize retriever model: {str(e)}")

    user_query = payload.query
    doc_stem = STATE["current_doc_stem"]

    print(f"\nUser Query Submitted via endpoint: '{user_query}' for document [{doc_stem}]")

    try:
        # Fire the visual router and pull matching Docling text blocks
        matched_chunks = STATE["retriever"].search(
            query_text=user_query,
            doc_stem_name=doc_stem,
            top_k=2  # Find text blocks strictly mapped to the two best visual pages
        )

        # Map to structured data response format
        formatted_chunks = [
            ChunkResponse(page_num=chunk["page_num"], chunk_text=chunk["chunk_text"])
            for chunk in matched_chunks
        ]

        # Combine matching contexts into an answer block 
        context_block = "\n\n".join([f"[Page {c.page_num}]: {c.chunk_text}" for c in formatted_chunks])
        
        # Simulating standard text execution summary
        ai_generated_answer = (
            f"Retrieved Context from [{doc_stem}]:\n\n"
            f"Based on the extracted text blocks, here are the details gathered:\n"
            f"{context_block[:400]}..."  # Truncated cleanly for clarity
        )

        return ChatResponse(
            answer=ai_generated_answer,
            matched_chunks=formatted_chunks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query retrieval phase broken: {str(e)}")

