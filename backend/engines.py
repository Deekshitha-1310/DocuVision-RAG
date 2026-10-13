#The UnifiedPipelineFactory has been refactored to align with your newly integrated multi-query expansion, hybrid RRF search, 
# neighborhood sliding windows, and real-time streaming components. It also extracts true visual asset counts from the document_metadata.json
#  produced during ingestion to return exact data metrics back to the client interface.

# engines.py
import json
import asyncio
import queue
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_pipeline'))

# Import Vision Pipeline Modules
from rag_pipeline.ingestion import ingest_document as vision_ingest_document
from rag_pipeline.langgraph_framework import orchestration_engine as vision_orchestration_engine

# Import Text Pipeline Modules
from rag_pipeline import ingestion_docling as text_ingestion
from rag_pipeline.vector_store import LocalDualTrackStore
from rag_pipeline.pipeline_orchestrator import RAGPipeline

class UnifiedPipelineFactory:
    def __init__(self):
        # LocalDualTrackStore handles its own internal lifecycle & embedding matrix loading
        self.text_store = LocalDualTrackStore()
        self.text_pipeline = RAGPipeline()
        # Ensure text_pipeline queries against the shared live in-memory store
        self.text_pipeline.db = self.text_store
        self.vision_engine = vision_orchestration_engine

    def run_text_ingestion_sync(self, target_path: str, progress_queue: queue.Queue):
        """Runs Docling parsing, chunk construction, and indices vector loading."""
        try:
            progress_queue.put(("progress", 15, "Parsing document structural layout..."))
            # Extracts elements and matches assets via internal loops
            rag_chunks = text_ingestion.run_split_and_store_ingestion(target_path)
            
            progress_queue.put(("progress", 50, "Generating token chunk matrix spaces..."))
            
            progress_queue.put(("progress", 80, "Calculating embeddings and indexing store..."))
            # Injects chunks directly; add_documents internally runs self.save()
            self.text_store.add_documents(rag_chunks)
            
            # Read structural asset metrics directly from the docling output metadata registry
            total_pages = 0
            table_count = 0
            image_count = 0
            
            if rag_chunks:
                total_pages = max([c.get("page_number", 1) for c in rag_chunks])
                input_path = Path(target_path)
                metadata_json_path = text_ingestion.BASE_OUTPUT_DIR / f"{input_path.stem}_output" / "document_metadata.json"
                
                if metadata_json_path.exists():
                    try:
                        with open(metadata_json_path, "r", encoding="utf-8") as f:
                            meta_registry = json.load(f)
                            assets = meta_registry.get("extracted_assets", [])
                            table_count = sum(1 for a in assets if a.get("type") == "table")
                            image_count = sum(1 for a in assets if a.get("type") == "image")
                    except Exception:
                        pass  # Fail gracefully back to 0 counts if I/O fails

            metadata = {
                "pages": total_pages,
                "tables": table_count,
                "images": image_count,
                "chunks": len(rag_chunks)
            }
            progress_queue.put(("done", metadata))
        except Exception as e:
            progress_queue.put(("error", str(e)))

    def answer_text_query_sync(self, query: str, stream_queue: queue.Queue, doc_stem: str = None):
        """Runs the synchronous RAG pipeline query execution layer."""
        try:
            print(f"BEFORE answer_question (doc_stem: {doc_stem})")

            stream_queue.put(("status", 40, "Querying Multi-Track Vector Space, Sparse BM25 & Reranker..."))

            # Sources are contextually generated inside the new orchestrator, passing blank to clean front-end UI initialization
            stream_queue.put(("sources", []))

            # Dispatches directly into the updated text pipeline with tokenized lambda streaming
            self.text_pipeline.answer_question(
                query,
                doc_filter=doc_stem,
                streaming_callback=lambda token: stream_queue.put(("token", token)),
                sources_callback=lambda sources: stream_queue.put(("sources", sources))
            )
            print("AFTER answer_question")

            stream_queue.put(("done", None))
            print("DONE pushed from engines")

        except Exception as e:
            stream_queue.put(("error", str(e)))

pipeline_factory = UnifiedPipelineFactory()