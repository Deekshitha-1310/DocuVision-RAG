# main.py

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from ingestion_docling import run_split_and_store_ingestion
from vector_store import LocalDualTrackStore
from pipeline_orchestrator import RAGPipeline

def main():
    # --- 1. RUN PARSING & STRATEGY INDEXING ---
    # Ingest the PDF, crop images/tables, and build the metadata registry
    print("Starting document ingestion and layout extraction...")
    chunks_payload = run_split_and_store_ingestion(r"C:\Users\Admin\Downloads\Brain_Computer_Interface_based_EEG_for_Emotion_Recognition_System_A_Systematic_Review_removed.pdf")
    
    # Store the parsed chunks directly into the local vector database
    print("\nIndexing chunks into the local vector store...")
    store = LocalDualTrackStore()
    store.add_documents(chunks_payload)
    
    # --- 2. EXECUTE THE LIVE RETRIEVAL RUN ---
    # Query the engine. The orchestrator rewrites the query, finds the best 
    # text chunks, attaches the stored image paths via metadata, and runs it through Ollama.
    print("\nExecuting RAG retrieval pipeline...")
    rag_system = RAGPipeline()
    answer = rag_system.answer_question("explain Electromagnetic Imaging Techniques")
    
    print("\n--- FINAL OUTPUT ANSWER ---")
    print(answer)

if __name__ == "__main__":
    main()