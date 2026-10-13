# vector_store.py
#ptimizations implemented: Standardized on BGE dense model via the absolute path. Retained pure track_b_markdown embeddings for optimal vector alignment. 
#Upgraded the BM25 tokenizer from standard white space splitting to regex alphanumerical word extraction (re.findall(r"\w+", text)), drastically reducing noisy terms and punctuation impacts.

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from config import DEVICE, VECTOR_INDEX_STORAGE

class LocalDualTrackStore:
    def __init__(self):
        from config import EMBEDDING_MODEL_DIR
        LOCAL_EMBEDDING_PATH = Path(EMBEDDING_MODEL_DIR)

        self.embedding_model = SentenceTransformer(
            str(LOCAL_EMBEDDING_PATH.resolve()),
            device=DEVICE,
            local_files_only=True
        )
        
        
        self.chunks: List[Dict[str, Any]] = []
        self.dense_embeddings: List[np.ndarray] = []
        self.bm25 = None
        
        # Load indices from cold storage if they exist
        self.load_store_indices()

    def _tokenize_text(self, text: str) -> List[str]:
        """Performs robust regex-driven tokenization to improve BM25 matching."""
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def add_documents(self, processed_chunks: List[Dict[str, Any]]):
        """Appends layout-aware chunks to storage."""
        if not processed_chunks:
            return

        # Embed track_b_markdown directly to maximize dual-track embedding quality
        texts_to_embed = [c["track_b_markdown"] for c in processed_chunks]
        new_embeddings = self.embedding_model.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)

        # Deduplicate chunks: replace existing chunks with matching source_file and chunk_index
        existing_keys = {(c["source_file"], c["chunk_index"]): i for i, c in enumerate(self.chunks)}
        
        for chunk, emb in zip(processed_chunks, new_embeddings):
            key = (chunk["source_file"], chunk["chunk_index"])
            if key in existing_keys:
                idx = existing_keys[key]
                self.chunks[idx] = chunk
                self.dense_embeddings[idx] = emb
            else:
                existing_keys[key] = len(self.chunks)
                self.chunks.append(chunk)
                self.dense_embeddings.append(emb)

        # Re-initialize Sparse Exact Keyword Index using regex tokenization
        tokenized_corpus = [self._tokenize_text(c["track_b_markdown"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.save_store_indices()

    def similarity_search(self, query: str, top_k: int = 8, doc_filter: str = None) -> List[Dict[str, Any]]:
        """Executes dual-track retrieval and attaches raw mathematical telemetry scores."""
        if not self.chunks:
            return []

        # Track A: Dense Semantic Matching Loop
        query_emb = self.embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        dense_scores = np.dot(self.dense_embeddings, query_emb)

        # Track B: Sparse BM25 Keyword Matching Loop
        tokenized_query = self._tokenize_text(query)
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Build comprehensive candidates with cross-track metadata payloads
        hit_records = []
        for idx, chunk in enumerate(self.chunks):
            src_file = chunk.get("source_file", "")
            if doc_filter and doc_filter.lower() not in Path(src_file).name.lower():
                continue

            hit_records.append({
                "metadata": {
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    
                    "track_a_surface": chunk["track_a_surface"],
                    "track_b_markdown": chunk["track_b_markdown"],
                    "page_number": chunk["page_number"],
                    "associated_images": chunk["associated_images"]
                },
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "chunk_index": chunk["chunk_index"],
                "source_file": chunk["source_file"],
                "track_a_surface": chunk["track_a_surface"],
                "track_b_markdown": chunk["track_b_markdown"],
                "page_number": chunk["page_number"],
                "associated_images": chunk["associated_images"]
            })

        # Union sorting strategy: Pull best performers across both paradigms
        top_dense = sorted(hit_records, key=lambda x: x["dense_score"], reverse=True)[:top_k]
        top_sparse = sorted(hit_records, key=lambda x: x["sparse_score"], reverse=True)[:top_k]

        # Deduplicate while preserving internal ranking properties
        seen_keys = set()
        unified_hits = []
        for hit in (top_dense + top_sparse):
            uid = (hit["source_file"], hit["chunk_index"])
            if uid not in seen_keys:
                seen_keys.add(uid)
                unified_hits.append(hit)

        return unified_hits[:top_k]

    def get_chunk_by_index(self, source_file: str, chunk_index: int) -> Dict[str, Any] | None:
        """Retrieves exact structural targets for sliding window neighborhood expansions."""
        for chunk in self.chunks:
            if chunk["source_file"] == source_file and chunk["chunk_index"] == chunk_index:
                return chunk
        return None

    def save_store_indices(self):
        """Persists indices to target destination directory blocks."""
        out_dir = Path(VECTOR_INDEX_STORAGE)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_dir / "chunks_db.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            
        if len(self.dense_embeddings) > 0:
            np.save(str(out_dir / "dense_vectors.npy"), np.array(self.dense_embeddings))

    def load_store_indices(self):
        """Loads and builds vector state matrices on boot."""
        idx_dir = Path(VECTOR_INDEX_STORAGE)
        chunks_file = idx_dir / "chunks_db.json"
        vectors_file = idx_dir / "dense_vectors.npy"

        if chunks_file.exists() and vectors_file.exists():
            try:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.dense_embeddings = list(np.load(str(vectors_file)))
                
                tokenized_corpus = [self._tokenize_text(c["track_b_markdown"]) for c in self.chunks]
                self.bm25 = BM25Okapi(tokenized_corpus)
                print(f"[Storage Engine] Local database online. Ingested records: {len(self.chunks)}")
            except Exception as e:
                print(f"[Storage Engine] Malformed cache index state encountered. Re-initializing: {e}")