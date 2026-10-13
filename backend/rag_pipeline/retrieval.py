import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, List
from colpali_engine.models import ColPali, ColPaliProcessor
from rag_pipeline import config

# -------------------------------------------------------------
# GLOBAL RETRIEVAL CACHING INFRASTRUCTURE (SINGLETON PATTERN)
# -------------------------------------------------------------
_RETRIEVAL_MODEL = None
_RETRIEVAL_PROCESSOR = None

def get_cached_retrieval_model():
    """
    Lazy-loads and caches the ColPali model and processor globally.
    Prevents weight re-allocation bottlenecks across search query invocations.
    """
    global _RETRIEVAL_MODEL, _RETRIEVAL_PROCESSOR

    if _RETRIEVAL_MODEL is None:
        print("COLPALI PATH:", config.COLPALI_MODEL_DIR)
        print("ABS PATH:", os.path.abspath(config.COLPALI_MODEL_DIR))
        print("FILES:")
        print(os.listdir(config.COLPALI_MODEL_DIR))
        print(f"\n[INIT] Loading ColPali Retrieval Weights into Engine Memory ({config.DEVICE.upper()})...")
        # Enforce local_files_only to guarantee offline infrastructure operation
        _RETRIEVAL_MODEL = ColPali.from_pretrained(
            config.COLPALI_MODEL_DIR, 
            torch_dtype=config.CURRENT_DTYPE,
            device_map=config.DEVICE, 
            attn_implementation="eager", 
            local_files_only=True
        ).eval()
        
        _RETRIEVAL_PROCESSOR = ColPaliProcessor.from_pretrained(
            config.COLPALI_MODEL_DIR, 
            local_files_only=True
        )
        print("[INIT] ColPali Retrieval Engine ready and cached in active space.")

    return _RETRIEVAL_MODEL, _RETRIEVAL_PROCESSOR

class TwoStageRetriever:
    def __init__(self):
        # Fetch the shared singleton instances instantly
        self.model, self.processor = get_cached_retrieval_model()
        
        # Keep local structural asset caches intact for fast repeated lookups
        self.embedding_cache = {}
        self.chunk_cache = {}

    def search(self, query_text: str, doc_stem_name: str, top_k: int = None) -> Dict[str, Any]:
        # Dynamically read top_k directly from central config if not overridden by node
        if top_k is None:
            top_k = getattr(config, "RETRIEVAL_TOP_K", 2)

        out_dir = Path(config.OUTPUT_DIR)
        emb_path = out_dir / f"{doc_stem_name}_colpali_embeddings.pt"
        json_path = out_dir / f"{doc_stem_name}_production_chunks.json"

        if not emb_path.exists() or not json_path.exists():
            raise FileNotFoundError(
                f"Missing processed assets for document stem '{doc_stem_name}' in output directory."
            )

        # Load file safely using structural configurations 
        if doc_stem_name not in self.embedding_cache:
            self.embedding_cache[doc_stem_name] = torch.load(
                emb_path,
                map_location="cpu",
                weights_only=False
            )

        page_embeddings = self.embedding_cache[doc_stem_name]

        # --- STAGE 1: Visual-First Search ---
        with torch.no_grad():
            print(query_text)
            query_batch = self.processor.process_queries([query_text])
            query_batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in query_batch.items()}
            query_embeddings = self.model(**query_batch)
            
            # Move list elements individually onto runtime hardware to bypass stacking shape errors
            processed_doc_embeddings = [
                emb.to(self.model.device).to(config.CURRENT_DTYPE) for emb in page_embeddings
            ]
            
            # Pass directly as a list of tensors to prevent Multi-Vector shape alignment errors
            scores = self.processor.score_multi_vector(query_embeddings, processed_doc_embeddings)
            raw_scores = scores[0].cpu().float().tolist()

        page_rankings = sorted(
            [{"page_num": idx + 1, "score": score} for idx, score in enumerate(raw_scores)],
            key=lambda x: x["score"], reverse=True
        )
        print(page_rankings)
        # FIX: Pin retrieval tracking down precisely to the top matching page asset from ColPali
        top_pages = page_rankings[:top_k]
        target_page_numbers = [item["page_num"] for item in top_pages]

        print(f"\n[Visual Matches] Top Page: {target_page_numbers[0]} (Score: {top_pages[0]['score']:.4f})")

        # --- STAGE 2: Text Matching Lookup ---
        if doc_stem_name not in self.chunk_cache:
            with open(json_path, "r", encoding="utf-8") as f:
                self.chunk_cache[doc_stem_name] = json.load(f)

        chunks_payload = self.chunk_cache[doc_stem_name]

        # FIX: Restrict text payload purely to the top matching target page context
        primary_target_page = target_page_numbers[0]
        target_pages_set = {primary_target_page}
            
        # Filter chunks to only send the context of that exact page
        retrieved_chunks = [
            chunk for chunk in chunks_payload 
            if chunk.get("page_num") in target_pages_set
        ]
        
        # Validation Logging: Track precisely what is passed into your LangGraph context
        page_distribution = {}
        for chunk in retrieved_chunks:
            p_num = chunk.get("page_num")
            page_distribution[p_num] = page_distribution.get(p_num, 0) + 1
            
        print(f"[Text Stage] Extracted context payload distribution: {page_distribution}")
        
        return {
            "chunks": retrieved_chunks,
            # Return only the precise target page for canvas rendering
            "page_numbers": [primary_target_page]
        }