# pipeline_orchestrator.py
import httpx
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from config import (
    OLLAMA_BASE_URL, VISION_LLM_MODEL, VISION_LLM_TEMPERATURE,
    RETRIEVAL_TOP_K, RERANK_TOP_K, RERANK_THRESHOLD, 
    MAX_TEXT_CONTEXT_CHARS, DEVICE, NORMALIZER_LLM_MODEL,
    CROSS_ENCODER_MODEL_DIR
)
from vector_store import LocalDualTrackStore

class RAGPipeline:
    def __init__(self):
        self.db = LocalDualTrackStore()
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL_DIR, device=DEVICE)
        self.client = httpx.Client(timeout=180.0)

    def _query_ollama(self, model: str, prompt: str, temperature: float, streaming_callback=None) -> str:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "num_ctx": 4096},
            "stream": True if streaming_callback else False
        }
        if not streaming_callback:
            try:
                response = self.client.post(url, json=payload, timeout=180.0)
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    return "".join([json.loads(l).get("message", {}).get("content", "") for l in lines if l]).strip()
            except Exception as e: return f"Execution error: {str(e)}"
        full_response = ""
        with self.client.stream("POST", url, json=payload, timeout=180.0) as r:
            for line in r.iter_lines():
                if line:
                    token = json.loads(line).get("message", {}).get("content", "")
                    if token:
                        full_response += token
                        streaming_callback(token)
        return full_response.strip()

    def verify_grounding_fallback(self, answer: str, context: str) -> Tuple[float, str]:
        prompt = (f"Strict compliance check. JSON: {{\"score\": 0.0-1.0, \"issues\": []}}.\n"
                  f"Context:\n{context}\nStatement:\n{answer}")
        res = self._query_ollama(NORMALIZER_LLM_MODEL, prompt, 0.0)
        try:
            data = json.loads(re.search(r'\{.*\}', res, re.DOTALL).group())
            return float(data.get("score", 1.0)), ", ".join(data.get("issues", []))
        except: return 1.0, ""

    def _extract_chunk_data(self, candidate: Any) -> Dict[str, Any]:
        # Preserved original multi-type extraction logic
        if isinstance(candidate, dict) and isinstance(candidate.get("metadata"), dict):
            meta = candidate["metadata"]
            return {
                "chunk_index": candidate.get("chunk_index", meta.get("chunk_index", 0)),
                "source_file": candidate.get("source_file", meta.get("source_file", "")),
                "track_a_surface": candidate.get("track_a_surface", meta.get("track_a_surface", "")),
                "track_b_markdown": candidate.get("track_b_markdown") or meta.get("track_b_markdown") or candidate.get("page_content", ""),
                "page_number": candidate.get("page_number", meta.get("page_number", 1)),
                "associated_images": candidate.get("associated_images", meta.get("associated_images", [])),
                "dense_score": candidate.get("dense_score", 0.0),
                "sparse_score": candidate.get("sparse_score", 0.0)
            }
        elif hasattr(candidate, "metadata") and isinstance(candidate.metadata, dict):
            meta = candidate.metadata
            return {
                "chunk_index": meta.get("chunk_index", 0), "source_file": meta.get("source_file", ""),
                "track_a_surface": meta.get("track_a_surface", ""), "track_b_markdown": meta.get("track_b_markdown") or getattr(candidate, "page_content", ""),
                "page_number": meta.get("page_number", 1), "associated_images": meta.get("associated_images", []),
                "dense_score": getattr(candidate, "dense_score", 0.0), "sparse_score": getattr(candidate, "sparse_score", 0.0)
            }
        return {"chunk_index": 0, "source_file": "", "track_a_surface": "", "track_b_markdown": str(candidate), "page_number": 1, "associated_images": [], "dense_score": 0.0, "sparse_score": 0.0}

    def answer_question(self, user_question: str, doc_filter: str = None, streaming_callback=None, sources_callback=None) -> str:
        import time
        t_start = time.perf_counter()
        
        t0 = time.perf_counter()
        hits = self.db.similarity_search(user_question, top_k=RETRIEVAL_TOP_K, doc_filter=doc_filter)
        candidates = [self._extract_chunk_data(h) for h in hits]
        t_retrieval = (time.perf_counter() - t0) * 1000
        
        if not candidates: 
            fallback_msg = "I do not have indexed chunks for this document yet. Please ensure the document has been fully parsed and indexed."
            if streaming_callback:
                streaming_callback(fallback_msg)
            return fallback_msg

        # Detect overview/summary intent queries
        overview_keywords = ["about", "summary", "summarize", "overview", "what is this", "main topic", "discuss", "explain the paper", "paper about", "document about"]
        is_overview_query = any(k in user_question.lower() for k in overview_keywords)

        # Rerank with concise surface snippets to keep CPU latency low (~3-5s instead of 50s)
        t1 = time.perf_counter()
        pairs = [[user_question, f"{c['track_a_surface'][:400]}"] for c in candidates[:6]]
        rerank_scores = self.reranker.predict(pairs)
        for idx, score in enumerate(rerank_scores): 
            candidates[idx]["rerank_score"] = float(score)

        sorted_candidates = sorted(candidates[:len(pairs)], key=lambda x: x["rerank_score"], reverse=True)
        t_rerank = (time.perf_counter() - t1) * 1000

        # Calibrated scoring:
        # CrossEncoder raw outputs:
        # - Specific topical queries: score >= 0.05 up to 0.90+
        # - Overview queries ("what is the doc about"): score ~ 0.00015
        # - Negative/unrelated queries ("diabetes medication"): score < 0.00009
        raw_top_score = float(sorted_candidates[0]["rerank_score"])
        raw_second_score = float(sorted_candidates[1]["rerank_score"]) if len(sorted_candidates) > 1 else 0.0
        gap = raw_top_score - raw_second_score

        if raw_top_score >= 0.05 or (raw_top_score >= 0.01 and gap >= 0.005):
            routing_tier = "HIGH"
            confidence_score = min(0.95, max(0.75, raw_top_score))
        elif raw_top_score >= 0.002 or (is_overview_query and raw_top_score > 0.00010):
            routing_tier = "MEDIUM"
            confidence_score = 0.70 if is_overview_query else min(0.68, max(0.40, raw_top_score * 5.0))
        else:
            routing_tier = "LOW"
            confidence_score = max(0.02, min(0.10, raw_top_score * 100.0))

        print(f"[Confidence] Reranker Top: {raw_top_score:.6f} | Second: {raw_second_score:.6f} | Overview: {is_overview_query} | Tier: {routing_tier} | Display: {confidence_score:.2f}")

        # Send structured sources back to client
        # In overview queries, include top overview chunks (e.g. Page 1, Abstract, Introduction)
        winning_chunks = sorted_candidates[:RERANK_TOP_K]
        extracted_sources = []
        seen_snippets = set()

        if routing_tier != "LOW":
            for idx, c in enumerate(winning_chunks):
                snippet_preview = c.get("track_a_surface", "").strip()[:180]
                if not snippet_preview or snippet_preview in seen_snippets:
                    continue
                seen_snippets.add(snippet_preview)

                display_relevance = 0.85 if is_overview_query else min(0.99, max(0.40, c.get("rerank_score", 0.0) * 1.5 if c.get("rerank_score", 0.0) < 0.5 else c.get("rerank_score", 0.0)))
                extracted_sources.append({
                    "page": c.get("page_number", 1),
                    "score": round(display_relevance, 2),
                    "text": c.get("track_a_surface", "")[:280] + "...",
                    "doc_items": c.get("doc_items", []),
                    "source_file": Path(c.get("source_file", "")).name
                })
            
        if sources_callback:
            sources_callback(extracted_sources)

        # Context Extension (Preserving your original neighbor loop)
        seen, final_context_blocks = set(), []
        for chunk in winning_chunks:
            src, curr = chunk["source_file"], chunk["chunk_index"]
            blocks = []
            for n_idx in [curr-1, curr, curr+1]:
                if (src, n_idx) not in seen and n_idx >= 0:
                    raw_n = self.db.get_chunk_by_index(src, n_idx) if n_idx != curr else chunk
                    c = self._extract_chunk_data(raw_n) if raw_n else None
                    if c and c["track_b_markdown"]:
                        blocks.append(c); seen.add((src, n_idx))
            if blocks:
                txt = f"--- {Path(src).name} | Page {chunk['page_number']} ---\n"
                for b in sorted(blocks, key=lambda x: x['chunk_index']): txt += f"{b['track_b_markdown']}\n"
                final_context_blocks.append(txt)
        
        context_string = "\n\n".join(final_context_blocks)[:MAX_TEXT_CONTEXT_CHARS]
        
        # Generation
        system_prompt = f"""
        You are an intelligent offline document assistant. Answer the user question based strictly on the context below.
        Format your response clearly using markdown. Be direct, clear, and concise (1-2 paragraphs).
        If the context does not contain sufficient information, state that clearly without guessing.
        Context:\n{context_string}\n\nQuestion: {user_question}\nAnswer:
        """
        t2 = time.perf_counter()
        answer = self._query_ollama(NORMALIZER_LLM_MODEL, system_prompt, VISION_LLM_TEMPERATURE, streaming_callback=streaming_callback)
        t_gen = (time.perf_counter() - t2) * 1000
        t_total = (time.perf_counter() - t_start) * 1000

        telemetry_info = (
            f"\n\n---\n"
            f"`⚡ Retrieval: {t_retrieval:.0f}ms | Rerank: {t_rerank:.0f}ms | Generation: {t_gen:.0f}ms | Confidence: {routing_tier} ({confidence_score:.2f})`"
        )
        if streaming_callback:
            streaming_callback(telemetry_info)
        
        # Gated Verification
        if routing_tier == "LOW" or any(kw in user_question.lower() for kw in ["patient", "diagnose", "legal"]):
            score, issues = self.verify_grounding_fallback(answer, context_string)
            if score < 0.6:
                warning = f"\n\n> ⚠️ **Low confidence warning (Score: {score:.2f})**: {issues}"
                if streaming_callback: streaming_callback(warning)
                return answer + telemetry_info + warning
                
        return answer + telemetry_info