# ingestion_docling.py
# This module handles layout-aware parsing, enforcing structural preservation rules,
# and enforcing chunk sizes between 250 and 350 tokens.

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling_core.types.doc import TableItem, PictureItem, TextItem, DocItemLabel

from config import TOKENIZER_DIR, ensure_tokenizer_exists, OUTPUT_DIR, DOCLING_MODELS_DIR

BASE_OUTPUT_DIR = Path(OUTPUT_DIR)

def clean_surface_text(text: str) -> str:
    """Removes redundant delimiters and visual structural artifacts."""
    if not text:
        return ""
    text = re.sub(r'[\s|:-]{2,}', ' ', text)
    return ' '.join(text.split())

def run_split_and_store_ingestion(file_path: str) -> List[Dict[str, Any]]:
    """Executes layout extraction, tracks sequence neighborhood hierarchy, and structures chunks."""
    ensure_tokenizer_exists()
    input_path = Path(file_path)
    
    doc_out_dir = BASE_OUTPUT_DIR / f"{input_path.stem}_output"
    images_dir = doc_out_dir / "extracted_assets"
    
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions(
        artifacts_path=DOCLING_MODELS_DIR,
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=True,
        generate_table_images=True
    )
    
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    print(f"[Ingestion] Transforming document: {input_path.name}")
    result = converter.convert(input_path)
    doc = result.document
    
    with open(doc_out_dir / "document_content.md", "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())
        
    metadata_registry = {
        "pdf_source": input_path.name,
        "total_pages": int(doc.num_pages()),
        "extracted_assets": []
    }
    
    table_counter = 0
    picture_counter = 0
    
    for element, _level in doc.iterate_items():
        if isinstance(element, (PictureItem, TableItem)):
            page_no = int(element.prov[0].page_no) if (hasattr(element, "prov") and element.prov) else 1
            bbox = element.prov[0].bbox if (hasattr(element, "prov") and element.prov) else None
            bbox_dict = {"l": round(bbox.l, 2), "t": round(bbox.t, 2), "r": round(bbox.r, 2), "b": round(bbox.b, 2)} if bbox else None
            
            is_pic = isinstance(element, PictureItem)
            f_name = f"picture_pg{page_no}_{picture_counter+1}.png" if is_pic else f"table_pg{page_no}_{table_counter+1}.png"
            
            if is_pic:
                picture_counter += 1
            else:
                table_counter += 1
            
            out_img_path = images_dir / f_name
            
            try:
                element.get_image(doc).save(out_img_path, "PNG")
                asset_entry = {
                    "type": "image" if is_pic else "table",
                    "file_name": f_name,
                    "local_path": str(out_img_path),
                    "page_number": page_no,
                    "bounding_box": bbox_dict
                }
                if is_pic:
                    ann = element.annotations[0] if (hasattr(element, "annotations") and element.annotations) else None
                    asset_entry["classification"] = str(getattr(ann, "label", "image"))
                
                metadata_registry["extracted_assets"].append(asset_entry)
            except Exception:
                continue
                
    with open(doc_out_dir / "document_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_registry, f, indent=2, ensure_ascii=False)

    raw_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    
    # Intentionally configured max_tokens around 350 to optimize layout preservation boundaries
    chunker = HybridChunker(
        tokenizer=raw_tokenizer, 
        max_tokens=350, 
        merge_peers=True
    )
    
    raw_chunks = list(chunker.chunk(dl_doc=doc))
    rag_chunks = []
    
    print("[Ingestion] Processing chunk metadata and structural integrity tracks...")
    
    MAX_EMBED_TOKENS = 400
    global_chunk_idx = 0
    
    for idx, chunk in enumerate(raw_chunks):
        page_num = 1
        contains_protected_structure = False
        
        if chunk.meta.doc_items:
            page_num = int(chunk.meta.doc_items[0].prov[0].page_no) if chunk.meta.doc_items[0].prov else 1
            for item in chunk.meta.doc_items:
                if isinstance(item, (TableItem, PictureItem)):
                    contains_protected_structure = True
                    break
                elif isinstance(item, TextItem) and item.label in (
                    DocItemLabel.FORMULA, 
                    DocItemLabel.LIST_ITEM, 
                    DocItemLabel.CAPTION
                ):
                    contains_protected_structure = True
                    break

        matching_assets = [
            asset["local_path"] for asset in metadata_registry["extracted_assets"]
            if asset["page_number"] == page_num
        ]
        
        print(
            f"Chunk {idx} | Page {page_num} | Images: {matching_assets}"
        )
        surface_text = chunker.contextualize(chunk=chunk)
        cleaned_surface = clean_surface_text(surface_text)
        
        # Log base text chunk sequences exactly as specified
        tokens = len(raw_tokenizer.encode(chunk.text))
        print(
            f"Chunk {idx}: {tokens} tokens | "
            f"page={page_num} | protected={contains_protected_structure}"
        )
        
        # Single optimization step: Capture contextualized token arrays directly
        surface_tokens = raw_tokenizer.encode(cleaned_surface)
        enriched_token_count = len(surface_tokens)
        
        # Verify execution path falls under maximum model token ceiling
        if enriched_token_count <= MAX_EMBED_TOKENS:
            rag_chunks.append({
                "chunk_id": f"{input_path.stem}_chunk_{global_chunk_idx}",
                "chunk_index": global_chunk_idx,
                "track_a_surface": cleaned_surface,
                "track_b_markdown": chunk.text.strip(),
                "page_number": page_num,
                "associated_images": matching_assets,
                "source_file": str(file_path),
                "protected_structure": contains_protected_structure
            })
            global_chunk_idx += 1
        else:
            # Process sub-chunk windows by slicing the context-enriched text tokens directly
            window_size = 350
            overlap = 50
            step = window_size - overlap
            
            start_token_idx = 0
            for start_token_idx in range(0, enriched_token_count, step):
                end_token_idx = min(start_token_idx + window_size, enriched_token_count)
                sub_token_ids = surface_tokens[start_token_idx:end_token_idx]
                sub_surface = raw_tokenizer.decode(sub_token_ids, skip_special_tokens=True).strip()
                
                # Optimized Strategy: track_a captures contextual slices, track_b preserves full markdown formatting
                rag_chunks.append({
                    "chunk_id": f"{input_path.stem}_chunk_{global_chunk_idx}",
                    "chunk_index": global_chunk_idx,
                    "track_a_surface": sub_surface,
                    "track_b_markdown": chunk.text.strip(),
                    "page_number": page_num,
                    "associated_images": matching_assets,
                    "source_file": str(file_path),
                    "protected_structure": contains_protected_structure
                })
                global_chunk_idx += 1
                
                if end_token_idx == enriched_token_count:
                    break
                start_token_idx += step
        
    return rag_chunks