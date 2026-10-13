import os
import json
import torch
import multiprocessing
import math  # Added to calculate batch progress steps accurately
from pathlib import Path
from shutil import copy2
from torch.utils.data import DataLoader
import pypdfium2 as pdfium
from transformers import AutoTokenizer

# Force offline mode for all huggingface-based calls globally
os.environ["HF_HUB_OFFLINE"] = "1"

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from colpali_engine.models import ColPali, ColPaliProcessor
from rag_pipeline import config


class OfflineHFTokenizerDoclingWrapper(BaseTokenizer):
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, local_model_path: str):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        
    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))
        
    def get_max_tokens(self) -> int: 
        return 256
        
    def get_tokenizer(self): 
        return self._tokenizer


# -------------------------------------------------------------
# GLOBAL CACHING INFRASTRUCTURE (SINGLETON PATTERN)
# -------------------------------------------------------------
_COLPALI_MODEL = None
_COLPALI_PROCESSOR = None

def get_colpali():
    """
    Lazy-loads and caches the ColPali model and processor globally 
    to avoid expensive VRAM/RAM re-allocations across multiple documents.
    """
    global _COLPALI_MODEL, _COLPALI_PROCESSOR

    
    if _COLPALI_MODEL is None:
        
        _COLPALI_MODEL = ColPali.from_pretrained(
            config.COLPALI_MODEL_DIR,
            torch_dtype=config.CURRENT_DTYPE,
            attn_implementation="eager",
            local_files_only=True
        )

        _COLPALI_MODEL = _COLPALI_MODEL.to(
            config.DEVICE
        ).eval()
        
        _COLPALI_PROCESSOR = ColPaliProcessor.from_pretrained(
            config.COLPALI_MODEL_DIR, 
            local_files_only=True
        )
        print("[INIT] ColPali model successfully cached in active memory.")

    return _COLPALI_MODEL, _COLPALI_PROCESSOR


def run_colpali_ingestion(pdf_path: Path, progress_callback=None):
    print(f"\n--- [1/2] Initializing ColPali Extraction ---")
    
    if progress_callback:
        progress_callback(10, "Initializing and waking up local ColPali vision framework...")
        
    model, processor = get_colpali()

    print("Rendering PDF pages for ColPali embedding generation...")
    if progress_callback:
        progress_callback(15, "Rendering PDF pages to memory canvases...")

    pdf = pdfium.PdfDocument(str(pdf_path))
    page_count = len(pdf)

    images = []
    for page in pdf:
        img = page.render(scale=config.SCALE_FACTOR).to_pil()
        images.append(img)

    pdf.close()

    dataloader = DataLoader(
        dataset=images, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=processor.process_images
    )
    
    total_batches = math.ceil(len(images) / config.BATCH_SIZE)
    all_embeddings = []

    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            embeddings_doc = model(**batch)
            all_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        
        # Scale batch parsing updates smoothly between 20% and 45% of overall job load
        if progress_callback:
            batch_ratio = (batch_idx + 1) / total_batches
            current_pct = 20 + int(batch_ratio * 25)
            progress_callback(current_pct, f"ColPali processing page batch {batch_idx + 1}/{total_batches}...")
            
    return all_embeddings, page_count


def run_docling_ingestion(pdf_path: Path, output_json_path: Path, progress_callback=None):
    print(f"\n--- [2/2] Initializing Docling Parsing ---")
    
    if progress_callback:
        progress_callback(50, "Docling starting document structural layout analysis...")
    
    options = PdfPipelineOptions(
        artifacts_path=config.DOCLING_MODELS_DIR,
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=True,
        generate_table_images=True
    )
    options.table_structure_options.mode = TableFormerMode.FAST

    system_cores = max(1, multiprocessing.cpu_count() - 1)
    options.accelerator_options = AcceleratorOptions(
        num_threads=system_cores,
        device=AcceleratorDevice.CUDA if config.DEVICE == "cuda" else AcceleratorDevice.CPU
    )

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    doc = converter.convert(str(pdf_path)).document

    if progress_callback:
        progress_callback(60, "Docling structural parsing complete. Starting partition grouping...")

    table_count = 0
    image_count = 0

    for item, _level in doc.iterate_items():
        item_type = type(item).__name__
        if "Table" in item_type:
            table_count += 1
        elif "Picture" in item_type:
            image_count += 1

    conforming_tokenizer = OfflineHFTokenizerDoclingWrapper(local_model_path=config.TOKENIZER_DIR)
    chunker = HybridChunker(tokenizer=conforming_tokenizer, max_tokens=256)
    
    production_payloads = []
    current_page = 1

    for chunk_idx, chunk_obj in enumerate(chunker.chunk(doc)):
        if chunk_obj.meta.doc_items:
            items_list = chunk_obj.meta.doc_items if isinstance(chunk_obj.meta.doc_items, list) else [chunk_obj.meta.doc_items]
            if items_list and hasattr(items_list[0], "prov") and items_list[0].prov:
                current_page = items_list[0].prov[0].page_no

        production_payloads.append({
            "node_id": f"node_{pdf_path.stem}_{chunk_idx:05d}",
            "chunk_index": chunk_idx,
            "page_num": current_page,
            "chunk_text": chunker.serialize(chunk_obj).strip()
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(production_payloads, f, indent=4, ensure_ascii=False)
        
    if progress_callback:
        progress_callback(65, f"Saved {len(production_payloads)} chunks. Building manifest maps...")
        
    return {
        "tables": table_count,
        "images": image_count,
        "chunks": len(production_payloads)
    }


def ingest_document(target_pdf_path: str, progress_callback=None):
    pdf_path = Path(target_pdf_path)
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    upload_dir = Path(getattr(config, "UPLOAD_DIR", "data/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    stored_pdf = upload_dir / pdf_path.name
    if not stored_pdf.exists():
        if progress_callback:
            progress_callback(5, f"Archiving artifact to internal storage location...")
        print(f"Archiving artifact to internal storage: {stored_pdf}")
        copy2(pdf_path, stored_pdf)
    
    pdf_path = stored_pdf

    emb_out_path = out_dir / f"{pdf_path.stem}_colpali_embeddings.pt"
    json_out_path = out_dir / f"{pdf_path.stem}_production_chunks.json"

    # Track 1: ColPali Image Multi-Vector Extraction
    embeddings, page_count = run_colpali_ingestion(pdf_path, progress_callback)
    torch.save(embeddings, emb_out_path)
    print(f"Saved {len(embeddings)} page tensors to: {emb_out_path}")

    # Track 2: Docling Document Structural Layout Splitting
    config.ensure_tokenizer_exists()
    docling_stats = run_docling_ingestion(pdf_path, json_out_path, progress_callback)
    print(f"Saved text chunks to: {json_out_path}")
    print("[SUCCESS] Ingestion Complete.")
    
    # Save Run Metadata
    metadata_path = out_dir / f"{pdf_path.stem}_metadata.json"
    metadata = {
        "pdf_path": str(pdf_path.resolve()),
        "pages": page_count,
        "tables": docling_stats["tables"],
        "images": docling_stats["images"],
        "chunks": docling_stats["chunks"]
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved run validation metadata: {metadata_path}")

    if progress_callback:
        progress_callback(70, "Data ingestion sequence completed successfully.")

    return metadata