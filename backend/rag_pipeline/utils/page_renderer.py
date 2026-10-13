import json
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image
from rag_pipeline import config

def render_pages(doc_stem: str, page_numbers):
    """
    Render requested PDF pages on demand.
    Already-rendered pages are loaded from cache.
    Resizes image assets proportionally to stay within strict token boundaries on CPU.
    """
    metadata_path = Path(config.OUTPUT_DIR) / f"{doc_stem}_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    pdf_path = metadata["pdf_path"]

    cache_dir = Path(config.OUTPUT_DIR) / f"{doc_stem}_cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    pdf = pdfium.PdfDocument(pdf_path)

    rendered_paths = []

    # Safe token dimension ceiling for local CPU vision models (e.g., Llama-3.2-Vision, Qwen2-VL)
    # 672px or 896px provides clear legibility without overloading the context buffer
    MAX_IMAGE_DIMENSION = 672 

    for page_no in sorted(set(page_numbers)):
        cache_file = cache_dir / f"page_{page_no}.png"

        if cache_file.exists():
            rendered_paths.append(str(cache_file))
            continue
        
        # pypdfium2 pages are 0-indexed internally
        page = pdf[page_no - 1]
        
        # 1. Render at higher scaling factors to preserve structural edge sharpness
        image = page.render(scale=config.SCALE_FACTOR).to_pil()
        
        # 2. Rescale proportionally WITHOUT cropping to fit the token budget constraints
        width, height = image.size
        if max(width, height) > MAX_IMAGE_DIMENSION:
            if width > height:
                new_width = MAX_IMAGE_DIMENSION
                new_height = int((height / width) * MAX_IMAGE_DIMENSION)
            else:
                new_height = MAX_IMAGE_DIMENSION
                new_width = int((width / height) * MAX_IMAGE_DIMENSION)
            
            # Use Resampling.LANCZOS to retain high quality text crispness for the LLM
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 3. Save to cache string path
        image.save(cache_file, format="PNG", optimize=True)
        rendered_paths.append(str(cache_file))
        
    pdf.close()
    return rendered_paths