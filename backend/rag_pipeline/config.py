import os
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Strict offline enforcement
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

# Root workspace path
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Paths for models
MODELS_DIR = BASE_DIR / "models"
COLPALI_MODEL_DIR = str(MODELS_DIR / "colpali")
TOKENIZER_DIR = str(MODELS_DIR / "local_tokenizer")
DOCLING_MODELS_DIR = str(MODELS_DIR / "docling_models")
EMBEDDING_MODEL_DIR = str(MODELS_DIR / "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_DIR = str(MODELS_DIR / "reranker_model")

# Redirected base output directory
OUTPUT_DIR = str(BASE_DIR / "processed_data")

# Hardware Setup
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("cuda")
    CURRENT_DTYPE = torch.bfloat16
    BATCH_SIZE = 4
    SCALE_FACTOR = 2.0  # Crisp 144 DPI
else:
    DEVICE = "cpu"
    print("cpu")
    CURRENT_DTYPE = torch.float32
    BATCH_SIZE = 1
    SCALE_FACTOR = 1.5  # ~100 DPI fallback

def ensure_tokenizer_exists():
    """Ensures a fallback sentence tokenizer exists locally."""
    model_dir = Path(TOKENIZER_DIR)
    print(model_dir)
    if not model_dir.exists() or not list(model_dir.glob("*")):
        print(f"Local weights not found at '{TOKENIZER_DIR}'. Fetching from HF Hub...")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").save_pretrained(str(model_dir))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

# -------------------------------------------------------------
# CENTRALIZED VISION LLM CONFIGURATION
# -------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_LLM_MODEL = "qwen2.5vl:3b"
VISION_LLM_TEMPERATURE = 0.0

UPLOAD_DIR = "data/uploads"

# --- Lightweight Query Normalizer LLM Configuration ---
NORMALIZER_LLM_MODEL = "qwen2.5:3b" 
NORMALIZER_LLM_TEMPERATURE = 0.0     # Strictly 0.0 for deterministic rewrites

# -------------------------------------------------------------
# RAG PIPELINE RESOURCE CEILINGS (CPU MEMORY MANAGEMENT)
# -------------------------------------------------------------
RETRIEVAL_TOP_K = 8      
RERANK_TOP_K = 4          
RERANK_THRESHOLD = 0.15 

MAX_TEXT_CONTEXT_CHARS = 4500

CUDA_MAX_PAGES = 2
CPU_MAX_PAGES = 2

# Reranker selection model ID running purely locally via Hugging Face/SentenceTransformers
RERANKER_MODEL_ID = "BAAI/bge-reranker-large"

# =============================================================
# REDIRECTED DOCLING STORAGE SUITE
# =============================================================

# Directory to dump cropped images, figures, and parsed data tables
ASSET_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "extracted_assets")
os.makedirs(ASSET_OUTPUT_DIR, exist_ok=True)

# Path to initialize your vector store file locally
VECTOR_DB_PATH = os.path.join(OUTPUT_DIR, "vector_store.json")

# Resolved Vector Storage Directory targeted by the Vector Store engine
VECTOR_INDEX_STORAGE = os.path.join(OUTPUT_DIR, "vector_store_index")
os.makedirs(VECTOR_INDEX_STORAGE, exist_ok=True)