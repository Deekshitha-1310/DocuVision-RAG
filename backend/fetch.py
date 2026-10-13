import os
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.models.layout_model import LayoutModel
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

from pathlib import Path

# Define your target storage directory
target_dir = str(Path(__file__).resolve().parent.parent / "models" / "docling_models")
os.makedirs(target_dir, exist_ok=True)

# Point Docling to your directory via the standard environment variable
os.environ["DOCLING_SERVE_ARTIFACTS_PATH"] = target_dir

print(f"Downloading models to: {target_dir}")
# This will trigger the download if the models aren't present
try:
    # Initializing a pipeline will force it to check/download models to the path above
    pipeline = StandardPdfPipeline(artifacts_path=target_dir)
    print("Models verified/downloaded successfully.")
except Exception as e:
    print(f"An error occurred: {e}")