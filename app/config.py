import os
from dotenv import load_dotenv

load_dotenv()

# API Keys validation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GROQ_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing GROQ_API_KEY or PINECONE_API_KEY in environment variables.")

# Paths & Settings
PDF_PATH = r"data/NovaCore_Multimodal_Company_Report_2026.pdf"
OUTPUT_IMG_DIR = "extracted_images"
CACHE_FILE = "image_summaries_cache.json"
INDEX_NAME = "multimodel-rag-project"

# Model Configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VISION_MODEL = "qwen/qwen3.8-27b"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"