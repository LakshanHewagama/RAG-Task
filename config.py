import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
# Extra models OpenRouter tries if the main one is down or rate limited
FALLBACK_MODELS = [m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()]
# Chunks farther than this (cosine distance) count as unrelated to the question
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"
