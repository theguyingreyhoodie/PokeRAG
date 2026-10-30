"""Paths and settings. Everything can be overridden with environment variables (or a .env file)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("POKEDEX_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"            # cached, slimmed PokéAPI responses
DATASET_PATH = DATA_DIR / "dataset.json"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "pokedex"

POKEAPI_BASE = "https://pokeapi.co/api/v2"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")

# LLM provider: "ollama" (local, free), "openai" (any OpenAI-compatible API, e.g. Groq),
# "anthropic" (Claude), or "none" (retrieval + analysis only, no generation).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")

TOP_K = int(os.getenv("TOP_K", "6"))
