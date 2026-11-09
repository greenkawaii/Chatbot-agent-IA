import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # dossier qui contient ce fichier
# --- Moteur LLM (Ollama) ----------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
# TODO : LLM_TIMEOUT (entier, 120 secondes par défaut)
LLM_TIMEOUT = int(120)
# TODO : LLM_NUM_CTX (entier, 8192 par défaut) : taille de la fenêtre de contexte
LLM_NUM_CTX = int(8192)

# --- Données ----------------------------------------------------------
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "ecommerce.db"))
# TODO : DOCS_DIR, le dossier des documents métier (data/docs)
DOCS_DIR = BASE_DIR / "data" / "docs"

# --- Recherche documentaire (RAG) -------------------------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# TODO : CHUNK_SIZE = 500, CHUNK_OVERLAP = 100, TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

# --- Agent ------------------------------------------------------------
# TODO : MAX_STEPS = 5, MAX_HISTORY_TURNS = 6, SQL_MAX_ROWS = 50
MAX_STEPS = 5
MAX_HISTORY_TURNS = 6
SQL_MAX_ROWS = 50

# --- Journaux ---------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"