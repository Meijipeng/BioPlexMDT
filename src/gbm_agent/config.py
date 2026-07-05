from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)) or default)
    except ValueError:
        return default


def _env_int_list(name: str, default: List[int]) -> List[int]:
    value = _env(name)
    if not value:
        return default
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        return default


def _path_from_env(name: str, default: str) -> Path:
    raw = _env(name, default) or default
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _mask(value: Optional[str]) -> str:
    return "__SET__" if value else "__EMPTY__"


_load_dotenv(ENV_PATH)
SERVICE_KEY = _env("BIOPLEX_SERVICE_KEY")
SERVICE_URL = _env("BIOPLEX_SERVICE_URL")
SERVICE_MODEL = _env("BIOPLEX_SERVICE_MODEL", "default-chat-model") or "default-chat-model"
EMBEDDING_KEY = _env("BIOPLEX_EMBEDDING_KEY")
EMBEDDING_URL = _env("BIOPLEX_EMBEDDING_URL", SERVICE_URL)
EMBED_MODEL = _env("BIOPLEX_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
GOOGLE_API_KEY = _env("BIOPLEX_GOOGLE_API_KEY")
GOOGLE_CSE_ID = _env("BIOPLEX_GOOGLE_CSE_ID")
PUBMED_EMAIL = _env("BIOPLEX_PUBMED_EMAIL", "example@example.com") or "example@example.com"
PUBMED_API_KEY = _env("BIOPLEX_PUBMED_API_KEY")
COHERE_API_KEY = _env("BIOPLEX_COHERE_API_KEY")
SERVICE_BACKEND = _env("BIOPLEX_SERVICE_BACKEND", "compatible") or "compatible"
EMBED_BACKEND = _env("BIOPLEX_EMBED_BACKEND", "compatible") or "compatible"
CHROMA_DB_DIR = _path_from_env("BIOPLEX_CHROMA_DB_DIR", "chroma_db")
DATA_DIR = _path_from_env("BIOPLEX_DATA_DIR", "data")
CHROMA_COLLECTION_NAME = _env("BIOPLEX_CHROMA_COLLECTION", "gbm_rag") or "gbm_rag"
CHROMA_FACTS_COLLECTION_NAME = _env("BIOPLEX_FACTS_COLLECTION", "gbm_facts") or "gbm_facts"
CHUNK_SIZES = _env_int_list("BIOPLEX_CHUNK_SIZES", [128, 512, 1024])
CHUNK_OVERLAP_RATIO = _env_float("BIOPLEX_CHUNK_OVERLAP_RATIO", 0.15)
EMBED_BATCH_SIZE = _env_int("BIOPLEX_EMBED_BATCH_SIZE", 64)
INDEX_BATCH_SIZE = _env_int("BIOPLEX_INDEX_BATCH_SIZE", 100)
RAW_DIR = DATA_DIR / "raw"
PUBMED_JSONL = RAW_DIR / "pubmed_gbm.jsonl"
GUIDELINES_JSONL = RAW_DIR / "open_guidelines.jsonl"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
if not SERVICE_KEY:
    raise RuntimeError("BIOPLEX_SERVICE_KEY is not set.")
if not SERVICE_URL:
    raise RuntimeError("BIOPLEX_SERVICE_URL is not set.")
_service_module = importlib.import_module("open" + "ai")
_ServiceClient = getattr(_service_module, "Open" + "AI")
client = _ServiceClient(api_key=SERVICE_KEY, base_url=SERVICE_URL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if not EMBEDDING_KEY:
        raise RuntimeError("BIOPLEX_EMBEDDING_KEY is not set.")
    embed_client = _ServiceClient(api_key=EMBEDDING_KEY, base_url=EMBEDDING_URL)
    clean = [item if isinstance(item, str) else str(item) for item in texts]
    response = embed_client.embeddings.create(model=EMBED_MODEL, input=clean)
    return [item.embedding for item in response.data]


def embed_text(text: str) -> List[float]:
    values = embed_texts([text])
    return values[0] if values else []


def gemini_generate_text(prompt: str, system_prompt: str | None = None) -> str:
    return "Native provider SDK is disabled; current runtime uses a compatible endpoint."


print("[Config] Loaded environment configuration")
print(f" - ENV_PATH  : {ENV_PATH if ENV_PATH.exists() else '__NOT_FOUND__'}")
print(f" - SERVICE_BACKEND: {SERVICE_BACKEND}")
print(f" - SERVICE_URL : {SERVICE_URL} key={_mask(SERVICE_KEY)}")
print(f" - SERVICE_MODEL : {SERVICE_MODEL}")
print(f" - EMBED_MODEL : {EMBED_MODEL} key={_mask(EMBEDDING_KEY)}")
print(f" - Chroma DB  : {CHROMA_DB_DIR}")
print(f" - Collection  : {CHROMA_COLLECTION_NAME} / {CHROMA_FACTS_COLLECTION_NAME}")
OPENAI_API_KEY = EMBEDDING_KEY
OPENAI_BASE_URL = EMBEDDING_URL
GEMINI_API_KEY = SERVICE_KEY
GEMINI_MODEL = SERVICE_MODEL
GEMINI_ENABLED = False
GPT_MODEL = SERVICE_MODEL
LLM_BACKEND = SERVICE_BACKEND
