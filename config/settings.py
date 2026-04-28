import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # ── LLM provider (free: groq / huggingface | paid: anthropic) ──────────
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    hf_api_token: str = Field(default_factory=lambda: os.getenv("HF_API_TOKEN", ""))
    hf_model: str = Field(default_factory=lambda: os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # ── Embeddings (local sentence-transformers — no key needed) ────────────
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # ── Vector store backend ─────────────────────────────────────────────────
    vector_store_backend: str = Field(
        default_factory=lambda: os.getenv("VECTOR_STORE_BACKEND", "chroma")
    )   # "chroma" | "faiss"

    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))
    )
    faiss_persist_dir: str = Field(
        default_factory=lambda: os.getenv("FAISS_PERSIST_DIR", str(BASE_DIR / "data" / "faiss_db"))
    )

    # ── Pipeline backend ─────────────────────────────────────────────────────
    pipeline: str = Field(
        default_factory=lambda: os.getenv("PIPELINE", "original")
    )   # "original" | "langchain"

    # ── Logging / performance ─────────────────────────────────────────────────
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    batch_size: int = Field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "10")))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))

    # ── RAG parameters ────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5

    # ── Audit thresholds ──────────────────────────────────────────────────────
    hallucination_threshold: float = 0.3
    credibility_threshold: float = 0.6
    greenwashing_threshold: float = 0.5

    # ── ChromaDB collection names ─────────────────────────────────────────────
    reports_collection: str = "climate_reports"
    benchmarks_collection: str = "ipcc_benchmarks"

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: Path = BASE_DIR / "outputs"
    data_dir: Path = BASE_DIR / "data"
    sample_reports_dir: Path = BASE_DIR / "data" / "sample_reports"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
