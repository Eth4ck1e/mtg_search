"""Project configuration.

Single source of truth for environment-driven settings. Values are loaded
from environment variables (and a local .env file in development) and
validated by Pydantic. Import the singleton `settings` everywhere; do not
re-read environment variables or reconstruct paths elsewhere.

Paths are exposed as properties so they always derive consistently from
the repo root. Only the user-tunable values (database URL, API keys,
model identifiers, hyperparameters) are env-driven.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Database ---------------------------------------------------------

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/mtg_search",
        description="Postgres URL. Must point at a DB with the pgvector extension enabled.",
    )

    # ---- LLM (HyDE query rewriter) ---------------------------------------

    anthropic_api_key: str | None = Field(
        default=None,
        description="API key for the HyDE query rewriter. Optional until Phase 4.",
    )
    hyde_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model identifier used for HyDE query rewriting.",
    )

    # ---- Embeddings -------------------------------------------------------

    embedding_model: str = Field(
        default="sentence-transformers/multi-qa-distilbert-cos-v1",
        description="HuggingFace sentence-transformer model for card text and HyDE outputs.",
    )
    embedding_dim: int = Field(
        default=768,
        description="Dimension produced by embedding_model. Must match the pgvector column.",
    )
    max_length: int = Field(
        default=512,
        description="Token truncation budget. The POC used 64 — documented failure mode.",
    )
    embed_batch_size: int = Field(default=32)

    # ---- Preprocessing ----------------------------------------------------

    preprocess_version: str = Field(
        default="v1",
        description="Version string bumped whenever build_embedding_text logic changes. Triggers re-embedding.",
    )

    # ---- Derived paths (not env-driven) ----------------------------------

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def keywords_dir(self) -> Path:
        return self.data_dir / "keywords"

    @property
    def logs_dir(self) -> Path:
        return self.repo_root / "logs"

    @property
    def prompts_dir(self) -> Path:
        return self.repo_root / "prompts"

    @property
    def embedding_version(self) -> str:
        """Composite version identifier stored alongside each embedding."""
        return f"{self.embedding_model}|preproc={self.preprocess_version}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — instantiate once per process."""
    return Settings()


settings = get_settings()
