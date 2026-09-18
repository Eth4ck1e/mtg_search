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
        default="postgresql://invalid:invalid@invalid/invalid_set_DATABASE_URL_in_dot_env",
        description=(
            "Postgres URL. Must point at a DB with the pgvector extension enabled. "
            "Set DATABASE_URL in .env (copy .env.example as a starting point)."
        ),
    )

    # ---- LLM (HyDE query rewriter) ---------------------------------------

    anthropic_api_key: str | None = Field(
        default=None,
        description=(
            "API key for optional Claude usage (e.g., LLM-crafted Scryfall queries for the "
            "comparator, or auxiliary evaluation tooling). Not used for HyDE — HyDE runs a "
            "local instruction-tuned model."
        ),
    )
    hyde_model: str = Field(
        default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        description=(
            "Local instruction-tuned LLM used for HyDE query rewriting. MLX-quantized variant "
            "for Apple Silicon inference (~57 tok/sec generation on M3 4-bit, 4.8GB peak). "
            "Selected 2026-09-18 during M4 candidate evaluation kickoff; override via "
            "HYDE_MODEL in .env for benchmarking other checkpoints (e.g., 6-bit or 8-bit "
            "quantizations, or an entirely different model)."
        ),
    )
    hyde_server_url: str = Field(
        default="http://localhost:8080/v1",
        description=(
            "OpenAI-compatible endpoint URL for the local MLX inference server "
            "(`mlx_lm.server`). Model stays hot between calls; sub-second responses "
            "after warm-up. Start with: `mlx_lm.server --model <HYDE_MODEL> --port 8080`."
        ),
    )
    hyde_max_tokens: int = Field(
        default=256,
        description=(
            "Max tokens HyDE may generate for its structured JSON output. Reduced from 512 "
            "(the original HyDE paper's default for full hypothetical documents) to 256 "
            "because our JSON schema is compact — filters object plus a single-sentence "
            "hypothetical_card typically fits in 100-180 tokens. Setting a tighter cap "
            "reclaims latency budget without truncating real outputs; raise if empirical "
            "outputs start hitting the cap. See docs/sources/2025_never-come-up-empty-*.pdf "
            "and the HyDE-latency section of docs/sources/README.md for the tradeoff."
        ),
    )
    hyde_temperature: float = Field(
        default=0.0,
        description=(
            "Sampling temperature for HyDE. Default 0.0 (deterministic) for reproducible "
            "structured output; raise cautiously if JSON reliability holds but hypothetical "
            "card text is too rigid."
        ),
    )

    # ---- Data sources -----------------------------------------------------

    scryfall_bulk_endpoint: str = Field(
        default="https://api.scryfall.com/bulk-data/oracle-cards",
        description=(
            "Scryfall bulk-data API endpoint for the oracle-cards dataset. Returns metadata "
            "including 'jsonl_download_uri' and 'compressed_size'; the actual bulk file is "
            "'.jsonl.gz'."
        ),
    )
    scryfall_user_agent: str = Field(
        default="mtg_search/0.2.0 (+https://github.com/Eth4ck1e/mtg_search)",
        description="User-Agent header sent to Scryfall API per their courtesy guidelines.",
    )

    # ---- Embeddings -------------------------------------------------------

    embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description=(
            "HuggingFace-hosted embedding model for card Oracle text and HyDE outputs. "
            "Nomic Embed v1.5 is a 137M-param bi-encoder in the sentence-transformer lineage "
            "with 768-dim output and Matryoshka representation support. Selected 2026-09-11 "
            "during the baseline-abandonment pivot. Requires task-specific prefixes on inputs "
            "(see src/preprocess_text.py: NOMIC_DOCUMENT_PREFIX, NOMIC_QUERY_PREFIX)."
        ),
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
