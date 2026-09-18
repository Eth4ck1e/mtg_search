# mtg_search

Natural-language semantic search for Magic: The Gathering cards. CSCI 5953 independent study, CSUSB. Author: Mitchell Trafford.

**Status:** M4 (query rewriting + SQL pre-filter) in progress. See [`CLAUDE.md`](CLAUDE.md) §8 for the current milestone map.

## Why this project exists

MTG players think in informal terms: *"a blue counterspell that costs 2"*, *"cards that flicker creatures"*, *"cheap red removal"*. Card text on the cards themselves is formal, prose-y, and 30+ years of terminology drift mean modern players type words that don't appear on older cards. Traditional keyword search misses this; naive vector search on raw card text misses it too. This project measures and addresses the **query–document asymmetry** at the heart of that mismatch, and evaluates whether the resulting system can match or beat the domain-standard search interface (Scryfall) for non-expert users.

## Architecture

A **three-stage retrieval cascade** over ~30,000 unique cards, backed by Postgres + pgvector:

1. **HyDE query rewriter** (local instruction-tuned LLM) transforms the user's natural-language query into (a) structured filter attributes and (b) hypothetical card ability text.
2. **SQL pre-filter** narrows the candidate set on the structured attributes (colors, mana value, types, legality).
3. **Semantic vector search** ranks the pre-filtered candidates by cosine similarity between the hypothetical ability text embedding and the card ability text embeddings.

The full architecture spec, working conventions, and anti-patterns live in [`CLAUDE.md`](CLAUDE.md).

## Repo layout

```
.
├── CLAUDE.md                    # Architecture spec + working conventions (read first)
├── README.md                    # This file
├── pyproject.toml               # Python package definition (>=3.11)
├── docker-compose.yml           # pgvector/pgvector:pg16 on localhost:5432
├── .env.example                 # Template for local credentials (copy to .env)
├── archive/poc_v1/              # Original POC, preserved for the paper's Background section
├── _planning-archive/           # Pre-repo local planning docs (historical)
├── configs/                     # Retrieval-run configuration files (YAML)
├── data/
│   ├── raw/                     # Scryfall bulk dumps (gitignored)
│   ├── processed/               # Corpus survey outputs (gitignored, regenerable)
│   ├── eval/                    # 26-query evaluation set + methodology notes
│   └── keywords/                # Reminder-text dictionary + manual overrides
├── docs/
│   ├── archive/                 # Original proposal + historical planning
│   ├── journal/                 # Dated decision entries (source material for paper)
│   ├── process/                 # Workflow rulebooks (milestone-checkpoints.md, timeline.md)
│   ├── roadmap/                 # Per-phase task lists (M0-M7 mapped in CLAUDE.md §8)
│   ├── sources/                 # Academic source PDFs (gitignored) + annotated bibliography
│   └── thesis/                  # Thesis-class deliverables (abstract, timeline)
├── scripts/                     # Entry-point CLI scripts
│   ├── download_scryfall.py     # Scryfall bulk-data download
│   ├── survey_corpus.py         # Corpus characterization
│   ├── ingest.py                # Bulk file → cards table UPSERT
│   ├── build_keyword_dict.py    # Reminder-text extraction from corpus
│   ├── embed.py                 # Encode cards → pgvector
│   ├── migrate.py               # SQL migration runner
│   ├── eval_lookup.py           # Scryfall candidate finder for eval curation
│   ├── render_review.py         # HTML review UI for eval set
│   └── evaluate.py              # Run a config, write experiment_runs row
├── src/                         # Library code (importable modules)
│   ├── config.py                # Pydantic settings (single source of truth)
│   ├── logging_utils.py         # PipelineRun JSONL context manager
│   ├── preprocess_text.py       # build_embedding_text + reminder-text loader
│   ├── data_processing/         # scryfall_classify, ingest_transform, keyword_extract
│   ├── db/                      # experiment_log writer + SQL migrations
│   ├── eval/                    # Pure-Python metric calculation (recall@K, MRR)
│   └── utils/                   # select_device, misc helpers
├── tests/                       # Test suite (~80 tests, unit + integration)
└── logs/                        # JSONL pipeline-run logs (gitignored)
```

## Quickstart

**Prerequisites:** Python ≥3.11 (3.13 recommended), Docker Desktop, Homebrew.

```bash
# 1. Clone and enter the repo
git clone https://github.com/Eth4ck1e/mtg_search.git
cd mtg_search

# 2. Set up the Python environment
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Copy the environment template and set your Postgres credentials
cp .env.example .env
# Edit .env — DATABASE_URL at minimum

# 4. Bring up pgvector
docker compose up -d

# 5. Run migrations to create the cards + experiment_runs tables
PYTHONPATH="$PWD" python scripts/migrate.py

# 6. Download the current Scryfall bulk-data (oracle-cards, ~24MB gzipped)
PYTHONPATH="$PWD" python scripts/download_scryfall.py

# 7. Parse the bulk file into the cards table
PYTHONPATH="$PWD" python scripts/ingest.py

# 8. Build the reminder-text keyword dictionary
PYTHONPATH="$PWD" python scripts/build_keyword_dict.py

# 9. Encode the corpus (embeddings → pgvector)
PYTHONPATH="$PWD" python scripts/embed.py

# 10. Start the MLX HyDE server (Apple Silicon only; leave running in a
#     separate terminal for the retrieval pipeline to call)
PYTHONPATH="$PWD" python -m mlx_lm server \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --port 8080 --log-level WARNING

# 11. Run a retrieval configuration and log the results
PYTHONPATH="$PWD" python scripts/evaluate.py --config configs/baseline.yaml
```

> **Note on `PYTHONPATH`:** Python 3.13.0 has a `.pth` file processing bug that breaks editable-install imports. Prefixing `PYTHONPATH="$PWD"` works around it. Upgrading to a patched Python 3.13.x (via `brew upgrade python@3.13`) removes the need for the prefix.

> **Note on MLX (Apple Silicon):** the HyDE stage uses [`mlx-lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms) for local LLM inference — ~50% faster than llama.cpp-based backends on M-series chips. The 4-bit Llama 3.1 8B Instruct variant runs at ~57 tokens/sec on an M3 with ~4.8GB peak memory. On non-macOS platforms, `mlx-lm` is skipped by the platform marker in `pyproject.toml`; substitute any OpenAI-compatible local-serve backend (vLLM, llama.cpp server, TGI) and point `HYDE_SERVER_URL` in `.env` at it.

## Where to look next

- **New here?** → [`CLAUDE.md`](CLAUDE.md)
- **What's the current milestone?** → [`CLAUDE.md`](CLAUDE.md) §8 milestone table
- **What's the semester schedule?** → [`docs/process/timeline.md`](docs/process/timeline.md)
- **Why does the architecture look this way?** → [`CLAUDE.md`](CLAUDE.md) §2, and [`docs/archive/2025-11-03-original-proposal.md`](docs/archive/2025-11-03-original-proposal.md) for historical context
- **What was just decided?** → [`docs/journal/`](docs/journal/) (most recent dated entry)
- **Which academic sources back the project?** → [`docs/sources/README.md`](docs/sources/README.md)

## License

MIT — see [`LICENSE`](LICENSE).
