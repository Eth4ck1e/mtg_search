# Archived: POC v0.1

This directory preserves the original proof-of-concept code that was the catalyst for the research independent study. It is **read-only history** — do not import from `archive/` into live code, and do not extend the modules here. They exist so the original exploration is retrievable, and so the research paper can cite "what we tried first" with concrete references.

The point-in-time snapshot of the entire repo at the moment of archival is also tagged `v0.1-poc` in git:

```
git checkout v0.1-poc      # see the full POC repo as it stood
git checkout main          # back to the current redesign
```

## What was here, and why it was archived

| Module | What it did | Why it was archived |
|---|---|---|
| `src/vector_db/build_vector_db.py` | Built a FAISS index over DistilBERT `[CLS]` token embeddings of cards. | Architecture moved to **pgvector** in Postgres. Single data store, no separate vector DB at ~30k cards. |
| `src/vector_db/query_vector_db.py` | Queried the FAISS index with the same DistilBERT-CLS embedding for the query. | Exhibited the **query-document asymmetry** failure that motivated the redesign: short informal queries embed too far from dense oracle-text representations to match. |
| `src/training/train_initial_model_mps.py` | Masked-language-model continued pretraining on card text, Apple Silicon (MPS). | Architecture no longer trains its own encoder. Uses `sentence-transformers/multi-qa-distilbert-cos-v1` off the shelf for the baseline. Fine-tuning was moved much later in the timeline and made contingent on measured failures. |
| `src/training/fine_tune_with_rulings.py` | Fine-tuned an `AutoModelForCausalLM` on card + rulings text. | Causal LM is the wrong base for a sentence encoder. The new pipeline fine-tunes (if and when justified) on a sentence-transformer with `MultipleNegativesRankingLoss`. |
| `src/training/distill_model.py` | Distilled the fine-tuned causal LM into `distilgpt2`. | Out of scope. Distillation isn't justified at this corpus size and doesn't address the asymmetry problem. |
| `src/training/validate_model.py` | Smoke-tested the saved MLM model. | Tied to the obsolete training pipeline. |
| `src/compile_initial_training_data.py` | Built training examples and used a **hand-maintained keyword-definition dictionary** (14 entries) to inject reminder text. | Hand-built dictionary doesn't scale and goes stale with each new set. The redesign auto-extracts reminder text from the corpus itself — every Wizards-printed reminder-text instance is harvested. |
| `src/web_app/` | Empty Flask scaffold. | The eventual frontend is PHP, not Flask. Removed from live code; if/when a Python web layer is wanted for quick demos, it can be rebuilt from scratch. |
| `scripts/fetch_and_process_mtg.py` | Combined download script with hardcoded Linux paths (`/home/eth4ck1e/`). | Duplicated `src/data_processing/download_bulk_data.py` and hardcoded paths broke portability. The new ingest script in Phase 2 supersedes both. |

## Key lessons the POC produced

These show up in the final paper's Background / Motivation section. Captured here so they survive the archive:

1. **Query-document asymmetry is the central retrieval problem in this domain.** Players type fragments; cards print formal prose. Embedding both with the same encoder doesn't bridge the gap. → motivates HyDE as a query-rewriting layer.
2. **`MAX_LENGTH = 64` was undersized.** Many cards have oracle text longer than 64 tokens, especially after reminder-text augmentation. Aggressive truncation hid information from the model. → new pipeline uses the full 512-token window.
3. **Hand-maintained keyword definition lists drift.** The 14-entry dictionary covered evergreen abilities but missed every set-mechanic released in the last decade and every player-jargon term ("flicker", "ramp"). → reminder-text auto-extraction.
4. **Raw `AutoModel` CLS-token embeddings aren't optimized for similarity.** `sentence-transformers` models are trained with contrastive objectives that explicitly shape the embedding space for retrieval. → use one.
5. **FAISS without a structured pre-filter loses recall on constrained queries.** "Cheap red removal" needs the SQL-level filter on `colors` and `cmc` before the semantic step, not after. → three-tower architecture.
