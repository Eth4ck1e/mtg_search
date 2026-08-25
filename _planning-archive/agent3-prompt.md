# MTG Vector Database Project Guide
## Pragmatic Development Assistant Configuration

### Project Context
Building a semantic search system for Magic: The Gathering cards using vector embeddings.

**Tech Stack:**
- Python 3.9+, PyTorch 2.0+, sentence-transformers 2.2+
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- FAISS vector database (local, CPU-optimized)
- Scryfall API (27k cards, ~50-100MB JSON)
- Streamlit for UI (optional)

**Timeline:** 14 weeks, 10-13 hrs/week (Spring 2026)

**Project Root:** `/Users/mitchelltrafford/Documents/Development/Independent Study - MTG Vector DB`

**Current Status:** Planning phase, empty directory, starting from scratch

---

## Communication Preferences

### Be Direct and Action-Oriented
- Start with what you'll do, not what could be done
- Show code, not descriptions of code
- Flag blockers immediately, propose solutions
- Skip theoretical discussions unless explicitly requested

### Minimize Back-and-Forth
- Ask clarifying questions upfront if requirements are ambiguous
- Make reasonable assumptions for minor decisions, document them
- Batch related questions together
- Provide 2-3 concrete options for major decisions, with recommendations

### Default to Implementation
- When asked "Can we do X?", respond with working code unless technically infeasible
- Prefer "Here's a working implementation..." over "Yes, you could try..."
- Include error handling in initial implementation, not as an afterthought

---

## Development Workflow

### File Operations
**ALWAYS:**
- Use absolute paths (never relative)
- Read files before editing them
- Verify parent directories exist before creating new files
- Use `.gitignore` for `data/`, `models/`, `index/`, `venv/`

**NEVER:**
- Create files proactively without explicit request
- Write README or documentation files unless asked
- Use emojis in code or commits (professional tone)

### Code Standards
**Default patterns:**
- Type hints for function signatures
- Docstrings for public functions (Google style)
- Descriptive variable names (no single-letter except loop indices)
- Error handling with try/except and informative messages
- Progress bars (`tqdm`) for long operations
- Logging with `loguru` for debugging

**Example function signature:**
```python
def embed_cards(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32
) -> np.ndarray:
    """Generate embeddings for card texts using sentence-transformers.

    Args:
        texts: List of card oracle texts to embed
        model_name: HuggingFace model identifier
        batch_size: Number of texts to encode per batch

    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
```

### Git Usage
**Commit strategy:**
- Commit after completing each logical unit (feature, fix, refactor)
- Wait for explicit request to commit - don't be proactive
- Use conventional commit format: `type: description`
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Include Co-Authored-By: Claude footer
- Never amend commits from other developers
- Check authorship before amending: `git log -1 --format='%an %ae'`

**Never:**
- Force push to main/master
- Skip hooks (--no-verify)
- Commit secrets (.env, credentials)
- Push without explicit request

---

## Testing and Validation

### Incremental Development
**Always test on small subsets first:**
- Start with 10 cards, then 100, then full 27k
- Profile code with small datasets before scaling
- Use `line_profiler` for performance bottlenecks
- Validate outputs at each pipeline stage

**Validation checkpoints:**
1. Data downloaded? Check file size, parse first 10 entries
2. Preprocessing? Verify no null oracle_text, inspect edge cases (DFCs, split cards)
3. Embeddings? Check shape, verify not all zeros, compute sample similarity
4. FAISS index? Test search with known query, measure latency
5. Search results? Manual review of top-5 for 3-5 queries

### Error Handling Priorities
**Must handle:**
- API failures (Scryfall rate limits, timeouts)
- Missing/malformed data (null oracle_text, invalid JSON)
- Out-of-memory errors (batch size too large)
- File not found (index/model missing)

**Pattern:**
```python
try:
    # Core logic
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Provide recovery action or clear error message
    raise RuntimeError("Helpful message for user") from e
```

---

## Project-Specific Guidelines

### Data Pipeline
**Scryfall data handling:**
- Fetch from `https://api.scryfall.com/bulk-data/oracle-cards`
- Save raw JSON with timestamp: `oracle-cards-YYYYMMDD.json`
- Parse fields: `name`, `oracle_text`, `type_line`, `cmc`, `colors`, `card_faces`
- Handle double-faced cards: concatenate faces with " // "
- Skip tokens, art cards, promotional variants (check `layout` field)

**Preprocessing:**
- Strip leading/trailing whitespace
- Remove reminder text (text in parentheses) if present
- Convert to lowercase for embedding (optional, test both)
- Store as CSV: `cards_clean.csv` with columns: `id, name, oracle_text, cmc, colors, types`

### Embedding Generation
**Optimization:**
- Use `batch_size=32` as starting point (adjust based on RAM)
- Enable `show_progress_bar=True` for visibility
- Cache embeddings: save to `embeddings.npy`, rebuild only on data changes
- Normalize vectors for cosine similarity: `faiss.normalize_L2(embeddings)`

**Code template:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
```

### FAISS Index
**Configuration:**
- Use `IndexFlatIP` (inner product) for exact search on <100k vectors
- Save index: `faiss.write_index(index, 'mtg_faiss.index')`
- Store metadata separately: `pickle.dump(metadata, 'card_metadata.pkl')`
- If query latency >100ms, profile and optimize batch operations

**Search interface:**
- Input: text query
- Process: embed query with same model
- Search: `index.search(query_embedding, k=10)`
- Output: list of (card_name, score, oracle_text, mana_cost)

### Fine-Tuning Dataset
**Creation strategy:**
- Start with 50 high-quality pairs, expand to 500-1000
- Focus on MTG-specific terminology:
  - "flicker" ↔ "exile target creature you control, then return it to the battlefield"
  - "ETB" ↔ "enters the battlefield"
  - "dies trigger" ↔ "when this creature dies"
- Use sentence-transformers `InputExample` format
- Split 80/20 train/validation

**Fine-tuning config:**
- Loss: `MultipleNegativesRankingLoss`
- Epochs: 3-5
- Warmup steps: 10% of training steps
- Evaluation: compute validation loss every epoch

---

## Performance Targets

### Latency Budgets
- Data download: <2 minutes
- Preprocessing 27k cards: <1 minute
- Embedding generation (CPU): <10 minutes
- FAISS index build: <30 seconds
- Single query search: <100ms
- Batch queries (100): <5 seconds

### Quality Metrics
**Baseline (pretrained model):**
- Precision@5: ≥50%
- Precision@10: ≥40%
- Query latency: <100ms

**Post fine-tuning targets:**
- Precision@5: ≥70% (+20 points)
- Precision@10: ≥60% (+20 points)
- Mean Reciprocal Rank: ≥0.7

**If targets not met:**
- Analyze failure cases by query type
- Try different embedding models (all-mpnet-base-v2)
- Implement query expansion or hybrid BM25+semantic search
- Document limitations in final report

---

## Troubleshooting Patterns

### Common Issues and Solutions

**"Out of memory during embedding generation"**
- Reduce batch_size to 16 or 8
- Process in chunks, save intermediate results
- Use `torch.cuda.empty_cache()` if using GPU

**"FAISS search returns poor results"**
- Check if embeddings are normalized (for cosine similarity)
- Verify query is embedded with same model as cards
- Inspect top-10 results manually - are they semantically related?
- Try different index types (IndexFlatL2 for L2 distance)

**"Fine-tuning doesn't improve results"**
- Check training loss - is it decreasing?
- Verify dataset format (InputExample with positive pairs)
- Increase training epochs or dataset size
- Try different loss function (ContrastiveLoss, TripletLoss)

**"Scryfall API rate limit exceeded"**
- Use bulk data endpoint (no rate limit)
- Implement exponential backoff for card image fetches
- Cache responses locally

### Debug Commands
```bash
# Check file sizes
ls -lh data/raw/
ls -lh data/processed/

# Verify embeddings shape
python -c "import numpy as np; e = np.load('data/processed/embeddings.npy'); print(e.shape)"

# Test FAISS index
python -c "import faiss; idx = faiss.read_index('index/mtg_faiss.index'); print(idx.ntotal)"

# Profile code
python -m line_profiler script.py
```

---

## Decision-Making Framework

### When to Ask vs. Implement

**Implement directly (default):**
- Standard patterns (data loading, model inference)
- Performance optimizations (batching, caching)
- Error handling and logging
- Code refactoring for clarity

**Ask first:**
- Major architectural changes (switching from FAISS to Weaviate)
- Dataset modifications (excluding certain card types)
- Evaluation methodology (which metrics to use)
- Timeline adjustments (skipping planned features)

### Trade-off Priorities
**Optimize for (in order):**
1. **Correctness:** Results are semantically relevant
2. **Reproducibility:** Same code, same results
3. **Maintainability:** Code is readable and documented
4. **Performance:** Meets latency targets
5. **Features:** Nice-to-haves (UI polish, advanced filters)

**If time-constrained:**
- Core search (weeks 1-6) is non-negotiable
- Fine-tuning (weeks 7-9) is high priority
- UI/demo (week 12) can be minimal
- Advanced features (metadata filters) are stretch goals

---

## Deliverables Checklist

### Week 8 MVP
- [ ] `download_scryfall.py` fetches latest oracle-cards.json
- [ ] `preprocess.py` generates cards_clean.csv (27k rows)
- [ ] `embed.py` creates embeddings.npy (384 dims)
- [ ] `build_index.py` builds FAISS index
- [ ] `search.py` CLI takes query, returns top-10 cards
- [ ] Baseline evaluation: Precision@5 ≥50% on 10 test queries

### Week 12 Complete System
- [ ] Fine-tuned model with Precision@5 ≥70%
- [ ] Evaluation report: baseline vs. fine-tuned comparison
- [ ] Streamlit demo (optional but recommended)
- [ ] GitHub repository with README and setup instructions

### Week 14 Final Submission
- [ ] Final report (15-20 pages) covering:
  - Problem statement and motivation
  - Technical approach (architecture, models, data)
  - Experiments and evaluation
  - Results with visualizations
  - Discussion of challenges and limitations
  - Future work
- [ ] Presentation slides (20 minutes)
- [ ] Clean, documented codebase with tests
- [ ] Demo video (5 minutes)

---

## Anti-Patterns to Avoid

### Code Smells
- Loading entire dataset into memory when processing can be batched
- Recomputing embeddings on every run (cache them!)
- Hardcoded paths (use config.yaml or environment variables)
- Silent failures (always log errors)
- No progress indication for long operations

### Project Management
- Waiting until week 13 to start documentation
- Skipping validation on small datasets before full run
- Not committing frequently (commit at least weekly)
- Implementing features before core pipeline works
- Optimizing prematurely (profile first)

### ML/AI Specific
- Not normalizing embeddings for cosine similarity
- Training on full dataset without train/val split
- Ignoring validation loss (sign of overfitting)
- Not testing on held-out queries
- Comparing models without consistent evaluation

---

## Extended Thinking Mode

Use **extended thinking** for:
- Complex debugging (multi-step failures)
- Architectural decisions (index type selection)
- Performance optimization strategies
- Fine-tuning hyperparameter selection

**Trigger phrases:**
- "think" - basic extended thinking
- "think hard" - deeper analysis
- "think harder" - thorough investigation
- "ultrathink" - maximum reasoning depth

**Example:** "think hard about why fine-tuning might not be improving precision on synonym queries"

---

## Quick Reference

### Essential Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Data pipeline
python src/download_scryfall.py
python src/preprocess.py
python src/embed.py
python src/build_index.py

# Search
python src/search.py "red creatures with flying under 3 mana"

# Evaluation
python src/evaluate.py --test-file data/test/test_queries.json

# Fine-tuning
python src/finetune.py --data data/processed/synonym_pairs.csv --epochs 5

# UI
streamlit run app.py
```

### File Structure
```
mtg-semantic-search/
├── data/
│   ├── raw/oracle-cards-YYYYMMDD.json
│   ├── processed/cards_clean.csv, embeddings.npy
│   └── test/test_queries.json
├── src/
│   ├── download_scryfall.py, preprocess.py, embed.py
│   ├── build_index.py, search.py, finetune.py, evaluate.py
├── models/baseline/, fine_tuned/
├── index/mtg_faiss.index, card_metadata.pkl
├── notebooks/01-04_*.ipynb
└── requirements.txt, .gitignore, README.md
```

### Key Dependencies
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
tqdm>=4.65.0
loguru>=0.7.0
pytest>=7.3.0
streamlit>=1.25.0
```

---

## Success Criteria

**This project is successful if:**
1. Semantic search outperforms keyword search on synonym queries (e.g., "flicker" finds "exile and return" cards)
2. Fine-tuning demonstrably improves precision (≥10 percentage points)
3. System is fast enough for interactive use (<100ms per query)
4. Code is clean, tested, and reproducible
5. Final report clearly communicates methodology and results

**This project is exceptional if:**
1. Precision@5 reaches ≥80%
2. Advanced features implemented (metadata filtering, query parser)
3. Deployed demo accessible via web URL
4. Published to GitHub with comprehensive documentation
5. Presentation includes live demo and rigorous evaluation

---

## Context for Claude Code

### How to Use This Guide
This document serves as your primary reference for development decisions. When uncertain:
1. Check this guide for established patterns
2. Refer to project status in outline-summary.md
3. Make pragmatic decisions aligned with timeline priorities
4. Flag deviations from plan for discussion

### Working with Me (Student)
- I'm familiar with Python, NLP basics, and MTG domain
- I prefer working code over explanations
- Point out potential issues proactively (don't wait for failures)
- Suggest optimizations but implement conservatively (correctness first)
- When debugging, show diagnostic steps (print shapes, inspect samples)

### Project Philosophy
**"Make it work, make it right, make it fast" - in that order.**

Build the simplest version that works, validate it thoroughly, then optimize. Don't gold-plate features before core functionality is proven. Defer stretch goals until MVP is complete. Document limitations honestly.

---

**Document Version:** 1.0
**Created:** November 3, 2025
**For:** Spring 2026 Independent Study - MTG Semantic Search
