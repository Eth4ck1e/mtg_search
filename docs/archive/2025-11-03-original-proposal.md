> **Archived 2026-05-17.** This is the original written proposal for the independent study, dated November 3, 2025. It predates the redesign that followed the POC. Technical decisions described below (FAISS, fine-tuning-first, sentence-transformers/all-MiniLM-L6-v2, 14-week Spring 2026 schedule) have been superseded by the architecture in `/CLAUDE.md` and the phased schedule in `/docs/roadmap/`. Preserved here for the paper's Background / Motivation section and as a record of how the project's scope evolved.

---

# Independent Study Project Outline: Semantic Search for MTG Cards
**Student:** Mitchell Trafford
**Project:** Building a Semantic Search System for Magic: The Gathering Cards Using Vector Embeddings
**Date:** November 3, 2025
**Semester:** Fall 2025 (Planning Phase) / Spring 2026 (Implementation Phase)

---

## Executive Summary

This independent study aims to develop a semantic search system for Magic: The Gathering (MTG) cards that transcends traditional keyword-based search by leveraging vector embeddings and natural language processing. The system will enable intuitive queries like "Find red creatures with flying under 3 mana" or "cards that flicker creatures" by understanding semantic relationships between card abilities, even when described using different terminology across MTG's 30+ year history.

**Key Value Proposition:**
- **Semantic Understanding**: Bridges terminology gaps (e.g., "flicker" = "exile and return" = "blink")
- **Natural Language Interface**: Players can search using colloquial MTG terms
- **Temporal Consistency**: Links modern and legacy card text syntaxes
- **Scalable Architecture**: ~27,000 cards, extensible to larger datasets

**Current Reality Check:**
Despite the detailed summary provided, the project directory is currently **empty**. The document appears to be a comprehensive proposal rather than a progress report. This outline reconciles the proposed plan with the actual current state (no implementation) and provides a realistic roadmap for Spring 2026 execution.

---

## Current Project Status

### What Actually Exists
- **Project Directory**: Created but empty
- **Planning Documentation**: Comprehensive technical plan from Grok analysis
- **Research Phase**: Conceptual architecture and tool selection completed

### What Needs to Be Built (Everything)
The entire implementation is pending. The project is currently in the **Planning/Design Phase** with no code, data, or infrastructure in place.

### Discrepancy Analysis
The summary from Grok lists weeks 1-10 as "complete," but examination of the project directory shows this is aspirational. The timeline appears to be a **proposed schedule** rather than historical progress. This is critical for setting realistic expectations for Spring 2026.

---

## Technical Architecture

### System Components

#### 1. Data Layer
**Primary Data Source:** Scryfall API (oracle-cards.json bulk data)
- **Size**: ~27,000 unique cards, ~50-100MB JSON
- **Update Frequency**: Daily
- **Key Fields**:
  - `oracle_text` (canonical card text)
  - `name`, `type_line`, `mana_cost`
  - `colors`, `cmc` (converted mana cost)
  - `card_faces` (for double-faced cards)
  - `legalities` (format legality)

**Data Challenges:**
- Double-faced cards (DFCs): Modal DFCs, transforming cards
- Split/Adventure cards: Multiple card faces with different effects
- Reminder text: Parenthetical explanations (stripped by oracle_text)
- Historical syntax evolution: "Bury" → "Destroy, can't regenerate"

#### 2. Embedding Model
**Selected Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Specifications:**
- **Dimensions**: 384
- **Parameters**: 22M (lightweight, CPU-friendly)
- **Trained On**: 1B+ sentence pairs (semantic similarity)
- **Performance**: Fast inference (~1000 sentences/sec on CPU)
- **Fine-tuning**: Supported via sentence-transformers library

**Why This Model:**
- Optimized for short text (MTG card text averages 20-50 words)
- Pre-trained on semantic similarity tasks
- Small enough for local development
- Strong baseline performance on domain-specific text

**Alternatives Considered:**
- `multi-qa-MiniLM-L6-cos-v1`: Better for Q&A but slower
- `all-mpnet-base-v2`: Higher quality (768-dim) but 3x slower
- OpenAI `text-embedding-3-small`: API-dependent, cost concerns
- Domain-specific fine-tuning of BERT-base: Computationally expensive

#### 3. Vector Database
**Selected:** FAISS (Facebook AI Similarity Search)

**Configuration:**
- **Index Type**: `IndexFlatIP` (inner product, exact search)
- **Storage**: Local filesystem
- **Size Estimate**: ~10MB for 27k vectors @ 384-dim
- **Query Time**: <10ms for exact search on CPU

**Why FAISS:**
- Zero cost (vs. Pinecone, Weaviate)
- CPU/GPU compatible
- Excellent for <100k vectors
- Simple Python integration
- Supports exact and approximate search

**Future Scalability Options:**
- `IndexIVFFlat`: Inverted file index for 100k-1M vectors
- `IndexHNSWFlat`: Graph-based index for very fast ANN search
- Hybrid approach: FAISS + metadata filtering via pandas

#### 4. Framework Stack
```
Python 3.9+
├── PyTorch 2.0+ (model inference, fine-tuning)
├── Transformers 4.30+ (Hugging Face model hub)
├── sentence-transformers 2.2+ (embedding utilities)
├── faiss-cpu 1.7+ (vector search)
├── pandas 2.0+ (data processing)
├── numpy 1.24+ (numerical operations)
├── requests (Scryfall API)
└── streamlit 1.25+ (optional UI)
```

**Development Tools:**
- `tqdm`: Progress bars for batch operations
- `loguru`: Structured logging
- `pytest`: Unit testing
- `black`: Code formatting
- `line_profiler`: Performance profiling

---

## Proposed File Structure

```
mtg-semantic-search/
├── README.md                          # Project overview, setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignore data/, models/, index/
├── config.yaml                        # Configuration (paths, model name, etc.)
│
├── data/
│   ├── raw/
│   │   └── oracle-cards-YYYYMMDD.json        # Scryfall bulk download
│   ├── processed/
│   │   ├── cards_clean.csv                   # Preprocessed cards
│   │   ├── keyword_freq.csv                  # Keyword frequency analysis
│   │   └── synonym_pairs.csv                 # Fine-tuning dataset
│   └── test/
│       └── test_queries.json                 # Evaluation queries
│
├── src/
│   ├── __init__.py
│   ├── download_scryfall.py                  # Fetch latest Scryfall data
│   ├── preprocess.py                         # Clean and normalize cards
│   ├── embed.py                              # Generate embeddings
│   ├── build_index.py                        # Create FAISS index
│   ├── search.py                             # Search interface
│   ├── finetune.py                           # Model fine-tuning
│   └── evaluate.py                           # Evaluation metrics
│
├── models/
│   ├── baseline/                             # Original all-MiniLM-L6-v2
│   └── fine_tuned/                           # Fine-tuned checkpoints
│
├── index/
│   ├── mtg_faiss.index                       # FAISS vector index
│   └── card_metadata.pkl                     # Card IDs, names, metadata
│
├── notebooks/
│   ├── 01_data_exploration.ipynb             # EDA on Scryfall data
│   ├── 02_embedding_analysis.ipynb           # Visualize embeddings
│   ├── 03_fine_tuning_experiments.ipynb      # Track fine-tuning
│   └── 04_evaluation.ipynb                   # Results analysis
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_embedding.py
│   └── test_search.py
│
└── scripts/
    ├── run_pipeline.sh                       # End-to-end data → index
    └── update_weekly.sh                      # Refresh Scryfall data
```

---

## Completed Work

**Current Status: None (Project in Planning Phase)**

The following are proposed milestones that appear as "complete" in the Grok summary but have not been implemented:

### Phase 0: Environment Setup (Proposed Week 1-2)
- [ ] Create project directory structure
- [ ] Set up Python virtual environment
- [ ] Install dependencies (PyTorch, transformers, FAISS)
- [ ] Configure version control (Git)
- [ ] Download initial Scryfall dataset

### Phase 1: Data Pipeline (Proposed Week 3-4)
- [ ] Implement `download_scryfall.py`
- [ ] Parse oracle-cards.json
- [ ] Handle edge cases (DFCs, split cards, tokens)
- [ ] Generate keyword frequency analysis
- [ ] Export to `cards_clean.csv`

### Phase 2: Baseline Embeddings (Proposed Week 5-6)
- [ ] Load `all-MiniLM-L6-v2` model
- [ ] Batch encode card texts
- [ ] Store embeddings (numpy array or HDF5)
- [ ] Dimensionality analysis

### Phase 3: FAISS Index (Proposed Week 7-8)
- [ ] Build `IndexFlatIP` from embeddings
- [ ] Implement basic search function
- [ ] CLI interface for queries
- [ ] Benchmark search latency

### Phase 4: Initial Evaluation (Proposed Week 9-10)
- [ ] Create 20 test queries (flicker, draw, counter, etc.)
- [ ] Measure precision@5, recall@10
- [ ] Identify failure cases

**Reality Check:** None of these phases have been implemented. The project is starting from scratch.

---

## Remaining Work (Prioritized for Spring 2026)

### Semester Timeline (14 Weeks)

#### Weeks 1-2: Foundation (Environment & Data)
**Priority: Critical**

**Tasks:**
1. Set up development environment
   - Install Python 3.9+, create virtual environment
   - Install PyTorch, transformers, FAISS, pandas
   - Configure IDE (VS Code recommended)
   - Initialize Git repository

2. Implement data pipeline
   - Write `download_scryfall.py` (fetch oracle-cards.json)
   - Write `preprocess.py` (parse, clean, handle edge cases)
   - Generate `cards_clean.csv` with fields: id, name, oracle_text, cmc, colors, types

3. Exploratory Data Analysis
   - Create `01_data_exploration.ipynb`
   - Analyze text length distribution
   - Identify most common keywords
   - Document edge cases (DFCs, split cards)

**Deliverables:**
- Working data download script
- Clean CSV with 27k cards
- EDA notebook with visualizations

**Estimated Effort:** 15-20 hours

---

#### Weeks 3-4: Baseline Embedding System
**Priority: Critical**

**Tasks:**
1. Implement embedding generation
   - Write `embed.py` (load model, batch encode)
   - Generate embeddings for all cards
   - Save to `data/processed/embeddings.npy`

2. Build FAISS index
   - Write `build_index.py` (create IndexFlatIP)
   - Store index and metadata separately
   - Test index persistence (save/load)

3. Basic search interface
   - Write `search.py` (query → top-k results)
   - CLI tool: `python search.py "red creatures with flying"`
   - Display: score, name, mana cost, oracle text

**Deliverables:**
- Embedding generation pipeline
- FAISS index (~10MB)
- Working CLI search tool

**Estimated Effort:** 20-25 hours

---

#### Weeks 5-6: Evaluation Framework
**Priority: High**

**Tasks:**
1. Create test dataset
   - 20 queries covering common search patterns:
     - Ability-based: "flicker", "card draw", "counterspells"
     - Color/cost: "red creatures under 3 mana"
     - Combo pieces: "infinite mana", "mill win conditions"
   - Manually label top-10 expected results per query

2. Implement evaluation metrics
   - Write `evaluate.py`:
     - Precision@K (K=5, 10)
     - Recall@K
     - Mean Reciprocal Rank (MRR)
   - Run baseline evaluation
   - Document failure cases

3. Analysis notebook
   - Create `04_evaluation.ipynb`
   - Visualize precision curves
   - Error analysis: Why do queries fail?

**Deliverables:**
- `test_queries.json` with ground truth
- Evaluation script
- Baseline metrics report

**Estimated Effort:** 15-20 hours

---

#### Weeks 7-9: Model Fine-Tuning
**Priority: High**

**Tasks:**
1. Create fine-tuning dataset
   - **Approach A**: Manual synonym pairs
     - "flicker" ↔ "exile target creature you control, then return it"
     - "ETB" ↔ "enters the battlefield"
     - "dies trigger" ↔ "when this creature dies"
   - **Approach B**: Synthetic data via GPT-4
     - Generate query-card pairs for common patterns
     - Use LLM to expand MTG slang → formal oracle text
   - Target: 500-1000 sentence pairs

2. Fine-tune model
   - Write `finetune.py` using sentence-transformers
   - Loss function: MultipleNegativesRankingLoss
   - Train for 3-5 epochs
   - Validation split (80/20)

3. Compare baseline vs. fine-tuned
   - Re-run evaluation on test set
   - Measure improvement in precision@5
   - Analyze which query types improved most

**Deliverables:**
- `synonym_pairs.csv` (training data)
- Fine-tuned model checkpoint
- Comparison report (baseline vs. fine-tuned)

**Estimated Effort:** 25-30 hours

---

#### Weeks 10-11: Metadata Filtering
**Priority: Medium**

**Tasks:**
1. Extend search with filters
   - Modify `search.py` to accept:
     - `colors` (e.g., "R", "WU")
     - `cmc_max`, `cmc_min`
     - `types` (e.g., "Creature", "Instant")
     - `legalities` (e.g., "commander")
   - Hybrid approach: FAISS for semantic → pandas for filters

2. Implement query parser
   - Parse: "red creatures with flying under 3 mana"
   - Extract: colors=[R], types=[Creature], cmc_max=3, query="flying"
   - Use spaCy or regex for basic NER

3. Test complex queries
   - "cheap blue counterspells"
   - "green ramp in commander"
   - "legendary artifacts that draw cards"

**Deliverables:**
- Enhanced search with metadata filters
- Query parser module
- 10 complex test queries

**Estimated Effort:** 20-25 hours

---

#### Week 12: User Interface
**Priority: Medium-Low**

**Tasks:**
1. Build Streamlit app
   - Input: Text query, filter dropdowns
   - Output: Top-10 cards with images (via Scryfall API)
   - Display: Name, mana cost, type, oracle text, similarity score

2. Deploy locally
   - Run: `streamlit run app.py`
   - Test on multiple queries
   - Gather feedback (from peers, professor)

**Deliverables:**
- Streamlit app (`app.py`)
- Demo video (5 minutes)

**Estimated Effort:** 10-15 hours

---

#### Weeks 13-14: Documentation & Presentation
**Priority: Critical**

**Tasks:**
1. Final report
   - Introduction (motivation, problem statement)
   - Technical approach (architecture, models, data)
   - Experiments (baseline, fine-tuning, evaluation)
   - Results (metrics, visualizations, case studies)
   - Discussion (challenges, limitations, future work)
   - Conclusion
   - Target: 15-20 pages

2. Presentation
   - 20-minute talk with slides
   - Live demo of search system
   - Q&A preparation

3. Code cleanup
   - Add docstrings, type hints
   - Write unit tests (pytest)
   - README with setup instructions
   - Push to GitHub

**Deliverables:**
- Final report (PDF)
- Presentation slides
- Clean, documented codebase
- GitHub repository

**Estimated Effort:** 20-25 hours

---

### Total Estimated Effort: 145-185 hours (10-13 hours/week)

---

## Key Challenges and Proposed Solutions

### Challenge 1: Inconsistent Ability Phrasing
**Problem:** MTG uses informal terms ("flicker", "blink") for mechanics described differently in oracle text ("exile, then return", "blink target creature").

**Impact:** Baseline model may not cluster semantically similar cards.

**Solutions:**
1. **Fine-tuning on synonym pairs** (Primary)
   - Create dataset: informal term ↔ oracle text
   - Example: "flicker" ↔ "exile target creature you control, then return it to the battlefield"
   - Use sentence-transformers fine-tuning with triplet loss

2. **Query expansion** (Secondary)
   - Expand "flicker" → ["flicker", "exile return", "blink", "cloudshift"]
   - Search multiple embeddings, merge results

3. **MTG glossary integration** (Tertiary)
   - Augment card text with keyword definitions from MTG Comprehensive Rules
   - Embed: oracle_text + " [Keywords: flicker, ETB]"

**Recommendation:** Implement solution 1 first (weeks 7-9), evaluate improvement, consider solution 2 if precision remains <70%.

---

### Challenge 2: Double-Faced Cards (DFCs)
**Problem:** Cards like "Delver of Secrets // Insectile Aberration" have two faces with different text.

**Impact:** How to represent in vector space? One embedding or two?

**Solutions:**
1. **Concatenate both faces** (Recommended)
   - Embed: `card_faces[0]['oracle_text'] + " // " + card_faces[1]['oracle_text']`
   - Pro: Single embedding, captures full card identity
   - Con: May dilute semantic signal if faces are unrelated

2. **Separate embeddings per face**
   - Create two entries: "Delver of Secrets" and "Insectile Aberration"
   - Pro: Precise matching per face
   - Con: Inflates index size, may confuse users

3. **Front face only**
   - Embed only `card_faces[0]`
   - Pro: Simplest
   - Con: Loses information about transformed state

**Recommendation:** Start with solution 1, evaluate on specific DFC test cases (e.g., "werewolf", "transform trigger").

---

### Challenge 3: Performance on CPU
**Problem:** Generating 27k embeddings and searching in real-time on CPU.

**Impact:** Slow embedding generation (~30 minutes), search latency.

**Solutions:**
1. **Batch encoding** (Critical)
   - Use `model.encode(texts, batch_size=32, show_progress_bar=True)`
   - Reduces overhead from ~1 hour to ~5 minutes on CPU

2. **Caching embeddings** (Critical)
   - Pre-compute and save embeddings, rebuild only on data updates
   - Use `numpy.save()` or HDF5 for efficient storage

3. **FAISS optimization**
   - Use `IndexFlatIP` for exact search (<10ms per query)
   - If scaling beyond 100k vectors, switch to `IndexIVFFlat` (approximate)

4. **Optional GPU acceleration**
   - If available: `model.to('cuda')`, `faiss-gpu`
   - Expected speedup: 10-50x for embedding generation

**Recommendation:** Implement solutions 1-2 immediately, solution 3 only if expanding dataset, solution 4 if CPU performance is unacceptable.

---

### Challenge 4: Evaluation Without Ground Truth
**Problem:** No existing labeled dataset for MTG semantic search.

**Impact:** Hard to measure "correctness" objectively.

**Solutions:**
1. **Manual labeling** (Primary)
   - Create 20 test queries with hand-labeled top-10 results
   - Use domain expertise (student plays MTG)
   - Validate with other players or online resources (EDHREC, MTGGoldfish)

2. **Community validation** (Secondary)
   - Post queries to MTG subreddit/Discord, ask for feedback
   - Use community consensus as ground truth

3. **Comparative evaluation** (Tertiary)
   - Compare against Scryfall text search
   - Measure: Does semantic search retrieve cards Scryfall misses?

**Recommendation:** Solution 1 for weeks 5-6, solution 2 for final validation, solution 3 for discussion section.

---

### Challenge 5: Reminder Text and Noise
**Problem:** Some cards have verbose reminder text or flavor text that may distort embeddings.

**Impact:** Example: "Flying (This creature can't be blocked except by creatures with flying or reach)" vs. just "Flying".

**Solutions:**
1. **Use oracle_text** (Already solved)
   - Scryfall's oracle_text already excludes reminder text in parentheses
   - Verify in preprocessing: strip any remaining parenthetical text

2. **Exclude flavor text** (Critical)
   - Do NOT use `flavor_text` field
   - Only embed `oracle_text`

**Recommendation:** Solution 1 is built into Scryfall data. Verify in `preprocess.py` that flavor text is not included.

---

### Challenge 6: Evolving Card Syntax
**Problem:** Old cards use obsolete keywords (e.g., "bury" → "destroy, can't regenerate").

**Impact:** Searches for "destroy permanently" may miss old cards with "bury".

**Solutions:**
1. **Oracle text already handles this** (Partial solution)
   - Scryfall oracle text is updated to modern syntax
   - Example: Old "bury" cards now say "destroy, can't be regenerated"

2. **Fine-tuning on historical pairs** (Additional solution)
   - Include pairs: "bury" ↔ "destroy, can't regenerate"
   - "Comes into play" ↔ "enters the battlefield"

**Recommendation:** Solution 1 should cover most cases. Add solution 2 to fine-tuning dataset if evaluation reveals gaps.

---

## Timeline Suggestions for Spring 2026

### Option 1: Standard 14-Week Semester (Recommended)
| Week | Phase | Focus | Deliverable |
|------|-------|-------|-------------|
| 1-2 | Foundation | Environment, data pipeline, EDA | Clean dataset, working scripts |
| 3-4 | Baseline | Embeddings, FAISS, CLI search | Working search system |
| 5-6 | Evaluation | Test queries, metrics, analysis | Baseline performance report |
| 7-9 | Fine-Tuning | Synonym dataset, training, comparison | Improved model |
| 10-11 | Enhancement | Metadata filters, query parser | Advanced search features |
| 12 | UI | Streamlit app, demo | Interactive demo |
| 13-14 | Finalization | Report, presentation, code cleanup | Final deliverables |

**Total Workload:** 145-185 hours (10-13 hours/week)
**Feasibility:** Appropriate for 3-credit independent study

---

### Option 2: Accelerated 10-Week Quarter
| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-2 | Foundation + Baseline | Working search system |
| 3-4 | Evaluation + Fine-Tuning | Improved model |
| 5-7 | Enhancements (metadata filtering) | Advanced search |
| 8-9 | UI + Documentation | Demo + draft report |
| 10 | Finalization | Final presentation |

**Total Workload:** 120-150 hours (12-15 hours/week)
**Trade-off:** Less experimentation, skip some optional features

---

### Option 3: Extended Timeline with Research Component
Add 4 additional weeks for research:
- **Weeks 1-4:** Literature review on semantic search, vector databases, domain-specific embeddings
- **Weeks 5-18:** Standard implementation (per Option 1)

**Deliverable:** Research paper-style final report with related work section
**Suitable for:** Graduate-level independent study or thesis

---

## Alternative Approaches and Improvements

### Alternative 1: Use OpenAI Embeddings Instead of Local Model
**Approach:** Replace `all-MiniLM-L6-v2` with `text-embedding-3-small` via OpenAI API.

**Pros:**
- Likely better out-of-box performance (trained on larger corpus)
- Higher dimensions (1536-dim) → richer representations
- No local GPU/CPU requirements for embedding generation

**Cons:**
- API costs: ~$0.13 per 1M tokens, ~27k cards × 50 tokens = 1.35M tokens = $0.18 one-time + $0.18 per weekly update
- API dependency (latency, rate limits)
- Cannot fine-tune on MTG-specific data
- Less educational value (black box)

**Recommendation:** Stick with local model for learning experience and fine-tuning flexibility. Consider OpenAI as benchmark comparison in evaluation phase.

---

### Alternative 2: Use ChromaDB or Weaviate Instead of FAISS
**Approach:** Replace FAISS with a managed vector database.

**Pros:**
- Built-in metadata filtering (no need for hybrid pandas approach)
- Easier API (`.query()` vs. manual FAISS operations)
- Supports CRUD operations (update/delete vectors)
- Cloud deployment options

**Cons:**
- Added complexity (Docker setup for Weaviate)
- Overkill for 27k vectors
- Less educational (abstracts vector search internals)

**Recommendation:** Use FAISS for initial implementation (simpler, faster for small scale). Migrate to ChromaDB/Weaviate in future work if scaling beyond 100k cards or adding real-time updates.

---

### Alternative 3: Multi-Stage Retrieval (Semantic + BM25)
**Approach:** Combine vector search with traditional keyword search.

**Implementation:**
1. **Stage 1 (Broad Retrieval):** BM25 retrieves top-100 candidates based on keyword overlap
2. **Stage 2 (Reranking):** Semantic search reranks top-100 using embeddings

**Pros:**
- Ensures keyword matches aren't missed
- Common pattern in production search (e.g., Elasticsearch + BERT)
- May improve recall on rare abilities

**Cons:**
- Increased complexity
- Requires implementing BM25 (use rank-bm25 library)
- Slower (two-stage pipeline)

**Recommendation:** Consider for "Improvements" section in final report if semantic-only search shows low recall on keyword-heavy queries.

---

### Alternative 4: Use Sentence-BERT with Cross-Encoder for Reranking
**Approach:** Use bi-encoder (all-MiniLM) for fast retrieval, then cross-encoder (e.g., `ms-marco-MiniLM-L-12-v2`) for reranking top-10.

**Pros:**
- Cross-encoders are more accurate (jointly encode query + card)
- Best of both: speed (bi-encoder) + precision (cross-encoder)

**Cons:**
- Added complexity
- Cross-encoder is 10-100x slower (acceptable for reranking only top-k)

**Recommendation:** Advanced enhancement for weeks 10-11 if baseline precision <60%. Defer to future work if time-constrained.

---

### Alternative 5: Add Image Embeddings (CLIP)
**Approach:** Embed card images using CLIP, enable visual search ("Find cards with art showing dragons").

**Pros:**
- Unique feature (no existing MTG search supports this)
- Educational value (multimodal learning)
- Scryfall provides image URLs

**Cons:**
- Significant scope expansion
- Image download/storage (~27k images × ~200KB = ~5GB)
- CLIP embeddings (512-dim) need separate index or concatenation

**Recommendation:** Out of scope for Spring 2026. Mention in "Future Work" section. Excellent idea for summer research or follow-up project.

---

### Alternative 6: Build a Chrome Extension for Scryfall.com
**Approach:** Instead of standalone app, integrate semantic search directly into Scryfall as a browser extension.

**Pros:**
- Better UX (users already use Scryfall)
- Real-world deployment
- Showcases full-stack skills (Python backend + JavaScript frontend)

**Cons:**
- Requires hosting backend (FastAPI on Heroku/Railway)
- Chrome extension development learning curve
- Scryfall API rate limits (75 requests per second)

**Recommendation:** Ambitious but feasible. Consider for weeks 12-14 if Streamlit demo is completed early. Otherwise, propose for future work.

---

## Resources Needed

### Computational Resources
**Minimum Requirements:**
- **CPU:** 4+ cores (Intel i5/Ryzen 5 or better)
- **RAM:** 8GB (16GB recommended for large batch encoding)
- **Storage:** 10GB (dataset, models, embeddings, index)
- **OS:** macOS, Linux, or Windows with WSL2

**Optional (Accelerated Development):**
- **GPU:** NVIDIA GPU with CUDA support (even entry-level GTX 1650 provides 10x speedup for embeddings)
- **Cloud Credits:** Google Colab Pro ($10/month) or AWS/GCP credits for experimentation

**Current Setup:** macOS system detected, should be sufficient for CPU-based development.

---

### Software & Libraries
**Core Dependencies:**
```
python>=3.9
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4  # or faiss-gpu if CUDA available
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
```

**Development Tools:**
```
jupyter>=1.0.0
tqdm>=4.65.0
loguru>=0.7.0
pytest>=7.3.0
black>=23.0.0
streamlit>=1.25.0  # for UI
```

**Installation:**
```bash
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install faiss-cpu  # or faiss-gpu
pip install pandas numpy requests tqdm loguru
pip install jupyter streamlit pytest black
```

---

### Data Resources
**Primary:**
- **Scryfall Bulk Data:** https://scryfall.com/docs/api/bulk-data
  - Endpoint: `https://api.scryfall.com/bulk-data/oracle-cards`
  - Format: JSON (~50-100MB)
  - License: Public domain-ish (Scryfall Terms of Service)
  - Update: Daily (fetch weekly for development)

**Secondary (Fine-Tuning):**
- **MTG Glossary:** MTG Wiki (https://mtg.fandom.com/wiki/List_of_Magic_slang)
- **Synonym Sources:**
  - EDHREC (https://edhrec.com) for common search terms
  - MTGGoldfish (https://www.mtggoldfish.com) for deck archetypes
  - Manual curation (student's domain knowledge)

---

### Learning Resources
**Vector Embeddings & Search:**
- *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"* (Reimers & Gurevych, 2019)
- FAISS documentation: https://github.com/facebookresearch/faiss/wiki
- Pinecone Learning Center: https://www.pinecone.io/learn/ (conceptual, not tool-specific)

**Fine-Tuning:**
- Sentence-Transformers documentation: https://www.sbert.net/docs/training/overview.html
- Hugging Face course (fine-tuning chapter): https://huggingface.co/course/chapter3

**MTG Domain Knowledge:**
- MTG Comprehensive Rules: https://magic.wizards.com/en/rules
- Scryfall API docs: https://scryfall.com/docs/api

---

### Academic Support
**Recommended:**
- **Weekly meetings** with advisor (30 minutes)
  - Progress updates, unblock technical issues
  - Feedback on experimental design
- **Mid-semester checkpoint** (week 7)
  - Review baseline system, adjust timeline if needed
- **Peer review** (weeks 10-12)
  - Have 2-3 MTG players test search system, provide feedback

---

## Evaluation Metrics for Success

### Quantitative Metrics

#### 1. Search Relevance (Primary)
**Precision@K:** Of top-K results, how many are relevant?
- **Target:** Precision@5 ≥ 70%, Precision@10 ≥ 60%
- **Baseline Expectation:** ~50-60% (pretrained model)
- **Post-Fine-Tuning:** ~70-80%

**Recall@K:** Of all relevant cards, how many are in top-K?
- **Target:** Recall@10 ≥ 75%
- **Challenge:** Requires complete labeling of relevant cards per query

**Mean Reciprocal Rank (MRR):** Average of 1/rank of first relevant result
- **Target:** MRR ≥ 0.7
- **Interpretation:** First relevant result typically in top-2

#### 2. Search Performance
**Query Latency:** Time from query to results
- **Target:** <100ms for top-10 results (FAISS on CPU)
- **Measure:** Average over 100 queries

**Index Build Time:** Time to rebuild FAISS index
- **Target:** <5 minutes for 27k cards on CPU

**Embedding Generation Time:** Time to encode all cards
- **Target:** <10 minutes on CPU (with batching)

#### 3. Fine-Tuning Improvement
**Delta Precision:** Improvement from baseline to fine-tuned
- **Target:** +10-15 percentage points on Precision@5
- **Example:** Baseline 55% → Fine-tuned 70%

**Query Category Breakdown:** Which query types improved most?
- Example: "Flicker" queries: 40% → 85% (strong improvement)
- Example: "Card draw" queries: 65% → 70% (marginal)

---

### Qualitative Metrics

#### 1. User Feedback (Weeks 12-13)
**Test with 3-5 MTG players:**
- **Survey Questions:**
  1. "Does the system find cards you expected?" (1-5 Likert scale)
  2. "Are results better than Scryfall text search?" (Yes/No/Same)
  3. "What queries failed or gave unexpected results?" (Open-ended)

**Target:** Average satisfaction ≥ 4/5

#### 2. Failure Case Analysis
**Document 5-10 queries that fail:**
- What did system return?
- What should it have returned?
- Why did it fail? (embedding similarity, missing metadata, etc.)

**Example:**
- Query: "cheap removal in black"
- Expected: Terror, Doom Blade, Fatal Push
- Actual: Dark Ritual, Thoughtseize, Entomb
- Diagnosis: Model misunderstood "removal" → returned black cards with "remove" in text (e.g., "remove from graveyard")

#### 3. Edge Case Coverage
**Test special card types:**
- Double-faced cards (e.g., "werewolves")
- Split cards (e.g., "Fire // Ice")
- Historical syntax (e.g., "bury creature")
- Slang terms (e.g., "ETB triggers", "dies triggers", "mana dorks")

**Target:** ≥70% success rate on 20 edge case queries

---

### Comparison Benchmarks

#### Baseline 1: Scryfall Text Search (Keyword-Based)
**Method:** Query Scryfall API's `/cards/search?q=o:flicker` endpoint
- **Expectation:** High precision for exact keywords, low recall for synonyms
- **Comparison:** Semantic search should have higher recall on synonym queries

#### Baseline 2: Random Retrieval
**Method:** Return 10 random cards
- **Expectation:** ~0% precision (sanity check)

#### Baseline 3: TF-IDF + Cosine Similarity
**Method:** Classic IR approach without neural embeddings
- **Expectation:** Better than random, worse than neural embeddings
- **Implementation:** Use scikit-learn's TfidfVectorizer
- **Comparison:** Demonstrates value of semantic embeddings

---

### Success Criteria by Phase

**Minimum Viable Product (MVP) - Week 8:**
- [ ] System retrieves top-10 results for any text query in <100ms
- [ ] Precision@5 ≥ 50% on 10 test queries (baseline)
- [ ] All 27k cards indexed and searchable

**Successful Project - Week 12:**
- [ ] Fine-tuned model with Precision@5 ≥ 70%
- [ ] Metadata filtering functional (color, CMC, type)
- [ ] Streamlit demo deployed
- [ ] 3/5 user testers rate system 4+ stars

**Exceptional Project - Week 14:**
- [ ] Precision@5 ≥ 80% on broad test set (30+ queries)
- [ ] Published to GitHub with full documentation
- [ ] Advanced features: query parser, cross-encoder reranking, or Chrome extension
- [ ] Presentation includes live demo and rigorous evaluation

---

## Risk Assessment and Mitigation

### High-Risk Items

**Risk 1: Fine-Tuning Doesn't Improve Performance**
- **Probability:** Medium (30%)
- **Impact:** High (core hypothesis)
- **Mitigation:**
  - Start fine-tuning early (week 7) to allow iteration
  - Try multiple training strategies (triplet loss, contrastive loss)
  - Have backup: Query expansion or hybrid BM25 approach
- **Fallback:** Focus on metadata filtering and UI, acknowledge limitation in report

**Risk 2: Data Quality Issues**
- **Probability:** Low-Medium (20%)
- **Impact:** Medium
- **Examples:** Scryfall API changes, missing oracle text, encoding errors
- **Mitigation:**
  - Validate data in week 1-2 (check for nulls, parse errors)
  - Version control raw data (save downloaded JSON with timestamp)
  - Implement robust error handling in preprocessing

**Risk 3: Performance Bottlenecks on CPU**
- **Probability:** Medium (30%)
- **Impact:** Low-Medium (affects user experience, not core functionality)
- **Mitigation:**
  - Profile code with line_profiler in week 3-4
  - Optimize batch sizes, use multiprocessing for embedding generation
  - If severe: Request GPU access (lab machines, Colab Pro)

---

### Medium-Risk Items

**Risk 4: Scope Creep**
- **Probability:** High (50%)
- **Impact:** Medium (delays core deliverables)
- **Mitigation:**
  - Strictly follow timeline, defer non-critical features
  - Maintain "must-have" vs. "nice-to-have" list
  - Weekly check-ins with advisor to reset priorities

**Risk 5: Insufficient Ground Truth for Evaluation**
- **Probability:** Medium (30%)
- **Impact:** Medium (hard to demonstrate success quantitatively)
- **Mitigation:**
  - Create test set early (week 5), validate with MTG community
  - Use multiple evaluators (inter-annotator agreement)
  - Supplement with qualitative analysis (case studies)

---

### Low-Risk Items

**Risk 6: Dependencies Break**
- **Probability:** Low (10%)
- **Impact:** Low (fixable with version pinning)
- **Mitigation:** Pin all versions in requirements.txt, use virtual environment

**Risk 7: Hardware Failure**
- **Probability:** Very Low (5%)
- **Impact:** High (data loss)
- **Mitigation:**
  - Git for code (push weekly)
  - Cloud backup for data (Google Drive, Dropbox)
  - Save intermediate results (embeddings, indices)

---

## Additional Recommendations

### Best Practices for Success

1. **Version Control from Day 1**
   - Initialize Git repository in week 1
   - Commit frequently (daily during active development)
   - Use meaningful commit messages
   - Push to GitHub weekly (public or private)

2. **Document as You Go**
   - Maintain a lab notebook (can be Markdown file)
   - Log experiments: date, hyperparameters, results
   - Screenshot interesting results
   - Write README incrementally (not at the end)

3. **Test Early and Often**
   - Write unit tests for preprocessing, embedding, search
   - Use pytest for automated testing
   - Test on small subset (100 cards) before full dataset

4. **Incremental Development**
   - Build simplest version first (hardcoded queries, no UI)
   - Add features iteratively
   - Always have a working system (main branch stable)

5. **Seek Feedback**
   - Share demo with advisor by week 8
   - User testing by week 12 (not week 14)
   - Incorporate feedback before final submission

---

### Stretch Goals (If Ahead of Schedule)

1. **Advanced Search Features**
   - Fuzzy card name matching (Levenshtein distance)
   - "More like this card" functionality
   - Search history and saved queries

2. **Deployment**
   - Deploy backend via FastAPI on Railway/Heroku
   - Public URL for demo (include in report)
   - API documentation with Swagger

3. **Evaluation Rigor**
   - A/B test: Baseline vs. fine-tuned (blind user study)
   - Statistical significance testing (paired t-test)
   - Error rate by card rarity, color, set

4. **Research Contribution**
   - Write short paper (4-6 pages) for conference (e.g., ACL Student Research Workshop)
   - Release fine-tuned model on Hugging Face Hub
   - Create blog post or tutorial

---

### Learning Outcomes

By completing this project, you will gain:

**Technical Skills:**
- **NLP:** Transformer models, embeddings, fine-tuning
- **Information Retrieval:** Vector search, FAISS, evaluation metrics
- **Data Engineering:** API integration, data cleaning, ETL pipelines
- **Software Engineering:** Python project structure, testing, documentation
- **MLOps:** Model versioning, experiment tracking, deployment

**Domain Expertise:**
- Deep understanding of semantic search systems
- Hands-on experience with vector databases
- Knowledge of fine-tuning strategies for domain-specific tasks

**Research Skills:**
- Experimental design (baseline, ablations, evaluation)
- Technical writing (report, documentation)
- Presentation and demo skills

---

## Conclusion

This independent study represents an excellent opportunity to build a practical, domain-specific AI system from scratch. The scope is ambitious but achievable within a 14-week semester with disciplined execution. The recommended timeline prioritizes **core functionality first** (weeks 1-6), **quality improvements second** (weeks 7-11), and **polish last** (weeks 12-14).

**Critical Success Factors:**
1. **Start immediately** with environment setup (week 1)
2. **Achieve MVP by week 8** (working baseline search)
3. **Allocate sufficient time** for fine-tuning (weeks 7-9)
4. **Iterate based on evaluation** (don't wait until week 13)
5. **Communicate regularly** with advisor (unblock issues early)

**Realistic Expectation:**
- This is a **from-scratch project** (current directory is empty)
- The Grok summary is a **proposal**, not a progress report
- Expect 10-13 hours/week of dedicated work
- Some features may need to be deferred to "Future Work"

**Key Differentiators:**
- **Semantic understanding** of MTG abilities (not just keyword matching)
- **Fine-tuning** on domain-specific data (demonstrates ML expertise)
- **End-to-end system** (data pipeline → model → search → UI)
- **Rigorous evaluation** (quantitative metrics + user testing)

**Final Recommendation:** Follow the 14-week timeline (Option 1) for a thorough, high-quality project. If time permits, pursue stretch goals in weeks 12-13. Defer advanced features (CLIP, Chrome extension) to summer research or follow-up project. Focus on **depth over breadth**: A well-executed core system with strong evaluation is more impressive than a feature-rich system with weak foundations.

Good luck, and enjoy building your semantic search system!

---

## Appendix: Quick Start Checklist

**Week 1, Day 1:**
- [ ] Create project directory: `mtg-semantic-search/`
- [ ] Initialize Git: `git init`
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Install dependencies: `pip install torch transformers sentence-transformers faiss-cpu pandas`
- [ ] Download Scryfall data: `curl -o data/raw/oracle-cards.json https://api.scryfall.com/bulk-data/oracle-cards`
- [ ] Create first notebook: `notebooks/01_data_exploration.ipynb`

**Week 1, Day 7:**
- [ ] `cards_clean.csv` generated with 27k rows
- [ ] EDA notebook complete (text length histograms, keyword frequency)
- [ ] First commit pushed to GitHub

**Week 4, Day 7:**
- [ ] CLI search working: `python search.py "red creatures with flying"`
- [ ] FAISS index saved: `index/mtg_faiss.index`
- [ ] Second commit with baseline system

**Week 8, Day 7:**
- [ ] MVP demo to advisor
- [ ] Baseline evaluation complete (Precision@5 documented)
- [ ] Go/no-go decision on fine-tuning

**Week 12, Day 7:**
- [ ] Streamlit app deployed locally
- [ ] User testing sessions scheduled
- [ ] Draft report outline complete

**Week 14, Day 7:**
- [ ] Final report submitted
- [ ] Presentation slides ready
- [ ] Code cleaned and pushed to GitHub
- [ ] Demo video recorded

---

**Document Version:** 1.0
**Last Updated:** November 3, 2025
**Author:** Mitchell Trafford (Student) + Claude (AI Assistant)
