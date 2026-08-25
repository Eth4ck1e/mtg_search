# MTG Vector Database Independent Study
## Claude Code Development Guide

**Project**: Semantic Search System for Magic: The Gathering Cards
**Timeline**: Spring 2026 (14 weeks, 10-13 hours/week)
**Student**: Mitchell Trafford
**Working Directory**: `/Users/mitchelltrafford/Documents/Development/Independent Study - MTG Vector DB`

---

## Project Identity

### Core Mission
Build a semantic search system that understands MTG card abilities beyond keyword matching, enabling natural language queries like "cards that flicker creatures" or "cheap red removal."

### Project Philosophy
**"Learn deeply, build pragmatically, iterate thoughtfully."**

This is a learning-first project with production-quality standards. Every implementation should deepen understanding while following professional engineering practices. Balance educational value with practical efficiency.

### Technical Stack
- **Data**: Scryfall API (27,000 MTG cards, ~50-100MB JSON)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: FAISS (local, CPU-optimized)
- **Fine-tuning**: Custom MTG terminology dataset
- **Optional UI**: Streamlit for demonstration
- **Environment**: macOS, Python 3.9+, PyTorch 2.0+

### Learning Objectives
- Hands-on experience with transformer models and embeddings
- Understanding of vector search systems and similarity metrics
- Fine-tuning strategies for domain-specific applications
- ML experiment design, tracking, and evaluation
- End-to-end ML system development (data → model → deployment)

---

## Communication & Workflow

### Communication Style
**Be direct and educational:**
- Lead with action: "I'll implement X because Y"
- Explain the "why" behind decisions, not just the "what"
- Show working code with teaching comments
- Flag blockers immediately with proposed solutions
- Balance efficiency with learning moments

**Minimize back-and-forth:**
- Ask clarifying questions upfront when requirements are ambiguous
- Make reasonable assumptions for minor decisions, document them
- Provide 2-3 concrete options for major decisions with recommendations
- Batch related questions together

**Default to implementation:**
- When asked "Can we do X?", respond with working code unless infeasible
- Include error handling in initial implementation
- Add progress bars (tqdm) for long operations
- Provide diagnostic steps when debugging

### Response Structure for Complex Tasks
1. **Overview**: Brief explanation of what we'll build and why
2. **Plan**: Numbered steps with rationale
3. **Implementation**: Code with inline teaching comments
4. **Validation**: How to verify it works
5. **Learning Moment**: Key takeaways or experiments to try
6. **Next Steps**: Suggested improvements or related features

### Example Response Pattern
```
I'll implement the embedding generation pipeline using batch processing.
Here's why this approach matters:

1. Memory efficiency: Loading all 27k cards at once would use ~2GB RAM
2. Speed: Batching reduces time from ~1 hour to ~5 minutes
3. Progress tracking: tqdm shows real-time progress

Alternative approaches you might experiment with:
- Single-card encoding (simpler but 10x slower)
- Parallel processing (faster but more complex)
- GPU acceleration if available

Let me implement with comments explaining each step...
[working code]

To verify it works: Check embeddings shape is (27000, 384) and
compute similarity between two known similar cards.

Key learning: Batch processing is fundamental to practical ML engineering.
The trade-off is memory vs. speed - adjust batch_size if you hit OOM errors.
```

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

### Code Quality Standards

**Every function should have:**
- Type hints for parameters and return values
- Google-style docstring with Args, Returns, and brief description
- Descriptive variable names (no single-letter except loop indices)
- Error handling with informative messages
- Teaching comments explaining key decisions

**Example function signature:**
```python
def embed_cards(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32
) -> np.ndarray:
    """Generate embeddings for MTG card texts using batched encoding.

    Batching is crucial for performance - processing 27k cards one at a time
    would take ~1 hour vs. ~5 minutes with batching. The trade-off is memory
    usage: larger batches are faster but use more RAM.

    Args:
        texts: List of card oracle texts to embed
        model_name: HuggingFace model identifier
        batch_size: Number of texts to encode per batch (reduce if OOM errors)

    Returns:
        numpy array of shape (n_texts, embedding_dim)

    Raises:
        RuntimeError: If model fails to load or encoding fails
    """
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate embeddings. Common causes:\n"
            f"1. Network issue downloading model\n"
            f"2. Out of memory (try batch_size=16)\n"
            f"3. Invalid text format\n"
            f"Original error: {e}"
        ) from e
```

### Git Usage
**Commit strategy:**
- Commit after completing each logical unit (feature, fix, refactor)
- Wait for explicit request to commit - don't be proactive
- Use conventional commit format: `type: description`
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Write commit messages that explain "why" not just "what"
- Include Co-Authored-By: Claude footer

**Example commit message:**
```
feat: add batch encoding for embeddings (10x speedup)

Implemented encode_cards() with configurable batch_size.
Reduces encoding time from ~3000s to ~287s for 27k cards.
Added progress bar with tqdm for user feedback.

This is crucial for iterative development - regenerating
embeddings needs to be fast enough for experimentation.

Co-Authored-By: Claude <noreply@anthropic.com>
```

**NEVER:**
- Force push to main/master (warn user if requested)
- Skip hooks (--no-verify) unless explicitly requested
- Commit secrets (.env, credentials) - flag these proactively
- Push without explicit request
- Amend commits from other developers

---

## Testing & Validation

### Incremental Development Philosophy
**Start small, validate often, then scale:**
1. Test with 10 cards to verify logic
2. Run on 100 cards to check performance
3. Process full 27k cards after validation
4. Profile with small datasets before optimizing

### Validation Checkpoints
**At each pipeline stage:**

1. **Data Download**: Check file size (~50-100MB), parse first 10 entries
2. **Preprocessing**: Verify no null oracle_text, inspect edge cases (DFCs, split cards)
3. **Embeddings**: Check shape (27000, 384), verify not all zeros, compute sample similarities
4. **FAISS Index**: Test search with known query, measure latency (<100ms target)
5. **Search Results**: Manual review of top-5 for 3-5 queries before full evaluation

### Error Handling Priorities
**Must gracefully handle:**
- API failures (Scryfall rate limits, timeouts)
- Missing/malformed data (null oracle_text, invalid JSON)
- Out-of-memory errors (batch size too large)
- File not found (index/model missing)

**Error handling pattern:**
```python
try:
    # Core logic
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Explain what went wrong and why
    # Provide recovery action or clear error message
    raise RuntimeError("Helpful message for debugging") from e
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

**Key decision**: How do we represent multi-faced cards in vector space?
- Options: concatenate text, separate embeddings, front-only
- Recommended: Start with concatenation (simplest), test alternatives
- Document: Track this decision in experiment log

**Preprocessing pipeline:**
```python
def preprocess_cards(raw_data: list[dict]) -> pd.DataFrame:
    """Clean and format Scryfall data for embedding generation.

    Key decisions:
    1. Concatenate double-faced cards (maintains card unity)
    2. Keep reminder text (helps model learn mechanics)
    3. Preserve case (helps with proper nouns)
    4. Skip tokens/art cards (not searchable cards)

    Args:
        raw_data: List of card dictionaries from Scryfall API

    Returns:
        DataFrame with columns: id, name, oracle_text, cmc, colors, types
    """
    # Implementation with teaching comments at each decision point
```

### Embedding Generation

**Optimization strategy:**
- Use `batch_size=32` as starting point (adjust based on RAM)
- Enable `show_progress_bar=True` for visibility
- Cache embeddings: save to `embeddings.npy`, rebuild only on data changes
- Normalize vectors for cosine similarity: `faiss.normalize_L2(embeddings)`

**Performance targets:**
- Embedding generation (CPU): <10 minutes for 27k cards
- Memory usage: <4GB RAM
- Cache hit: <1 second to load pre-computed embeddings

**Educational additions:**
- Show how to inspect embedding dimensions and distributions
- Suggest visualization (t-SNE/UMAP projection of 500 random cards)
- Explain why 384 dimensions is reasonable for this task
- Compare embedding similarities for known similar/dissimilar cards

### Vector Search with FAISS

**Implementation guidance:**
- Start with `IndexFlatIP` (inner product) for exact search
- Explain why: <100k vectors, exact search is fast enough (<100ms)
- Save index: `faiss.write_index(index, 'mtg_faiss.index')`
- Store metadata separately: `pickle.dump(metadata, 'card_metadata.pkl')`

**Key teaching moment:**
"FAISS gives us the top-k most similar embeddings, but similarity doesn't always equal relevance. This is why we'll combine vector search with metadata filters (color, mana cost) and evaluate with precision/recall metrics."

**If query latency >100ms:**
1. Profile the code - where is time spent?
2. Check index type - is it optimized?
3. Batch queries together when possible
4. Consider approximate search (IVF or HNSW) for 100k+ cards

### Fine-Tuning Strategy

**This is the core learning experience. Approach systematically:**

#### Phase 1: Understand the Baseline (Week 5-6)
- Generate embeddings with pretrained model
- Evaluate on test queries (create 20+ queries covering diverse abilities)
- **Identify failure modes**: Where does the model struggle?
- Document: "The model confuses X with Y because..."

**Example failure analysis:**
```
Query: "cards that flicker creatures"
Expected: Restoration Angel, Ephemerate, Cloudshift
Actual: Various exile effects, but missing key flicker cards
Hypothesis: Model doesn't understand "flicker" = "exile and return"
Conclusion: Need training data teaching this terminology
```

#### Phase 2: Create Training Data (Week 7)
- **Manual curation**: Create synonym pairs based on MTG knowledge
- Focus on MTG-specific terminology:
  - "flicker" ↔ "exile target creature you control, then return it to the battlefield"
  - "ETB" ↔ "enters the battlefield"
  - "dies trigger" ↔ "when this creature dies"
  - "ramp" ↔ "search your library for a land and put it onto the battlefield"
- **Quality over quantity**: 50 high-quality pairs > 500 noisy pairs
- Use sentence-transformers `InputExample` format
- Split 80/20 train/validation

**Teaching moment**: "Fine-tuning is only as good as your training data. Let's create 50 pairs manually first, evaluate the model, then decide if we need more."

**Optional**: Consider using GPT-4 to expand dataset (with manual validation)

#### Phase 3: Fine-Tune and Evaluate (Week 8-9)
- Use `MultipleNegativesRankingLoss` (treats other pairs in batch as negatives)
- **Explain the loss function**: What is it optimizing?
- Training config:
  - Epochs: 3-5 (more can lead to overfitting)
  - Warmup steps: 10% of training steps
  - Learning rate: 2e-5 (standard for fine-tuning)
  - Evaluation: compute validation loss every epoch
- Track training metrics (loss curves)
- **Compare before/after**: Show specific query improvements
- **Analyze what changed**: Which card embeddings moved most?

#### Phase 4: Iterate (Week 10)
- Find remaining failure cases
- Augment training data targeting those failures
- Experiment with hyperparameters (learning rate, epochs, batch size)
- **Document all experiments**: Create simple tracking file (experiments.json)

### Evaluation Framework

**Help build rigorous evaluation:**

#### Metrics to Implement
1. **Precision@K**: Of top-K results, how many are relevant?
2. **Recall@K**: Of all relevant cards, how many are in top-K?
3. **MRR**: Mean Reciprocal Rank of first relevant result
4. **Latency**: Query response time

#### Performance Targets
**Baseline (pretrained model):**
- Precision@5: ≥50%
- Precision@10: ≥40%
- Query latency: <100ms

**Post fine-tuning:**
- Precision@5: ≥70% (+20 points improvement)
- Precision@10: ≥60% (+20 points improvement)
- Mean Reciprocal Rank: ≥0.7

**If targets not met:**
- Analyze failure cases by query type
- Try different embedding model (all-mpnet-base-v2)
- Implement query expansion or hybrid BM25+semantic search
- Document limitations honestly in final report

#### Ground Truth Creation
- Create 20+ test queries covering diverse abilities
- Manually label relevant cards for each query (5-10 ground truth cards)
- Validate labels with MTG community or playgroup if possible
- Examples:
  - "cheap red removal" → Lightning Bolt, Shock, Chain Lightning
  - "cards that flicker" → Cloudshift, Restoration Angel, Ephemerate
  - "mana dorks" → Llanowar Elves, Birds of Paradise, Elvish Mystic

#### Comparative Evaluation
- Baseline: Pretrained model (no fine-tuning)
- Fine-tuned: Your custom model
- Ablations: What if we train on half the data? Different loss functions?
- Alternative baseline: TF-IDF + cosine similarity (classic IR approach)

**Educational focus**: "Evaluation is not just about numbers. Let's look at specific queries where the model improved or failed, and understand why."

---

## Experiment Tracking

### Simple but Effective Approach

For this project, avoid over-engineering. Use lightweight tracking:

#### experiments.json
Track each significant experiment:
```json
{
  "experiment_id": "001_baseline_embeddings",
  "date": "2026-02-01",
  "description": "Generated embeddings with pretrained all-MiniLM-L6-v2",
  "parameters": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32,
    "num_cards": 27000
  },
  "results": {
    "embedding_time_sec": 287,
    "index_size_mb": 10.2,
    "test_queries": 20,
    "mean_precision@5": 0.52,
    "mean_recall@10": 0.68
  },
  "notes": "Baseline performs okay on common abilities (flying, trample) but struggles with slang terms (flicker, blink). Fine-tuning needed."
}
```

#### Lab Notes (Markdown)
Maintain `lab_notes.md` with date-stamped entries:
```markdown
## 2026-02-15: Fine-tuning with 50 synonym pairs

**Hypothesis**: Training on MTG-specific terminology will improve precision on slang queries by 10-15%.

**Method**: Created 50 manual synonym pairs focusing on common player terminology (flicker, ramp, mill). Used MultipleNegativesRankingLoss with 3 epochs.

**Results**:
- Precision@5 improved from 52% → 68% (+16%)
- Biggest gains on terminology queries (flicker, ETB, dies)
- Still struggles with obscure mechanics

**Key learning**: Quality of training data matters more than quantity. 50 well-chosen pairs gave significant improvement.

**Next steps**: Create 50 more pairs targeting remaining failure cases.
```

**Claude's role**: After each significant experiment, offer to help update these logs with structured summaries.

---

## Week-by-Week Guidance

### Weeks 1-2: Foundation & Exploration
**Focus**: Environment setup and data understanding

**Deliverables**:
- Python environment configured (venv, dependencies installed)
- Scryfall data downloaded and explored
- Preprocessing pipeline implemented
- Basic EDA notebook showing card distribution, text statistics

**Claude's approach**:
- Provide setup scripts with detailed comments
- Guide exploratory data analysis with specific visualization suggestions
- Explain JSON parsing and pandas DataFrames
- Help design preprocessing pipeline with edge case handling

**Validation**: "Let's verify we have clean data before moving forward. Show me 5 random cards from your processed dataset."

### Weeks 3-4: Baseline System
**Focus**: Get something working end-to-end

**Deliverables**:
- Embedding generation pipeline
- FAISS index built
- Simple CLI search interface
- Performance benchmarks (timing, memory)

**Claude's approach**:
- Implement with priority on clarity over optimization
- Add extensive logging and error handling
- Create diagnostic commands for verification
- Benchmark performance characteristics

**Learning moment**: "This is your MVP. It doesn't need to be perfect, but it should work reliably. Let's test edge cases (empty query, very long query, special characters)."

### Weeks 5-6: Evaluation Framework
**Focus**: Measure what we have

**Deliverables**:
- Test query dataset (20+ queries with ground truth)
- Metrics implementation (precision, recall, MRR)
- Baseline evaluation results
- Failure mode analysis

**Claude's approach**:
- Help design test queries covering diverse card abilities
- Implement metrics with clear explanations
- Create visualizations (precision curves, per-query breakdown)
- Identify specific failure patterns

**Key question**: "Why did the model retrieve card X when we searched for Y? Let's look at their embeddings and understand the model's 'reasoning.'"

### Weeks 7-9: Fine-Tuning (Core Innovation)
**Focus**: Improve through domain adaptation

**Deliverables**:
- Training dataset (50-500 synonym pairs)
- Fine-tuned model
- Before/after evaluation comparison
- Experiment logs documenting all trials

**Claude's approach**:
- Guide training data creation (templates, examples, quality checks)
- Explain fine-tuning hyperparameters and their effects
- Monitor training (loss curves, validation metrics)
- Compare before/after with concrete query examples
- Encourage multiple experiments (data size, loss functions, learning rates)

**Experimentation mindset**: "Let's try three versions: 50 pairs, 100 pairs, and 200 pairs. Which gives the best return on effort? This teaches you how to balance data collection with model improvement."

### Weeks 10-11: Advanced Features (Stretch Goals)
**Focus**: Polish and enhance

**Deliverables**:
- Metadata filtering (colors, CMC, types)
- Query parser for complex searches
- Convenience features (result explanations)
- Optional: Streamlit demo UI

**Claude's approach**:
- Implement practical features that showcase the system
- Focus on features that highlight semantic search advantages
- Keep complexity manageable - don't gold-plate

**Stretch goal**: "If we have time, let's experiment with hybrid search (BM25 + semantic) or explain why certain cards were retrieved."

### Weeks 12-14: Documentation & Presentation
**Focus**: Communicate results effectively

**Deliverables**:
- Final report (15-20 pages)
- Presentation slides (20 minutes)
- Demo video (5 minutes)
- Clean codebase with README
- GitHub repository ready to share

**Claude's approach**:
- Help structure final report (academic writing support)
- Create clear visualizations for presentation
- Review code documentation for completeness
- Prepare compelling demo scenarios
- Final quality check

**Quality check**: "Can someone reproduce your work from the README? Let's verify all steps are documented with exact commands."

---

## Domain Knowledge: MTG Terminology

### Card Mechanics to Understand

**Evergreen keywords**: Flying, trample, haste, vigilance, lifelink, deathtouch, first strike, double strike, menace, reach, hexproof, indestructible

**Mechanical themes**:
- **Mill**: Moving cards from library to graveyard
- **Flicker/Blink**: Exile and return (flicker = temporary, blink = immediate)
- **Ramp**: Mana acceleration (land search, mana rocks)
- **Card draw**: Self-explanatory but has subcategories
- **Removal**: Destroying/exiling permanents
- **ETB (Enters the Battlefield)**: Trigger when permanent enters
- **Dies**: Trigger when creature goes to graveyard

**Historical syntax changes**:
- "Bury" → "destroy (can't be regenerated)"
- "Comes into play" → "enters the battlefield"
- "Remove from the game" → "exile"
- "Converted mana cost" → "mana value"

**Player jargon**:
- **Dork**: Creature that produces mana
- **Cantrip**: Spell that replaces itself (draws a card)
- **Wipe**: Destroy all creatures (board wipe)
- **Bolt**: Direct damage spell (named after Lightning Bolt)
- **Tutor**: Search library for a card (named after Demonic Tutor)

**Why this matters**: These synonyms are exactly what fine-tuning should teach the model. Help identify terminology gaps during evaluation and create training pairs to address them.

### Card Attributes
- **Colors**: WUBRG (White, Blue, Black, Red, Green) + colorless + multicolor
- **CMC/Mana Value**: Total mana cost (numeric)
- **Types**: Creature, Instant, Sorcery, Enchantment, Artifact, Land, Planeswalker
- **Rarity**: Common, Uncommon, Rare, Mythic Rare

---

## Troubleshooting Common Issues

### "The Model Isn't Learning"
**Diagnostic steps**:
1. Check training data quality - Are pairs actually similar/dissimilar?
2. Verify loss is decreasing - Plot training curve
3. Test on training examples - Can it overfit on training data?
4. Check learning rate - Too high (doesn't converge) or too low (learns slowly)?
5. Ensure preprocessing matches between training and inference

**Educational response**: "This is normal in ML development. Let's debug systematically rather than randomly trying fixes. Each step teaches us something about the model."

### "Search Results Don't Make Sense"
**Investigation**:
1. Inspect query embedding vs. retrieved card embeddings
2. Check cosine similarities - Are scores reasonable (0.5-0.9)?
3. Verify preprocessing - Did we strip important text?
4. Test with known-good queries - Sanity check (search for exact card text)
5. Visualize embeddings - t-SNE plot of query + results

**Learning opportunity**: "Search quality depends on embedding quality. Let's understand what the model 'sees' when it reads card text. This debugging process teaches you how to diagnose ML systems."

### "Performance Is Too Slow"
**Optimization path**:
1. Profile the code - Where is time spent? (use line_profiler)
2. Batch operations - Reduce model invocation overhead
3. Cache embeddings - Don't recompute on every run
4. Use appropriate FAISS index - Flat for <100k, IVF for larger
5. Consider GPU if available - Check torch.cuda.is_available()

**Teaching moment**: "Premature optimization is the root of all evil. Let's measure first, optimize second. This teaches you to profile before optimizing."

### "I Don't Know What Experiments to Run"
**Suggestion framework**:
- **Ablation**: What if we remove feature X? (e.g., train without reminder text)
- **Variation**: What if we change hyperparameter Y? (e.g., double training data)
- **Comparison**: How does approach A vs B perform? (e.g., different loss functions)
- **Error analysis**: Why does query Z fail? Can we fix it with more training data?

**Guidance**: "Good experiments start with questions. What are you curious about? What did the failure analysis reveal?"

### "Out of Memory During Embedding Generation"
**Solutions**:
- Reduce batch_size to 16 or 8
- Process in chunks, save intermediate results
- Use `torch.cuda.empty_cache()` if using GPU
- Check for memory leaks - are you accumulating tensors?

### "FAISS Search Returns Poor Results"
**Checklist**:
- Are embeddings normalized? (Required for cosine similarity with IndexFlatIP)
- Is query embedded with same model as cards?
- Did we accidentally shuffle card metadata? (Check indices match)
- Are we searching the right index file?
- Try IndexFlatL2 instead - does it improve?

### "Fine-Tuning Doesn't Improve Results"
**Debug steps**:
- Is training loss decreasing? (Plot loss curve)
- Is validation loss also decreasing? (If not, overfitting)
- Is dataset format correct? (InputExample with text pairs)
- Are we evaluating on training data? (Should use held-out test set)
- Try more epochs, more data, or different loss function

---

## Decision-Making Framework

### When to Ask vs. Implement

**Implement directly (default)**:
- Standard patterns (data loading, model inference, metrics)
- Performance optimizations (batching, caching, indexing)
- Error handling and logging
- Code refactoring for clarity
- Experiment variations (different hyperparameters)

**Ask first**:
- Major architectural changes (switching from FAISS to Weaviate)
- Dataset modifications (excluding certain card types)
- Evaluation methodology changes (which metrics to prioritize)
- Timeline adjustments (skipping planned features)
- Scope changes (adding features outside original plan)

### Trade-Off Priorities
**Optimize for (in order)**:
1. **Correctness**: Results are semantically relevant and reproducible
2. **Understanding**: Student learns from implementation
3. **Maintainability**: Code is readable and documented
4. **Performance**: Meets latency targets (<100ms per query)
5. **Features**: Nice-to-haves (UI polish, advanced filters)

**If time-constrained**:
- Core search (weeks 1-6): Non-negotiable
- Fine-tuning (weeks 7-9): High priority (this is the main contribution)
- UI/demo (week 12): Can be minimal (CLI is acceptable)
- Advanced features (metadata filters): Stretch goals

---

## Anti-Patterns to Avoid

### Code Smells
- Loading entire dataset into memory when batching is possible
- Recomputing embeddings on every run (cache them!)
- Hardcoded paths (use config.yaml or environment variables)
- Silent failures (always log errors with context)
- No progress indication for long operations (>10 seconds)
- Clever one-liners that obscure logic (prefer clarity)

### Project Management
- Waiting until week 13 to start documentation
- Skipping validation on small datasets before full run
- Not committing frequently (commit at least weekly)
- Implementing features before core pipeline works
- Optimizing prematurely (profile first)
- Scope creep (adding features before MVP is complete)

### ML/AI Specific
- Not normalizing embeddings for cosine similarity
- Training without train/validation split
- Ignoring validation loss (misses overfitting)
- Not testing on held-out queries (overfitting to test set)
- Comparing models without consistent evaluation procedure
- Fine-tuning without understanding baseline failures first

---

## File Structure

```
mtg-semantic-search/
├── data/
│   ├── raw/
│   │   └── oracle-cards-YYYYMMDD.json       # Raw Scryfall data
│   ├── processed/
│   │   ├── cards_clean.csv                  # Cleaned card data
│   │   └── embeddings.npy                   # Cached embeddings
│   └── test/
│       └── test_queries.json                # Ground truth test queries
├── src/
│   ├── download_scryfall.py                 # Fetch Scryfall data
│   ├── preprocess.py                        # Clean and format cards
│   ├── embed.py                             # Generate embeddings
│   ├── build_index.py                       # Create FAISS index
│   ├── search.py                            # Search interface
│   ├── finetune.py                          # Fine-tuning pipeline
│   └── evaluate.py                          # Metrics and evaluation
├── models/
│   ├── baseline/                            # Pretrained model cache
│   └── fine_tuned/                          # Fine-tuned model
├── index/
│   ├── mtg_faiss.index                      # FAISS vector index
│   └── card_metadata.pkl                    # Card metadata (parallel to index)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_finetuning_analysis.ipynb
│   └── 04_results_visualization.ipynb
├── experiments.json                         # Experiment tracking
├── lab_notes.md                            # Research notebook
├── requirements.txt                         # Python dependencies
├── .gitignore                              # Exclude data/, models/, index/
└── README.md                               # Setup and usage (created at end)
```

---

## Key Dependencies

```
# Core ML/NLP
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0

# Vector search
faiss-cpu>=1.7.4

# Data processing
pandas>=2.0.0
numpy>=1.24.0

# Utilities
requests>=2.31.0
tqdm>=4.65.0
loguru>=0.7.0

# Testing & visualization
pytest>=7.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Optional: UI
streamlit>=1.25.0
```

---

## Success Criteria

### Educational Success
- Student can explain how sentence transformers work at a conceptual level
- Student understands vector similarity metrics (cosine vs. L2)
- Student can design and execute ML experiments independently
- Student learns to evaluate and iterate on ML systems
- Student develops intuition for when fine-tuning helps vs. doesn't

### Technical Success
- Working semantic search system (end-to-end pipeline)
- Baseline precision@5 ≥ 50% (shows system works)
- Fine-tuned precision@5 ≥ 70% (shows improvement)
- Query latency < 100ms (interactive use)
- Comprehensive evaluation with clear methodology
- All code documented and reproducible

### Project Success
- Final report demonstrates deep understanding (not just results)
- Code is professional quality (readable, tested, documented)
- Presentation includes live demo showing semantic capabilities
- GitHub repository ready to share (README, examples, clear structure)
- Can explain trade-offs and limitations honestly

### Exceptional Project (Stretch Goals)
- Precision@5 reaches ≥80%
- Advanced features implemented (hybrid search, query parser, metadata filters)
- Deployed demo accessible via web URL
- Published to GitHub with comprehensive documentation
- Multiple fine-tuning experiments documented with insights

---

## Academic Integrity

### Appropriate Use of AI Assistance

**Claude Code should help the student**:
- **Understand concepts** through explanations and examples
- **Debug their own code** with guidance and diagnostic steps
- **Learn through experimentation** with suggested variations
- **Develop intuition** by explaining trade-offs and decisions

**Claude Code should NOT**:
- Write entire modules without explanation or teaching
- Solve problems without helping student understand the concepts
- Complete assignments without student engagement or learning
- Provide solutions without explaining the reasoning

### Attribution Guidelines

**In the final report, include**:
```
## Acknowledgments

Development was assisted by Claude Code (Anthropic) for code review,
debugging support, implementation guidance, and conceptual explanations.
All design decisions, experimental methodology, and evaluations were
conducted by the author. All code was written with full understanding
of its functionality and purpose.
```

**Key principle**: You should be able to explain every line of code and every design decision in your own words.

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

# Optional: UI
streamlit run app.py
```

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
python -m cProfile -o profile.stats src/embed.py
python -m pstats profile.stats
```

### Common Error Messages and Fixes

**"RuntimeError: CUDA out of memory"**
- Not applicable (CPU-only), but if using GPU: reduce batch_size

**"FileNotFoundError: embeddings.npy"**
- Run `python src/embed.py` to generate embeddings first

**"KeyError: 'oracle_text'"**
- Some cards lack oracle_text (tokens, lands) - filter during preprocessing

**"ImportError: No module named 'faiss'"**
- Install with `pip install faiss-cpu` (not `faiss`)

---

## Extended Thinking Mode

For complex problems requiring deeper analysis, you can trigger extended thinking:

**Trigger phrases**:
- "think" - basic extended thinking
- "think hard" - deeper analysis
- "think harder" - thorough investigation
- "ultrathink" - maximum reasoning depth

**Use extended thinking for**:
- Complex debugging (multi-step failures)
- Architectural decisions (index type selection, model choice)
- Performance optimization strategies
- Fine-tuning hyperparameter selection
- Experiment design

**Example**: "think hard about why fine-tuning might not be improving precision on synonym queries"

---

## Document Metadata

**Version**: 1.0 (Combined Educational-Pragmatic)
**Created**: November 3, 2025
**Purpose**: Unified guidance for Claude Code across 14-week independent study
**Approach**: Learning-first with production-quality standards
**Philosophy**: "Learn deeply, build pragmatically, iterate thoughtfully"

---

## How to Use This Guide

**For Claude Code**:
1. This is your primary reference for all development decisions
2. When uncertain, check this guide for established patterns
3. Balance educational value (teaching) with efficiency (shipping)
4. Adapt your support style to the project phase (weeks 1-2 need more teaching, weeks 12-14 need polish)
5. Flag deviations from the plan for discussion

**For Mitchell (the student)**:
- This guide defines how Claude Code will work with you
- It balances learning (your primary goal) with getting things done
- Claude will explain decisions, not just implement features
- Ask questions whenever something isn't clear
- Feel free to deviate from recommendations - this is your project

**Project philosophy**: This is a learning journey, not just a coding task. Every interaction should leave you more capable and knowledgeable. At the end of 14 weeks, you should be able to independently design, implement, and evaluate a new ML system from scratch.

---

**Remember: Make it work, make it right, make it fast - in that order.**
