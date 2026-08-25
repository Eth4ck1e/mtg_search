# Agent 2 Prompt Summary: Educational ML Development Guide

## Research Conducted

### 1. Claude Code Best Practices (2025)
**Key findings:**
- **CLAUDE.md files**: Highest-impact practice for providing persistent project context
- **Context management**: Clear frequently to avoid degradation from stale information
- **Mode separation**: Keep Build/Learn/Critique as separate interactions
- **Custom slash commands**: Store repeated workflows in `.claude/commands/`
- **Iterative workflow**: Plan → small diff → tests → review
- **Checkpoints**: Use /rewind to explore safely

### 2. Educational ML Prompting Patterns
**Core strategies:**
- Provide context, be specific, build on conversation
- Clear communication with structured prompts
- Use examples and constraints
- Contemporary documentation enhances collaboration
- Prompt engineering essential for educational applications

### 3. ML Experiment Tracking Best Practices
**Industry standards:**
- **MLflow**: Automatic logging, language-agnostic, structured workloads
- **Weights & Biases**: Dynamic deep learning, git integration, reproducibility
- **Core principles**: Reproducibility, comparison, collaboration
- Track inputs (code, datasets, hyperparameters) and outputs (metrics, models)

### 4. Sentence Transformer Fine-Tuning
**Domain adaptation approaches:**
- **Supervised**: Use NLI or semantic similarity pairs (most common)
- **Unsupervised**: TSDAE for domains without labeled data
- **Dataset requirements**: 500-5000 pairs of semantically similar/dissimilar texts
- **Minimal resources**: Can achieve gains with <5000 pairs, <1 minute training
- **Loss functions**: MultipleNegativesRankingLoss for retrieval tasks

### 5. Vector Database Project Structure
**Key considerations:**
- **Chunking strategy**: Most critical decision affecting quality, speed, cost
- **Index types**: HNSW for performance, flat for accuracy
- **Hybrid search**: Combine vector + keyword search for best results
- **Batch processing**: Essential for memory management
- **Domain-specific tuning**: Fine-tune embeddings for specialized vocabulary

---

## Prompt Design Philosophy

### Differentiation from Standard Development Prompts

**Agent 1 might create**: Production-focused, efficiency-oriented, feature-complete system
**Agent 2 creates**: Learning-focused, experimentation-driven, understanding-oriented guide

### Core Principles

1. **Teaching Through Implementation**
   - Explain reasoning behind every decision
   - Highlight trade-offs and alternatives
   - Connect code to underlying theory
   - Encourage experimentation and variations

2. **Iterative Experimentation**
   - Hypothesis-driven development
   - Celebrate instructive failures
   - Promote ablation studies
   - Incremental improvements over perfection

3. **Documentation as Learning**
   - Log experiments with rationale
   - Create reproducible workflows
   - Maintain lab notebooks
   - Write commit messages that teach

4. **Code Quality with Educational Intent**
   - Clarity over cleverness
   - Teaching comments that explain "why"
   - Error messages that educate
   - Type hints and docstrings as learning tools

---

## Key Innovations in This Prompt

### 1. Educational Response Patterns
Structured approach for complex tasks:
- Overview (what and why)
- Plan (numbered steps)
- Implementation (teaching comments)
- Testing (verification)
- Next steps (experiments)

### 2. Domain-Specific Context
Comprehensive MTG terminology guide:
- Evergreen keywords and mechanics
- Player jargon and synonyms
- Historical syntax evolution
- Card attributes and formats

This enables Claude to understand WHY certain searches fail and HOW fine-tuning should address terminology gaps.

### 3. Experiment Tracking (Lightweight)
Simple JSON-based logging instead of heavy frameworks:
- `experiments.json` for structured results
- `lab_notes.md` for observations
- Git commits as experiment versioning

Rationale: For a 14-week project, learning the ML concepts is more valuable than mastering MLflow.

### 4. Progress Monitoring
Week-by-week guidance adapting to project phase:
- Weeks 1-2: Foundation (setup, exploration)
- Weeks 3-4: Baseline (get it working)
- Weeks 5-6: Evaluation (measure quality)
- Weeks 7-9: Fine-tuning (core innovation)
- Weeks 10-11: Enhancement (polish)
- Weeks 12-14: Communication (report, demo)

### 5. Debugging as Learning Opportunities
Systematic diagnosis with educational framing:
- "The model isn't learning" → Check data, loss, learning rate
- "Search results don't make sense" → Inspect embeddings, visualize
- "Performance is slow" → Profile first, optimize second

Each issue becomes a teaching moment about ML engineering.

### 6. Academic Integrity Framework
Clear guidance on appropriate AI assistance:
- Understand concepts vs. copy answers
- Debug with guidance vs. complete rewrites
- Learn through experimentation
- Acknowledge AI use transparently

---

## Structural Elements

### Major Sections

1. **Project Identity & Context** (Academic setting, scope, constraints)
2. **Claude Code's Role** (Teaching, experimentation, documentation, code quality)
3. **Communication Guidelines** (Explaining ML concepts, design decisions, error handling)
4. **Technical Guidelines** (Data preprocessing, embeddings, FAISS, fine-tuning, evaluation)
5. **Experiment Tracking** (Simple logging approach)
6. **Milestones** (Week-by-week guidance)
7. **Domain Knowledge** (MTG terminology)
8. **Common Challenges** (Debugging patterns)
9. **Code Standards** (Best practices with examples)
10. **Academic Integrity** (Attribution guidelines)
11. **Tools & Resources** (Required, recommended, optional)
12. **Example Interactions** (Response patterns for common scenarios)

### Unique Features

**Scenario-Based Examples**: Three detailed interaction patterns showing how Claude should respond to:
- "How do I fine-tune?" (Teaching moment)
- "Search results are bad" (Debugging pattern)
- "I'm ahead of schedule" (Advanced options)

**Progress Monitoring**: Weekly self-assessment questions and red flags to watch for (scope creep, perfectionism, timeline concerns)

**Multiple Learning Paths**: When student is ahead, suggest three options:
- Advanced features (impressive demo)
- Deeper analysis (better learning)
- Deployment (real-world experience)

Recommends Path 2 (analysis) as most educationally valuable.

---

## Comparison with Typical Development Prompts

| Aspect | Standard Prompt | Agent 2 (Educational) |
|--------|----------------|----------------------|
| **Primary Goal** | Build working system | Teach ML/NLP concepts |
| **Code Style** | Efficient, concise | Clear, heavily commented |
| **Error Handling** | Fix quickly | Explain root cause |
| **Experimentation** | Avoid unless necessary | Core methodology |
| **Documentation** | Minimum viable | Extensive with rationale |
| **Success Metric** | Features completed | Understanding gained |
| **Timeline Pressure** | Ship fast | Learn thoroughly |
| **Failure Framing** | Bugs to fix | Learning opportunities |

---

## Anticipated Impact

### For the Student
- Deeper understanding of transformer models and embeddings
- Hands-on experience with experiment design
- Confidence in ML engineering practices
- Portfolio-worthy project with clear methodology

### For the Project
- Well-documented, reproducible system
- Rigorous evaluation framework
- Clear learning progression
- Academic-quality final report

### For ML Education
- Reusable template for supervised ML projects
- Example of AI-assisted learning done right
- Balance of theory and implementation
- Emphasis on understanding over completion

---

## Technical Specifications

**File**: `/Users/mitchelltrafford/Documents/Development/Independent Study - MTG Vector DB/agent2-prompt.md`
**Size**: 27KB (comprehensive guidance document)
**Format**: Well-structured Markdown with:
- Clear section hierarchy
- Code examples with annotations
- Tables for comparisons
- Bullet lists for guidelines
- Callout boxes for key teaching moments

**Usage**: Can be used as:
1. System prompt for Claude Code
2. Project guide for student reference
3. Template for similar ML educational projects
4. Onboarding document for project advisors

---

## Key Differentiators from Agent 1 Approach

1. **Experimentation Over Optimization**: Encourages trying multiple approaches rather than finding the single best solution quickly

2. **Understanding Over Completion**: Values learning why something works more than just getting it working

3. **Lightweight Tooling**: Simple JSON logs instead of MLflow/W&B to reduce overhead and focus on ML concepts

4. **Domain Integration**: Deep MTG knowledge embedded so Claude can provide context-aware guidance

5. **Academic Framing**: Explicitly addresses independent study context, honors program expectations, and academic integrity

6. **Week-by-Week Adaptation**: Claude's behavior changes based on project phase (exploratory in weeks 1-2, rigorous in weeks 5-6, innovative in weeks 7-9)

7. **Failure as Pedagogy**: Treats debugging and failed experiments as core learning experiences rather than setbacks

---

## Potential Extensions

If this prompt is successful, it could be adapted for:
- Other domain-specific NLP projects (legal documents, scientific papers)
- Different ML tasks (classification, generation, reinforcement learning)
- Longer timelines (full semester courses, capstone projects)
- Team projects (adding collaboration guidance)
- Research projects (stronger emphasis on literature review and publication)

---

## Metadata

- **Created**: November 3, 2025
- **Research Sources**:
  - Anthropic Claude Code documentation
  - MLflow and Weights & Biases best practices
  - Sentence-transformers training guides
  - Vector database implementation patterns
  - Educational AI prompting literature
- **Word Count**: ~7,500 words (main prompt)
- **Target Audience**: Claude Code (as assistant) and Mitchell Trafford (as student)
- **Project Context**: Spring 2026 independent study (14 weeks)
- **Approach**: Learning-first, experimentation-driven, theory-connected

---

## Success Indicators

The prompt will be successful if:

1. **Student demonstrates deep understanding** of embeddings, vector search, and fine-tuning in final presentation
2. **Code is well-documented** with teaching comments that explain design decisions
3. **Experiments are tracked** with clear hypotheses and results
4. **Final report** shows rigorous evaluation methodology
5. **Project is reproducible** from README alone
6. **Student can independently** design new ML experiments after completion

The prompt prioritizes these learning outcomes over rapid feature development or production-ready code.
