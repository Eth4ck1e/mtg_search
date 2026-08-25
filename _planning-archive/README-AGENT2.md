# Agent 2: Educational ML Development Guide - Documentation Index

## Overview

This directory contains comprehensive documentation for Agent 2, an educational approach to developing the MTG Vector Database semantic search system. Agent 2 prioritizes learning and understanding over rapid feature development, making it ideal for academic independent study projects.

**Created**: November 3, 2025
**Purpose**: Spring 2026 Independent Study - MTG Semantic Search System
**Approach**: Learning-first, experimentation-driven ML development

---

## Core Documents

### 1. agent2-prompt.md (27KB)
**The main prompt/guide for Claude Code**

**Contents**:
- Project identity and academic context
- Claude Code's role as educational assistant
- Communication guidelines for ML concepts
- Technical guidelines (data, embeddings, FAISS, fine-tuning)
- Experiment tracking and logging strategies
- Week-by-week milestone guidance
- Domain-specific MTG knowledge
- Common ML project challenges
- Code style and best practices
- Academic integrity guidelines
- Example interaction patterns

**Use this for**:
- System prompt in Claude Code
- Reference guide during development
- Understanding the educational philosophy
- Structuring learning interactions

**Key Innovation**: Every technical implementation includes:
- Explanation of "why" not just "what"
- Trade-offs and alternatives
- Connection to underlying theory
- Suggestions for experiments
- Educational value

---

### 2. agent2-summary.md (10KB)
**Executive summary of research and design decisions**

**Contents**:
- Research findings (Claude Code best practices, ML experiment tracking, fine-tuning approaches)
- Prompt design philosophy
- Key innovations (educational response patterns, domain context, lightweight logging)
- Structural elements overview
- Comparison with typical development prompts
- Anticipated impact

**Use this for**:
- Quick understanding of Agent 2's approach
- Design rationale and research basis
- Understanding what makes this prompt unique
- Seeing the research foundations

**Key Takeaway**: Agent 2 differentiates through teaching-focused interactions, experiment encouragement, and theory-connected implementations.

---

### 3. AGENT-COMPARISON.md (13KB)
**Comparison of different agent approaches**

**Contents**:
- Agent 1 (Production): Build fast, ship product
- Agent 2 (Educational): Learn deeply, understand thoroughly
- Agent 3 (Research): Explore novel methods, publish papers
- Side-by-side comparison table
- Scenario-based comparisons (3 detailed scenarios)
- Guidance on choosing the right agent
- Hybrid approaches
- Rationale for Agent 2 selection for this project

**Use this for**:
- Understanding why Agent 2 is appropriate
- Comparing different development philosophies
- Seeing how approaches differ in practice
- Making informed choices for future projects

**Key Insight**: Agent choice depends on project goals (learning vs. product vs. research), timeline, and success criteria.

---

### 4. USING-AGENT2-PROMPT.md (15KB)
**Practical guide for effective usage**

**Contents**:
- Setup instructions (3 options)
- Effective usage patterns
- Weekly workflow templates
- Phase-specific tips (weeks 1-14)
- Maximizing learning strategies
- Red flags and solutions
- Integration with other tools
- Common pitfalls to avoid
- Sample session transcript
- Success metrics

**Use this for**:
- Day-to-day interactions with Claude Code
- Starting each work session
- Debugging and experimentation
- Weekly planning and reflection
- Ensuring productive learning

**Key Feature**: Concrete prompts for each project phase with examples of effective and ineffective interactions.

---

## Quick Start

### For Immediate Use

1. **Read**: `USING-AGENT2-PROMPT.md` (Section: Setup)
2. **Implement**: Choose setup option (CLAUDE.md recommended)
3. **Begin**: Use phase-specific prompts for current week
4. **Reference**: `agent2-prompt.md` for detailed guidance

### For Understanding the Approach

1. **Read**: `agent2-summary.md` (Overview of philosophy)
2. **Compare**: `AGENT-COMPARISON.md` (Why this approach?)
3. **Deep Dive**: `agent2-prompt.md` (Full specification)
4. **Apply**: `USING-AGENT2-PROMPT.md` (Practical usage)

---

## Document Relationships

```
agent2-prompt.md (The Guide)
    ├── What: Comprehensive prompt specification
    ├── Why: Explained in agent2-summary.md
    ├── How: Detailed in USING-AGENT2-PROMPT.md
    └── Compared: Against other agents in AGENT-COMPARISON.md
```

**Flow**:
1. Understand philosophy → `agent2-summary.md`
2. Compare approaches → `AGENT-COMPARISON.md`
3. Learn full specification → `agent2-prompt.md`
4. Apply practically → `USING-AGENT2-PROMPT.md`

---

## Key Principles (Quick Reference)

### 1. Teaching Through Implementation
Every code section explains reasoning, trade-offs, and alternatives

### 2. Iterative Experimentation
Try variations, learn from failures, build incrementally

### 3. Theory-Connected
Link implementations to underlying ML/NLP concepts

### 4. Hypothesis-Driven
Frame work as experiments with clear predictions

### 5. Documentation-Focused
Track experiments, maintain lab notes, explain decisions

### 6. Reproducibility-First
Others should be able to recreate your work from documentation

---

## Usage by Project Phase

### Planning (Pre-Week 1)
- Read all documentation to understand approach
- Set up CLAUDE.md file
- Review week-by-week milestones

### Foundation (Weeks 1-2)
- Use `agent2-prompt.md` Section: "Weeks 1-2 Foundation"
- Emphasize exploration and understanding
- Document data structure insights

### Baseline (Weeks 3-4)
- Focus on clarity over optimization
- Test small before scaling
- Benchmark performance

### Evaluation (Weeks 5-6)
- Design rigorous test queries
- Implement multiple metrics
- Analyze failure patterns

### Fine-Tuning (Weeks 7-9)
- Create high-quality training data
- Understand loss functions
- Compare before/after

### Enhancement (Weeks 10-11)
- Add metadata filtering
- Implement query parsing
- Consider advanced features

### Communication (Weeks 12-14)
- Structure final report
- Create clear visualizations
- Prepare demo

---

## Expected Learning Outcomes

By following Agent 2's guidance:

### Technical Skills
- Transformer models and embeddings
- Vector search and similarity metrics
- Fine-tuning for domain adaptation
- Experiment design and evaluation
- ML engineering best practices

### Understanding
- Why sentence transformers work
- How vector similarity enables search
- When fine-tuning helps vs. hurts
- Trade-offs in system design

### Professional Skills
- Code documentation and testing
- Experiment tracking and logging
- Technical writing
- Presentation and demo skills

### Project Artifacts
- Working semantic search system
- Comprehensive evaluation
- Well-documented codebase
- Academic-quality final report
- Professional GitHub repository

---

## Success Indicators

You're on track if:

### Weekly
- [ ] Can explain what you built and why
- [ ] Tried at least one experiment/variation
- [ ] Updated experiments.json and lab_notes.md
- [ ] Code has teaching comments
- [ ] Made meaningful git commits

### Monthly (Every 4 weeks)
- [ ] Completed phase deliverables
- [ ] Understanding deepened (can teach concept)
- [ ] No major blockers unresolved
- [ ] Documentation reflects current state

### End of Semester
- [ ] Can explain transformer models to non-experts
- [ ] System achieves target metrics (≥70% precision@5)
- [ ] Work is fully reproducible
- [ ] Final report demonstrates deep understanding
- [ ] Capable of designing new ML systems independently

---

## Research Foundations

Agent 2's approach is informed by:

### Claude Code Best Practices (2025)
- CLAUDE.md for persistent context
- Context management and checkpoints
- Custom slash commands
- Iterative development patterns

### Educational AI Research
- Prompt engineering for learning
- Active engagement over passive receipt
- Structured feedback and reflection
- Theory-practice connections

### ML Experiment Tracking
- Lightweight logging for learning projects
- Reproducibility through documentation
- Hypothesis-driven experimentation
- Comparison and ablation studies

### Sentence Transformer Fine-Tuning
- Supervised pairs for domain adaptation
- Loss function selection (MultipleNegativesRankingLoss)
- Quality over quantity in training data
- Minimal resources for significant gains

### Vector Database Design
- Chunking strategies for semantic search
- Index type selection (FAISS)
- Hybrid search approaches
- Domain-specific optimization

---

## Files at a Glance

| File | Size | Purpose | Read When |
|------|------|---------|-----------|
| `agent2-prompt.md` | 27KB | Complete specification | Setting up, need reference |
| `agent2-summary.md` | 10KB | Research & philosophy | Understanding approach |
| `AGENT-COMPARISON.md` | 13KB | Compare approaches | Deciding on agent |
| `USING-AGENT2-PROMPT.md` | 15KB | Practical usage | Daily work sessions |
| `README-AGENT2.md` | This file | Navigation & overview | Starting point |

**Total Documentation**: ~80KB / ~20,000 words

---

## Common Questions

### Q: Do I need to read all documents?

**A**: Minimum: `USING-AGENT2-PROMPT.md` for practical usage. Recommended: All documents for full understanding. Order: summary → comparison → prompt → usage guide.

### Q: Can I mix Agent 2 with other approaches?

**A**: Yes! See `AGENT-COMPARISON.md` section "Hybrid Approaches." Common pattern: Start with Agent 2 (learning), transition to Agent 1 (polish).

### Q: What if I get stuck?

**A**: Refer to `USING-AGENT2-PROMPT.md` section "Red Flags and Solutions." Use the prompts provided for systematic debugging.

### Q: How do I track experiments?

**A**: Agent 2 recommends simple JSON tracking. See `agent2-prompt.md` section "Experiment Tracking and Logging" for templates.

### Q: What about academic integrity?

**A**: See `agent2-prompt.md` section "Academic Integrity and Attribution" for guidelines on appropriate AI use and acknowledgment.

---

## Getting Started Today

### First-Time Setup (15 minutes)

1. **Read** this file (you're doing it!)
2. **Skim** `agent2-summary.md` (understand philosophy)
3. **Read** `USING-AGENT2-PROMPT.md` → "Setup" section
4. **Create** `.claude/CLAUDE.md` in project root
5. **Start** first session with Agent 2

### First Session (1-2 hours)

Prompt Claude Code with:
```
I'm starting the MTG Vector Database independent study.
Following the Agent 2 educational approach from agent2-prompt.md.

Week 1, Session 1:
- Goal: Set up environment and understand data structure
- Learning focus: Scryfall API, data preprocessing

Let's start by:
1. Explaining the Scryfall data structure
2. Setting up Python environment
3. Downloading sample data (100 cards)
4. Exploring data characteristics

Teach me as we go - I want to understand each step.
```

---

## Updating Documentation

As the project progresses, consider updating:

### CLAUDE.md (Weekly)
Update "Current Phase" section with:
- Current week and focus area
- Recent accomplishments
- Immediate next steps

### experiments.json (After Each Experiment)
Log:
- Experiment ID and date
- Hypothesis and parameters
- Results and comparison to baseline
- Observations and next steps

### lab_notes.md (2-3x per Week)
Record:
- What you learned
- Interesting findings
- Questions for further investigation
- Ideas for future experiments

---

## For Advisors and Evaluators

This documentation demonstrates:

1. **Systematic Approach**: Clear methodology and structure
2. **Learning Focus**: Prioritizes understanding over speed
3. **Reproducibility**: Experiment tracking and documentation
4. **Academic Rigor**: Proper evaluation and attribution
5. **Professional Development**: Industry-standard practices

The Agent 2 approach ensures the student develops deep ML/NLP expertise while producing a high-quality, well-documented system suitable for academic evaluation.

---

## Next Steps

1. **Review** the appropriate document based on your need
2. **Set up** your development environment with CLAUDE.md
3. **Begin** week 1 with educational prompts
4. **Track** your experiments from day one
5. **Reflect** weekly on learning progress

---

## Feedback and Iteration

This is a living approach. As you use Agent 2:

- Note what works well → reinforce those patterns
- Identify gaps → address in next session
- Document learnings → improve for future projects
- Share insights → help others learn

**Remember**: The goal is not just building a system, but becoming an ML engineer who can independently design, implement, and evaluate novel systems.

---

## Contact and Attribution

**Project**: MTG Vector Database Semantic Search
**Student**: Mitchell Trafford
**Semester**: Spring 2026 (14 weeks)
**Agent Design**: November 3, 2025
**Approach**: Educational, Experimentation-Driven ML Development

**Acknowledgment**: Agent 2 prompt designed with research into Claude Code best practices, educational AI patterns, and ML engineering workflows.

---

**Ready to begin? Start with `USING-AGENT2-PROMPT.md` and dive into your learning journey!**
