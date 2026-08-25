# Quick Start Guide: Using the Agent 2 Educational Prompt

## Overview

This guide explains how to effectively use the Agent 2 prompt (`agent2-prompt.md`) with Claude Code for your MTG Vector Database independent study project.

---

## Setup

### Option 1: Use as System Prompt (Recommended)

If your Claude Code interface supports custom system prompts:

1. Open `agent2-prompt.md`
2. Copy the entire contents
3. Paste into your Claude Code system prompt field
4. Start your conversation

**Pros**: Claude will consistently follow the educational approach
**Cons**: May use more context tokens

### Option 2: Reference at Start of Session

Begin each major work session by pasting:

```
I'm working on the MTG Vector Database independent study project.
Please follow the guidance in agent2-prompt.md for an educational,
experimentation-driven approach. Key points:

- Explain "why" not just "what"
- Encourage experimentation with alternatives
- Teach ML/NLP concepts through implementation
- Support hypothesis-driven development
- Maintain reproducibility and documentation

Current phase: [Week X - Focus Area]
Today's goal: [Specific task]
```

**Pros**: Flexible, can adjust per session
**Cons**: Need to re-establish context each time

### Option 3: Create CLAUDE.md File

Create `.claude/CLAUDE.md` in your project root:

```markdown
# MTG Vector Database - Educational Independent Study

## Project Context
- Spring 2026 semester (14 weeks)
- Learning-focused: Understanding is as important as implementation
- See agent2-prompt.md for full guidance

## Current Phase
Week [X]: [Phase name and focus]

## Assistant Behavior
- Explain reasoning and trade-offs
- Suggest experiments and variations
- Link code to underlying ML/NLP theory
- Encourage reproducibility
- Support iterative learning

## Common Commands
- `python src/embed.py`: Generate embeddings
- `python src/search.py "query"`: Test search
- `jupyter notebook`: Launch exploration environment

## Key Files
- data/processed/cards_clean.csv: Preprocessed cards
- experiments.json: Experiment tracking
- lab_notes.md: Observations and findings
```

**Pros**: Automatically loaded by Claude Code, persistent context
**Cons**: Need to update as project progresses

---

## Effective Usage Patterns

### Starting a New Feature

**Good Prompt**:
```
I'm ready to implement [feature name]. Before we start:
1. Explain the approach and why it's appropriate
2. Highlight trade-offs vs. alternatives
3. Show how to test it
4. Suggest experiments to run after

Current context: [What's already built]
Learning goal: [What I want to understand]
```

**Why This Works**: Triggers Agent 2's teaching mode, ensures you understand before coding

---

### Debugging Issues

**Good Prompt**:
```
I'm seeing [error/unexpected behavior]. Let's debug systematically:

What I tried: [Your attempts]
What I expected: [Expected result]
What happened: [Actual result]

Help me understand:
1. What's the root cause?
2. Why did this happen?
3. How do I fix it?
4. How do I prevent this in future?
```

**Why This Works**: Encourages learning from errors, not just fixes

---

### Running Experiments

**Good Prompt**:
```
Experiment: [Name]
Hypothesis: [What you expect to happen]
Current baseline: [Performance metrics]

Let's implement this experiment, then:
1. Measure the impact
2. Analyze why it worked/didn't work
3. Update experiments.json
4. Decide on next experiment
```

**Why This Works**: Frames work as hypothesis-driven science, promotes learning

---

### When Stuck or Lost

**Good Prompt**:
```
I'm stuck on [problem] and not sure how to proceed.

Help me:
1. Break this into smaller pieces
2. Understand what's blocking me
3. Identify what I need to learn
4. Create a plan forward
```

**Why This Works**: Addresses learning gaps, not just task completion

---

## Weekly Workflow

### Monday: Plan the Week
```
Starting week [X] with focus on [phase].

Goals for this week:
- [Goal 1]
- [Goal 2]
- [Goal 3]

What should I prioritize? What experiments should I run?
Are these goals realistic for ~10-12 hours of work?
```

### Daily: Focused Sessions
```
Today's task: [Specific implementation]

Before we start:
- Where does this fit in the overall system?
- What will I learn from implementing this?
- How will I verify it works?

Let's build this step-by-step with teaching comments.
```

### Friday: Weekly Reflection
```
Week [X] review:

Completed:
- [What worked]

Learned:
- [New concepts understood]

Challenges:
- [What was difficult]

Help me:
1. Update experiments.json
2. Write lab_notes.md entry
3. Plan next week's priorities
```

---

## Phase-Specific Tips

### Weeks 1-2: Foundation
**Focus**: Setup and exploration

**Good Interactions**:
```
"Let's set up the environment step-by-step. Explain why we need each dependency."

"Walk me through the Scryfall data structure. What are the edge cases to watch for?"

"Let's create a preprocessing pipeline. Show me the trade-offs between different approaches."
```

**Avoid**:
- Asking for complete solutions without understanding
- Skipping exploratory data analysis
- Rushing to implementation

---

### Weeks 3-4: Baseline System
**Focus**: Get something working

**Good Interactions**:
```
"Let's implement batch encoding. Explain why batching matters and what size to use."

"Show me how to build a FAISS index. What's the difference between index types?"

"Help me create a CLI for testing. What should I test first?"
```

**Avoid**:
- Optimizing before profiling
- Skipping small-scale tests (try 100 cards first)
- Not logging intermediate results

---

### Weeks 5-6: Evaluation
**Focus**: Measure quality rigorously

**Good Interactions**:
```
"Let's design test queries together. What types of abilities should I cover?"

"Explain precision@K and recall@K. Why use both?"

"Help me analyze failure cases. What can they teach us about the model?"
```

**Avoid**:
- Only checking if code runs (need quality metrics)
- Cherry-picking good examples
- Not documenting baseline performance

---

### Weeks 7-9: Fine-Tuning
**Focus**: Core learning experience

**Good Interactions**:
```
"Let's create training pairs. Show me 5 examples and explain what makes them good."

"Explain MultipleNegativesRankingLoss. Why is it appropriate for retrieval?"

"Let's analyze what changed after fine-tuning. Which embeddings moved most?"
```

**Avoid**:
- Generating huge datasets without quality checks
- Training without validation split
- Not comparing before/after embeddings

---

### Weeks 10-11: Enhancement
**Focus**: Polish and advanced features

**Good Interactions**:
```
"Let's add metadata filtering. How should we combine FAISS with pandas?"

"Help me build a query parser. What's the simplest approach that could work?"

"Should we try cross-encoder reranking? What's the trade-off?"
```

**Avoid**:
- Adding features without evaluating impact
- Premature deployment
- Scope creep (defer to future work if needed)

---

### Weeks 12-14: Documentation
**Focus**: Communicate clearly

**Good Interactions**:
```
"Let's structure the final report. What should each section cover?"

"Help me create clear visualizations. What tells the story of this project?"

"Review my README. Can someone reproduce this work?"
```

**Avoid**:
- Leaving documentation to last minute
- Only showing successes (failures teach too)
- Not acknowledging AI assistance

---

## Maximizing Learning

### Ask "Why" Questions

Instead of:
```
"How do I implement X?"
```

Try:
```
"What are three approaches to implementing X?
What are the trade-offs?
Which should I try first and why?"
```

### Request Alternatives

Instead of:
```
"Give me the best solution."
```

Try:
```
"Show me two approaches: one simple, one optimized.
I'll implement the simple version first to understand it,
then we can optimize if needed."
```

### Connect to Theory

Instead of:
```
"Here's code that works."
```

Try:
```
"This code implements [technique] from [paper/concept].
Here's why it works: [explanation]
Let's verify by testing [specific case]."
```

### Document Learning

After each significant implementation:
```
"Help me write a lab_notes.md entry about what I learned:
- What was the challenge?
- What approach did we use?
- What did I learn about ML/NLP?
- What would I do differently next time?"
```

---

## Red Flags and Solutions

### Red Flag: You Don't Understand the Code

**Symptom**: Code works but you can't explain it

**Solution**:
```
"I have working code but don't fully understand it.
Let's walk through it line-by-line. For each section:
1. What does it do?
2. Why is this approach used?
3. What would break if we changed this?
4. What alternatives exist?"
```

### Red Flag: Experiments Aren't Tracked

**Symptom**: Can't remember what you tried or results

**Solution**:
```
"Let's create an experiment tracking system.
What should I log for each experiment?
Help me document the last 3 things we tried
and set up a template for future experiments."
```

### Red Flag: Behind Schedule

**Symptom**: Week 8 and no working system

**Solution**:
```
"I'm behind schedule. Help me:
1. Assess current state realistically
2. Identify what's blocking progress
3. Determine minimum viable deliverable
4. Create revised timeline
5. Identify what to defer to 'Future Work'"
```

### Red Flag: Too Much Scope

**Symptom**: Adding features instead of evaluating

**Solution**:
```
"I'm tempted to add [feature] but haven't evaluated yet.
Should I:
A) Finish evaluation first
B) Add this feature (why is it important?)
C) Defer to future work

Help me prioritize."
```

---

## Integration with Other Tools

### Jupyter Notebooks

Use notebooks for exploration, Claude for implementation:

```
Notebook workflow:
1. Explore data visually (plots, samples)
2. Prototype approaches (quick experiments)
3. Find what works

Then with Claude:
"I've prototyped [approach] in notebook.
Let's refactor into clean modules with:
- Proper error handling
- Documentation
- Tests
- Logging"
```

### Git Integration

After significant milestones:
```
"Help me write a commit message for [changes].
It should explain:
- What was implemented
- Why this approach
- What was learned
- What's next

Keep it clear for my future self and advisor."
```

### Experiment Tracking

After each experiment:
```
"Let's update experiments.json with:
- What we changed
- Hypothesis
- Results
- Comparison to baseline
- Next steps

Then update lab_notes.md with observations."
```

---

## Common Pitfalls

### Pitfall: Treating Claude as a Code Generator

**Wrong**:
```
"Write the complete search system."
```

**Right**:
```
"Let's design the search system together:
1. Explain the components needed
2. Implement the core (teach me)
3. I'll add the extensions
4. Review my implementation"
```

### Pitfall: Not Testing Ideas

**Wrong**:
```
"Is approach X better than Y?"
```

**Right**:
```
"Let's test approach X vs Y:
- Hypothesis: X will improve [metric] because [reason]
- Implementation: [quick version]
- Measurement: [how to evaluate]
- Analysis: [what results mean]"
```

### Pitfall: Ignoring the Educational Prompt

**Wrong**:
```
"Just give me working code."
```

**Right**:
```
"Following the Agent 2 approach, let's:
1. Understand the problem
2. Explore solutions with trade-offs
3. Implement with teaching comments
4. Test and iterate
5. Document what was learned"
```

---

## Sample Session Transcript

### Beginning of Week 3

**Student**:
```
Starting week 3. Goal: Implement baseline embedding system.

Current state:
- Have clean data (cards_clean.csv with 27k cards)
- Environment set up

Learning goals:
- Understand how sentence transformers work
- Learn about batch encoding trade-offs
- Get first embeddings generated

Let's start by explaining the architecture, then implement step-by-step.
```

**Expected Claude Response**:
```
Great! Week 3 is exciting - you'll see your first semantic search results.

Let me explain the architecture we'll build:

[Diagram/explanation of: text → tokenizer → model → embeddings → FAISS]

Key concepts to understand:
1. Sentence transformers create fixed-size embeddings (384-dim)
2. Batch encoding trades memory for speed
3. FAISS indexes embeddings for fast similarity search

Here's our plan:
1. Load the model (I'll explain what's happening)
2. Encode a sample (5 cards) to understand output
3. Implement batch encoding for all cards
4. Save embeddings (avoid regeneration)
5. Test: can we retrieve similar cards?

Let's start with step 1. Here's the code with teaching comments...
```

---

## Success Metrics

You're using the Agent 2 prompt well if:

### You Can Explain Your Code
- [ ] Can describe why you chose each approach
- [ ] Understand trade-offs vs. alternatives
- [ ] Know what would break if components changed

### You're Learning, Not Just Building
- [ ] Trying variations to see what changes
- [ ] Reading papers/docs referenced by Claude
- [ ] Asking "why" before "how"

### Your Work Is Reproducible
- [ ] Experiments logged with parameters
- [ ] README lets others run your code
- [ ] Clear documentation of decisions

### You're Progressing Steadily
- [ ] Weekly deliverables achieved
- [ ] Not stuck for days without asking
- [ ] Building on previous weeks' work

---

## Getting Help

### When to Ask for More Explanation

If Claude's response includes terms you don't know:
```
"You mentioned [term]. Can you explain:
1. What it means in simple terms
2. Why it matters for this project
3. Where I can learn more"
```

### When to Push Back

If you disagree or want to try something different:
```
"I see your recommendation, but I'm curious about [alternative].
Can we try that first? I want to understand the trade-off
by experiencing it myself."
```

### When to Ask for Simplification

If overwhelmed:
```
"This feels complex. Can we:
1. Simplify to the core functionality
2. Get that working
3. Then add complexity incrementally

I want to understand each piece before combining."
```

---

## Final Tips

1. **Start Each Session with Context**: Remind Claude what phase you're in and what you're learning

2. **End Sessions with Reflection**: Document what you learned, not just what you built

3. **Embrace Experiments**: Try things, even if they might not work. That's how you learn.

4. **Ask for Alternatives**: "What are three ways to do X?" generates better learning than "What's the best way to do X?"

5. **Connect to Theory**: Ask how implementations relate to concepts from papers or courses

6. **Document Decisions**: Your future self (and advisor) will thank you

7. **Celebrate Learning**: It's okay if something doesn't work - you learned why!

---

## Quick Reference

### Week 1-2: "Explain why we need X"
### Week 3-4: "Show me how to test Y"
### Week 5-6: "Help me analyze Z"
### Week 7-9: "Let's experiment with A vs B"
### Week 10-11: "Should I add feature C?"
### Week 12-14: "Review my documentation"

---

**Remember**: Agent 2 is designed to teach, not just deliver. Engage actively, ask questions, try variations, and document your learning journey. The goal is not just a working system, but deep understanding of ML/NLP that you can apply to future projects.

Good luck with your independent study!
