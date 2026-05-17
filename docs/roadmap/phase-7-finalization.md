# Phase 7 — Finalization & Paper

**Weeks:** 14–15 (and beyond if the conference target is end-of-Fall 2026 or Spring 2027)
**Status:** Not started
**Depends on:** Phase 6 closed with measured results

## Goal

Convert the project into something a conference committee can read and a future student can reproduce. Three artifacts: a final paper draft, a presentation deck, and a clean public repo with a real `README.md` that walks a stranger from clone to working search. The journal entries and `experiment_runs` rows are the raw material — Phase 7's job is largely curation and prose, not new technical work.

## Deliverables

- [ ] Final report — `docs/paper/main.md` (or LaTeX if conference requires)
- [ ] Presentation slides — `docs/presentation/`
- [ ] (Optional) 3–5 minute demo video — link in README
- [ ] Final auto-generated comparison report — `docs/reports/final.md`
- [ ] Final `README.md` — quickstart that actually works on a clean clone
- [ ] Frozen evaluation set + frozen `experiment_runs` snapshot in `data/eval/` and `data/reports/` so the numbers in the paper are bit-for-bit reproducible from the repo
- [ ] License file (MIT default unless CSUSB requires otherwise)
- [ ] (Optional) HuggingFace upload of any fine-tuned model from Phase 6, with a model card

## Sub-tasks

### Paper
1. [ ] Pull together a section-by-section outline. The section map in `phase-0-overview.md` says where each phase's notes feed; collect them.
2. [ ] **Abstract** — 200 words. Problem, contribution (three-tower architecture), main result (the headline delta from Phase 5 or Phase 6), conclusion.
3. [ ] **Introduction** — motivate from the player's perspective ("a blue counterspell that costs 2"), introduce query-document asymmetry, name the architecture, preview the result.
4. [ ] **Background / Related work** — DPR, HyDE (Gao et al. 2022), sentence-transformers, pgvector, any prior MTG-specific information retrieval work. Cite generously.
5. [ ] **Methodology** — corpus & schema (Phase 1, 2), embedding pipeline (Phase 2, 3), evaluation design (Phase 3), three-tower retrieval (Phase 4), optimization (Phase 6).
6. [ ] **Experiments** — the configuration sweep (Phase 5), ablations (Phase 5), optimization comparison (Phase 6).
7. [ ] **Results** — the comparison tables, per-category breakdowns, ablation table. Headline numbers in prose.
8. [ ] **Discussion** — failure analysis, limitations, what the numbers do and do not show.
9. [ ] **Future work** — what Phase 6 didn't address; cross-domain generalization (this isn't really MTG-specific — the asymmetry framing applies to any retrieval task with informal queries and formal documents); fine-tuning if not done; multilingual cards; the PHP frontend.
10. [ ] **Conclusion** — restate the contribution; one paragraph on what the project taught us as a research process.
11. [ ] **References** — BibTeX. Maintain `docs/paper/references.bib` from Phase 1 onward; don't leave it to the end.
12. [ ] **Appendix** — full evaluation set, full HyDE prompt, full results table, additional failure examples.

### Presentation
13. [ ] 20-minute slot, give or take. 15 minutes of slides + 5 minutes of demo + Q&A.
14. [ ] Slide structure: title, problem, motivating example, architecture diagram, key result chart, ablation chart, demo, limitations, future work.
15. [ ] One **memorable example query** carried through the deck — pick something where the system clearly shines (jargon + constraint combination is usually the cleanest demo).
16. [ ] Demo: live or recorded? Recorded is safer for a conference; live is more impressive if the network is reliable. Decide based on venue.

### Reproducibility hygiene
17. [ ] Real `README.md` quickstart. **Test it on a fresh clone.** If the README says `python scripts/ingest.py`, run that command on a fresh checkout and confirm it works end-to-end. This is the single most common gap in academic-project repos.
18. [ ] `pyproject.toml` pinned versions (or a `requirements.lock`) so dependencies don't drift.
19. [ ] One-line repro: ideally `make demo` or `python scripts/demo.py` produces a working search over a small sample of cards without needing to download the full Scryfall bulk file.
20. [ ] `experiment_runs` table exported to a CSV under `data/reports/experiment_runs_final.csv` so the paper's numbers are reproducible without re-running the pipeline.

### Code hygiene
21. [ ] Run `ruff check` and `ruff format` across the whole codebase. Fix or `# noqa:` everything.
22. [ ] If using mypy: run it. Fix or suppress.
23. [ ] Walk each module's docstrings: do they describe what the module does, not what the module used to do?
24. [ ] Read each script's `--help` output. Is it useful?
25. [ ] License file. README badge.

### Conference logistics
26. [ ] Identify the target conference. Check submission deadline, page limit, formatting requirements.
27. [ ] Reformat to required template if necessary (IEEE, ACM, Springer).
28. [ ] Submit. Save the submission artifact in `docs/submissions/<conference>/<date>/`.

## Decisions log

| Date | Decision | Reasoning | Alternatives considered |
|---|---|---|---|
| | | | |

## Open questions

- LaTeX vs. Markdown for the paper. Markdown is easier to maintain alongside the rest of the project; some venues require LaTeX. Decide once a conference target is locked in.
- HuggingFace upload of fine-tuned model: yes/no. Adds reproducibility surface area but also a maintenance burden if the venue is asynchronous (people emailing about it).
- Should the paper include a "lessons learned" or "process" appendix discussing the evaluation-first methodology as a meta-contribution? It's defensible and not common in student work.

## Notes for final report (meta)

This is where the report-generation script earns out. Most of the methodology section is paraphrasing the roadmap files; most of the results section is wrapping prose around `generate_report.py` output. If the journal discipline held throughout, this phase should be heavy on prose and synthesis, light on new technical work.

If a section feels hard to write, that often means the underlying work wasn't documented well enough — go back and read the relevant journal entries, and if they're thin, that's a signal that future phases should journal more carefully.

## Journal entries

- (none yet)
