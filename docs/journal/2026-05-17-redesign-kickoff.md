# 2026-05-17 — Redesign kickoff and pre-Phase-1 cleanup

## Context

Kicking off the research-track redesign after the POC ([retrospective](2026-05-17-poc-retrospective.md)). Today's work is structural, not technical: archive the POC, install the new architecture spec (`CLAUDE.md`), stand up the roadmap and journal scaffolding so subsequent work has somewhere to land. Goal of the session is to leave the repo in a state where Phase 1 ("Foundation & Logging Infrastructure") can start cleanly.

## What happened

1. **Tagged the POC** as `v0.1-poc` on the existing `main` HEAD. The tag message describes the POC's architecture and frames the redesign that follows. Tagging preserves the POC's exact state regardless of how `main` evolves.
2. **Skipped the redesign branch.** Considered cutting a `redesign` branch and developing there; chose linear `main` instead. Rationale: solo project, paper-track, no concurrent contributors. Branch-juggling buys nothing and dirties the history that the paper's version-control footnote will reference. The tag is sufficient archive.
3. **Archived POC code** under `archive/poc_v1/` with a README documenting what each module did and the specific failure it represents. Modules: `src/vector_db/`, `src/training/`, the keyword-dictionary preprocessing, `scripts/fetch_and_process_mtg.py`, the empty `src/web_app/`. The `src/data_processing/download_bulk_data.py` module was kept live — it's reasonable as-is and the new Phase 2 ingest script will eventually supersede it without urgency.
4. **Cleaned committed noise.** Removed `.DS_Store`, `.idea/`, `__pycache__/` from the git index; expanded `.gitignore` to keep them out going forward.
5. **Dropped empty stub files** that were misrepresenting the directory tree: five `tests/test_*.py` files, `docs/api_endpoints.md`, `docs/project_plan.md`, `scripts/update_bulk_data.sh`, `src/data_processing/fetch_bulk_data.py`, `src/data_processing/compile_rulings_data.py`, and the empty `README.md`. Empty test files in particular were a worse-than-nothing condition — they neither pass nor fail.
6. **Migrated the original proposal** (`outline-summary.md` from the iCloud planning directory) to `docs/archive/2025-11-03-original-proposal.md` with a header noting its superseded technical choices.
7. **Installed the redesign guidance**: `CLAUDE.md` at the repo root (replaces the old empty `README` as the primary entry point for any future Claude session), `docs/roadmap/phase-0..7` files describing the phased work, this journal directory with a template and the first two real entries.
8. **Did NOT migrate** the agent-prompt-engineering files (`AGENT-COMPARISON.md`, `agent2-prompt.md`, `agent3-prompt.md`, `README-AGENT2.md`, `USING-AGENT2-PROMPT.md`, `agent2-summary.md`, `prompt-review-analysis.md`). They were meta-process artifacts from the exercise of synthesizing a Claude prompt for the original POC plan. The current `CLAUDE.md` is their surviving descendant; carrying the source materials into the research repo would dilute the archive's signal.

## Decision

Repo is now structured to support the redesign: live code in `src/` and `scripts/` is intentionally minimal (only `src/data_processing/download_bulk_data.py` and `src/config.py` survived the archive pass), guidance lives in `CLAUDE.md` + `docs/`, and the POC is preserved both as a tag and as an annotated archive directory.

## Reasoning

The single most important property after this cleanup is that **future sessions can read `CLAUDE.md` and `docs/roadmap/phase-0-overview.md` and immediately understand both the architecture and where work is currently focused**. The old repo state failed this badly — empty stubs, a `web_app/` directory with no app, a POC training pipeline next to FAISS code, no architecture document.

The decision to delete-rather-than-archive the agent-prompt-engineering files reflects a discipline I want to maintain through the project: the archive is for things the paper might cite or future-us might want to compare against. It is not a junk drawer. Meta-process artifacts about how a prompt was written don't meet that bar.

## Alternatives considered

- **Branch instead of tag for POC preservation.** Rejected — branch maintenance overhead for a solo project, and tags are the standard tool for "this exact state, retrievable forever."
- **Keep the FAISS code live alongside new pgvector code.** Rejected — invites accidental imports and confuses readers. Archive cleanly.
- **Migrate everything from iCloud verbatim.** Rejected — most of the iCloud planning files were prompt-engineering meta-work, not research content.

## Notes for the final report

- **Methodology — version control / reproducibility:** the v0.1-poc tag + `archive/poc_v1/` are reproducibility infrastructure. Worth a sentence in methodology.
- **Background:** `archive/poc_v1/README.md` plus the [POC retrospective](2026-05-17-poc-retrospective.md) are the source material.

## Open follow-ups

- [ ] Phase C / pre-Phase-1 design work: `pyproject.toml`, replacement `src/config.py` (Pydantic Settings), `src/utils/device.py`. Pending user input on schema choices.
- [ ] Decide whether to push these commits to origin/main now or hold until pre-Phase-1 cleanup is fully complete.
- [ ] Decide what to do with the remaining iCloud planning directory once the migration is complete (delete entirely vs. leave as a local-only archive on the iCloud side).

## Related

- [POC retrospective](2026-05-17-poc-retrospective.md) — the failures this redesign responds to
- `CLAUDE.md` — architecture and conventions
- `docs/roadmap/phase-0-overview.md` — phased work plan
