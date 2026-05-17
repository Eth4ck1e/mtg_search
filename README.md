# mtg_search

Natural-language semantic search for Magic: The Gathering cards. CSCI 5953 independent study, CSUSB.

**Status:** redesign in progress. The original proof-of-concept (FAISS + raw DistilBERT) is preserved under tag `v0.1-poc` and archived at `archive/poc_v1/`. The current architecture is a **three-tower retrieval system** (SQL pre-filter + HyDE query rewriting + semantic vector search on pgvector). See `CLAUDE.md` for the architecture spec and `docs/roadmap/phase-0-overview.md` for the phased work plan.

## Why this project exists

MTG players think in informal terms: "a blue counterspell that costs 2", "cards that flicker creatures", "cheap red removal". Card text on the cards themselves is formal, prose-y, and 30+ years of terminology drift mean modern players type words that don't appear on older cards. Traditional keyword search misses this; the POC showed that naive vector search on raw card text misses it too. This project measures and addresses the **query-document asymmetry** at the heart of that mismatch.

## Repo layout

```
.
├── CLAUDE.md                # Architecture spec + working conventions (read first)
├── archive/poc_v1/          # Original POC, preserved for the paper's Background section
├── data/                    # Raw Scryfall dumps (gitignored), eval set, keyword dict
├── docs/
│   ├── archive/             # Older proposal docs, preserved for traceability
│   ├── journal/             # Dated decisions and analysis writeups
│   └── roadmap/             # Phase-by-phase task breakdown
├── scripts/                 # Entry-point scripts (ingest, embed, evaluate, ...)
├── src/                     # Library code
└── tests/
```

## Quickstart

> The redesign is at Phase 1. None of the new scripts exist yet. Quickstart will be filled in as Phase 1 lands.

## Where to look next

- New here? → `CLAUDE.md`
- What's the current phase? → `docs/roadmap/phase-0-overview.md`
- Why does it look this way? → `docs/archive/2025-11-03-original-proposal.md` + `archive/poc_v1/README.md`
- What was just decided? → `docs/journal/` (most recent dated entry)
