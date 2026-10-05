# 2026-10-05 — Source library and citation agent

## Decision

Before the paper is written, claims about sources get the same treatment numbers got on 2026-10-03: they come from a tool that reads the artifact, not from memory. Two pieces were added.

1. **A source library** — the text of every PDF in `docs/sources/`, indexed in the project's own Postgres + pgvector.
2. **A citation agent** (`.claude/agents/citation-librarian.md`) restricted to that library, which handles lookups, quotations and citation checks, and reports "not found" instead of guessing.

The final review of sources is deferred until the draft exists, and will cover only the sources the paper actually cites (Mitchell, 2026-10-05).

## Why

The project already has two recorded failures of the same kind: the fabricated May baseline (superseded 2026-09-11), and on 2026-10-03 hand-copied numbers in the draft that came from a 21-query run labelled as 26, plus two citations attributed to the wrong authors ("Wang (2024)" was Zhang et al.; "Nigam et al. 2020 (Amazon)" was Choi et al. 2020, Home Depot). Each was caught by checking against the artifact. A language model asked what a paper says will produce a fluent answer whether or not it has read the paper. The fix is structural: make the lookup cheap and make the quotation check mechanical.

## What was built

| Piece | File | What it does |
|---|---|---|
| Tables | `src/db/migrations/0005_source_library.sql` | `source_pages` (full text per PDF page: ground truth) and `source_chunks` (passages with a 768-d embedding and a full-text index). |
| Indexer | `scripts/index_sources.py` | `pdftotext` per page → paragraph passages (~160 words, max 260) → stock Nomic embeddings with the `search_document: ` prefix. Full replace; logs a pipeline run. Maps each PDF to its BibTeX key through the arXiv id in `docs/sources/README.md`; a PDF with no key is skipped, so nothing unciteable is searchable. |
| Library | `src/source_library.py` | `search()` (semantic or exact-word), `match_quote()` / `verify_quote()`, `page_text()`. |
| CLI | `scripts/source_search.py` | `"question"`, `--exact`, `--source <key>`, `--page <key> <n>`, `--verify "quote"`, `--list`. |
| Agent | `.claude/agents/citation-librarian.md` | Tools: Bash, Read, Grep, Glob. Rules below. |

First index: 35 PDFs, 532 pages, 1,745 passages, all 35 mapped to a BibTeX key.

### Design choices

- **Same database, not a separate RAG stack.** The project's own argument is that pgvector is sufficient at this scale; 1,745 passages is far inside it. No new dependency, no new service.
- **Stock Nomic, not the MTG-tuned checkpoint.** The sources are general academic prose. The tuned model moved away from general text slightly (NanoBEIR 0.514 → 0.495) and was trained on card text.
- **Semantic and exact-word search both.** Semantic finds a passage when the wording is unknown; full-text finds a specific term or number.
- **Verification is string matching, not a model.** `--verify` normalises only what PDF extraction changes (line breaks, hyphenation at line ends, ligatures, curly quotes and dashes, letter case), then requires the quote to appear in the named source. It runs over a file's pages joined in order, so a quote crossing a page break verifies and the start page is reported. A second pass ignoring hyphens only catches compounds that extraction joined ("surfacelevel"); it is flagged so the hyphens are copied from the page.
- **Pages are stored whole** so a verified quote can be copied with original capitalisation (`--page`) and so the check does not depend on how passages were cut.

### Checks run

| Test | Result |
|---|---|
| True sentence from the HyDE abstract, spanning three extracted lines, `--source gao2022precise` | VERIFIED, p. 1 |
| Same sentence with one word changed ("ground" → "grounds") | NOT FOUND |
| A paraphrase of HyDE, no source given | NOT FOUND |
| True HyDE sentence checked against `karpukhin2020dense` | NOT FOUND |
| True HyDE sentence, no source given | VERIFIED, attributed to `gao2022precise` p. 1 |

Eight unit tests cover the same cases on fixed text (`tests/test_source_library.py`).

## Agent rules (summary)

Never answer from memory; every quotation passes `--verify` against the cited source and is copied from `--page`; a paraphrased claim gets a verdict (supported / partly / not supported / contradicted) with the passage and page; a paper's own findings are distinguished from its summary of other work; "not found" is a complete answer; no invented references; no paper prose (Mitchell writes); read-only.

## Known limits

- **Seven references have no PDF and cannot be checked:** `hu2021lora`, `muennighoff2022mteb`, `ouyang2022training`, `jarvelin2002cumulated`, `voorhees2000variations`, `sormunen2002liberal`, `wang2011cascade`. Add the PDF to `docs/sources/` (with a README entry) and re-index, or check those by hand. "Guo et al., 2016" remains unresolved.
- **Tables and equations extract badly.** Numbers in tables lose their row and column labels; the agent is told to read the PDF page itself for any table value.
- **Page numbers are PDF page numbers**, and most sources are arXiv preprints: a quote cited to the published venue may sit on a different page there.
- **Verification proves the words are in the source, not that they mean what the sentence around them claims.** Context (a negation in the previous sentence, a claim about someone else's work) still needs a reader. The agent reports surrounding text and flags related-work passages; the final judgment is Mitchell's.
- The agent definition is loaded when a Claude Code session starts; it was written in this session and has not yet been run as an agent. The tool underneath it was tested directly.

## For the paper

Not paper content. Relevant to the methodology's reproducibility note only if the paper describes its own tooling.
