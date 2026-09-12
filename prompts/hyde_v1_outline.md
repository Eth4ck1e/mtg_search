# HyDE Prompt — v1 Outline

**Status:** scaffold with DB schema reference + JSON output schema + placeholder few-shot slots. The actual few-shot example content is yours to write — those are the load-bearing artifact that shapes retrieval quality. The scaffold below is safe to lift and iterate on.

**File this becomes:** `prompts/hyde_v1.yaml` (or `.txt` / `.json`, whichever your `src/query_rewriter.py` expects). Version as `v2`, `v3` as prompt iterations produce different `experiment_runs` rows.

**Reference sources:**
- Gao et al. 2022 (HyDE) — `docs/sources/2022_gao_hyde.pdf`
- Wang 2024 (Best Practices for LLM Query Expansion) — `docs/sources/2024_wang_query-expansion-best-practices.pdf`

---

## 1. Target model

- **Default:** `meta-llama/Llama-3.1-8B-Instruct` (via `settings.hyde_model` — override in `.env` if the M4 candidate eval picks a different model)
- **Runtime:** local inference on Apple Silicon M3 via `llama.cpp` / MLX / Ollama (TBD by M4 sub-task #27)
- **Prompt format:** instruction-tuned chat template (system + user role) — the exact template depends on which serving framework you pick

---

## 2. Task specification

HyDE receives a **natural-language user query** and must produce **structured JSON output** with two fields:

- `filters` — structured attribute filters passed to Stage 2 (SQL pre-filter)
- `hypothetical_card` — hypothetical MTG card ability text passed to Stage 3 (semantic vector search after embedding)

Both fields must be present. Either may contain null / empty content when the query does not constrain that dimension (e.g., a pure ability-text query with no color/type constraints has `filters: null` for most fields; a purely structural query — "1-mana instants" — may have an empty `hypothetical_card`).

---

## 3. Full DB schema reference (paste into system prompt)

Below is the exact filterable schema from `src/db/migrations/0002_cards.sql`. **HyDE must only produce filter values within these enumerations — the SQL pre-filter validates against this schema.**

### Per-face columns (queryable at face granularity)

| Column | Type | Valid values / format | Filter example |
|---|---|---|---|
| `name` | TEXT | Any card name | Rarely a HyDE-extracted filter; user usually asks by name only if they already know the card |
| `mana_cost` | TEXT | Symbol string like `{2}{U}{U}`, `{X}{R}`, or `{0}` | Prefer `cmc` for filters; `mana_cost` is display-oriented |
| `colors` | TEXT[] | Array of `W`, `U`, `B`, `R`, `G` (any subset). Colorless = empty array. | `WHERE colors && ARRAY['R']` (has red), `WHERE colors <@ ARRAY['W','U']` (only white/blue) |
| `type_line` | TEXT | Full type string like `Legendary Creature — Elf Druid`, `Instant`, `Artifact`, `Land` | `WHERE type_line LIKE '%Creature%'`, `WHERE type_line LIKE '%Instant%'` |
| `oracle_text` | TEXT | The card's rules text (may be empty for vanilla creatures) | NOT filtered on directly — that's Stage 3's job |
| `power` | TEXT | Creature power. **TEXT** (not numeric) because `*` is a valid value (variable power). | `WHERE power::int >= 4` (cast; nulls skipped) |
| `toughness` | TEXT | Creature toughness. Same as power. | `WHERE toughness::int <= 2` |
| `loyalty` | TEXT | Planeswalker starting loyalty | `WHERE loyalty::int = 3` |

### Card-level columns (identical across faces of the same card)

| Column | Type | Valid values / format | Filter example |
|---|---|---|---|
| `cmc` | NUMERIC(4,2) | Converted mana cost — total mana value. NUMERIC because half-mana Un-set costs exist. | `WHERE cmc <= 2`, `WHERE cmc = 3`, `WHERE cmc BETWEEN 1 AND 4` |
| `color_identity` | TEXT[] | Same values as `colors`. Deck-builder/commander color identity (includes mana-symbol mentions in rules text). | `WHERE color_identity <@ ARRAY['W','U','B']` (Esper-compatible) |
| `keywords` | TEXT[] | Scryfall-parsed canonical keyword list — case-sensitive strings like `Flying`, `First strike`, `Trample`, `Deathtouch`, `Menace`, `Vigilance`, `Lifelink`, `Reach`, `Hexproof`, `Indestructible`, `Ward`, `Prowess`, `Landwalk`, `Flash`, `Haste`, `Defender`, `Fear`, `Intimidate`, `Shroud`, etc. | `WHERE 'Flying' = ANY(keywords)`, `WHERE keywords && ARRAY['Trample','First strike']` |
| `layout` | TEXT | `normal`, `transform`, `modal_dfc`, `split`, `adventure`, `flip`, `meld`, `prepare`, `saga`, `class`, `case`, `battle`, `mutate`, `leveler`, `augment`, `host`, `reversible_card`, `double_faced_token` (filtered out), etc. | `WHERE layout = 'normal'` (most cards) |
| `released_at` | DATE | Release date of the printing | `WHERE released_at >= '2020-01-01'` |
| `legalities` | JSONB | Object with keys per format: `standard`, `modern`, `pioneer`, `legacy`, `vintage`, `commander`, `pauper`, `historic`, `explorer`, `alchemy`, `brawl`, `historicbrawl`, `paupercommander`, `duel`, `oldschool`, `premodern`, `predh`, `future`, `oathbreaker`, `standardbrawl`, `timeless`. Values: `legal`, `not_legal`, `restricted`, `banned`. | `WHERE legalities->>'commander' = 'legal'`, `WHERE legalities->>'modern' != 'banned'` |
| `raw` | JSONB | Escape hatch — full Scryfall record. Use for `set`, `set_type`, `promo_types`, or anything not promoted to a column. | `WHERE raw->>'set_type' = 'expansion'` |

### Filter attribute conventions the prompt should enforce

- **Color arrays**: always uppercase single letters. Never `"white"` — always `"W"`.
- **Types**: use the substring that appears in `type_line` — `"Creature"`, `"Instant"`, `"Sorcery"`, `"Artifact"`, `"Enchantment"`, `"Land"`, `"Planeswalker"`, `"Battle"`, `"Tribal"`. For subtypes (`"Elf"`, `"Goblin"`, `"Wizard"`), use LIKE on the `type_line` — most subtypes appear after the `—` separator.
- **Keywords**: exact case, matches Scryfall's parsed list. `"Flying"` not `"flying"`, `"First strike"` not `"first-strike"`.
- **CMC comparisons**: use JSON operator objects like `{"op": "<=", "value": 2}` or `{"op": "=", "value": 3}` — the prompt should extract inequality intent, not just literal numbers.

---

## 4. JSON output schema

The HyDE model must produce output matching this JSON schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["filters", "hypothetical_card"],
  "properties": {
    "filters": {
      "type": ["object", "null"],
      "properties": {
        "colors": {
          "type": ["array", "null"],
          "items": {"enum": ["W", "U", "B", "R", "G"]},
          "uniqueItems": true
        },
        "colors_op": {
          "type": ["string", "null"],
          "enum": ["contains_any", "contains_all", "subset_of", "exactly"],
          "description": "How to match the colors array"
        },
        "color_identity": {
          "type": ["array", "null"],
          "items": {"enum": ["W", "U", "B", "R", "G"]},
          "uniqueItems": true
        },
        "types": {
          "type": ["array", "null"],
          "items": {"enum": ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land", "Planeswalker", "Battle", "Tribal"]}
        },
        "subtypes": {
          "type": ["array", "null"],
          "items": {"type": "string"},
          "description": "e.g., ['Elf', 'Wizard'] — LIKE-matched on type_line after the em-dash"
        },
        "cmc": {
          "type": ["object", "null"],
          "properties": {
            "op": {"enum": ["<=", "=", ">=", "<", ">", "between"]},
            "value": {"type": ["number", "array"]}
          },
          "required": ["op", "value"]
        },
        "keywords": {
          "type": ["array", "null"],
          "items": {"type": "string"},
          "description": "Canonical Scryfall keyword names, case-sensitive"
        },
        "power": {
          "type": ["object", "null"],
          "properties": {
            "op": {"enum": ["<=", "=", ">=", "<", ">"]},
            "value": {"type": "number"}
          }
        },
        "toughness": {
          "type": ["object", "null"],
          "properties": {
            "op": {"enum": ["<=", "=", ">=", "<", ">"]},
            "value": {"type": "number"}
          }
        },
        "format_legality": {
          "type": ["object", "null"],
          "properties": {
            "format": {"enum": ["standard", "modern", "pioneer", "legacy", "vintage", "commander", "pauper", "historic", "alchemy", "brawl"]},
            "status": {"enum": ["legal", "not_legal", "banned", "restricted"]}
          }
        }
      }
    },
    "hypothetical_card": {
      "type": ["string", "null"],
      "description": "Rules-text portion of a hypothetical MTG card that would perfectly match this query. Empty string or null if the query has no ability-text component."
    }
  }
}
```

Any field the query does not constrain: set to `null`. **Do not invent constraints** the query doesn't specify — over-constraining kills recall.

---

## 5. System prompt scaffold

Paste the schema reference above into the system prompt, followed by task framing:

```
You are a search-query rewriter for a Magic: The Gathering card database.

Given a user's natural-language query, produce a JSON object with two fields:
  1. "filters" — structured attribute filters extractable from the query
  2. "hypothetical_card" — the rules text of an imaginary MTG card that
     would perfectly match this query

The database schema and valid filter values are documented below. Follow
these rules strictly:

RULES
  - Output valid JSON matching the schema. No prose, no code fences, no commentary.
  - Set fields to null when the query does not constrain them.
  - Do not invent constraints the query does not specify.
  - "hypothetical_card" text should read like actual Wizards-authored MTG card
    text — concise, canonical terminology ("Destroy target creature", "Deal N
    damage to any target", "Draw a card", "Enters with N +1/+1 counters",
    "Whenever this creature enters the battlefield, [effect]").
  - Do not include the card's name, mana cost, or type line in "hypothetical_card"
    — those are in "filters". Only rules text.
  - Do not hallucinate specific card names. The hypothetical card is generic.

DATABASE SCHEMA (see docs — [paste schema summary from Section 3 above])

VALID FILTER VALUES
  colors, color_identity: W, U, B, R, G  (arrays; can be empty)
  types: Creature, Instant, Sorcery, Artifact, Enchantment, Land, Planeswalker, Battle, Tribal
  keywords (partial list): Flying, First strike, Trample, Deathtouch, Menace,
      Vigilance, Lifelink, Reach, Hexproof, Indestructible, Ward, Prowess,
      Flash, Haste, Defender  (case-sensitive, canonical Scryfall parsing)
  cmc: NUMERIC. Common values 0-8.
  formats: standard, modern, pioneer, legacy, vintage, commander, pauper,
      historic, alchemy, brawl

EXAMPLES
  [insert your few-shot examples here — see Section 6]
```

---

## 6. Few-shot example slots (YOU fill these)

**Coverage target:** 5–8 examples spanning your six query categories. One example per category is a defensible starting point; the empirical study of best-practices in LLM query expansion (Wang 2024) shows returns diminish sharply past ~5–8 for tasks of this shape.

Below are **placeholder skeletons** showing the format. **You write the actual example content** — the quality of these few-shots directly determines HyDE's output quality and is the load-bearing artifact of this stage.

### Example 1 — Natural-language category

**Query:** [a query in normal English, e.g., "cards that let me look at the top few cards of my library and put some back"]

**Expected output:**

```json
{
  "filters": null,
  "hypothetical_card": "[a hypothetical card's ability text — you write this — should look like real MTG rules text for what this query would match]"
}
```

### Example 2 — Jargon category

**Query:** [a query using MTG jargon, e.g., "cheap red removal" or "flicker effects" or "ramp"]

**Expected output:**

```json
{
  "filters": {
    "colors": ["R"],
    "colors_op": "contains_any",
    "cmc": {"op": "<=", "value": 2},
    "types": ["Instant", "Sorcery"]
  },
  "hypothetical_card": "[e.g., 'Deal 3 damage to any target.' for 'cheap red removal']"
}
```

### Example 3 — Fragmented category

**Query:** [a terse fragment, e.g., "wheels", "tutor", "counters matter"]

**Expected output:**

```json
{
  "filters": null,
  "hypothetical_card": "[the canonical mechanic these words refer to, in Wizards-authored style]"
}
```

### Example 4 — Hybrid category

**Query:** [structured + jargon, e.g., "cheap blue counterspell that also draws a card"]

**Expected output:**

```json
{
  "filters": {
    "colors": ["U"],
    "colors_op": "contains_any",
    "cmc": {"op": "<=", "value": 2},
    "types": ["Instant"]
  },
  "hypothetical_card": "Counter target spell. Draw a card."
}
```

### Example 5 — Constrained category

**Query:** [heavily structural, e.g., "blue creatures under 3 mana with flying"]

**Expected output:**

```json
{
  "filters": {
    "colors": ["U"],
    "colors_op": "contains_any",
    "cmc": {"op": "<", "value": 3},
    "types": ["Creature"],
    "keywords": ["Flying"]
  },
  "hypothetical_card": null
}
```

### Example 6 — Mechanical / edge case

**Query:** [an unusual case, e.g., "cards that reference a specific keyword combination", or "vanilla creatures"]

**Expected output:**

```json
{
  "filters": {"[whatever fits]": "..."},
  "hypothetical_card": "[or empty if purely structural]"
}
```

**Guidance for writing examples:**

- **Every example matters.** Diminishing returns kick in past ~5–8 examples, but each example is a training signal for the LLM's output distribution. Avoid similar-looking examples — cover the diversity of query shapes.
- **Write hypothetical_card in canonical Wizards-authored voice.** Study real card rules text before writing these. E.g., "Deal 3 damage to any target." not "Zap a creature for 3." The closer to Wizards' voice, the closer the embedding to real cards.
- **Show both over-constrained and under-constrained edge cases.** Include an example where a query mentions "blue" but the intended filter is `color_identity` (deck-building) rather than `colors` (spell-cost); include an example where the user says "cheap" and you interpret it as `cmc <= 2` (soft — many players consider "cheap" 3 or less).
- **Include null-heavy examples.** Not every field applies to every query. Show the model that `null` is a valid, expected value.

---

## 7. Iteration guidance

**Version the prompt.** Save as `prompts/hyde_v1.yaml`. Each meaningful prompt change → new version (`v2`, `v3`) → new `experiment_runs` row. Journal the prompt-version delta and reasoning per version — this is source material for the paper's Methodology / Prompt Design subsection.

**What to measure between versions:**

- Overall recall@10 change against the eval set
- Per-query-category improvements (which categories does the new prompt improve?)
- Filter-validity rate (does the prompt produce schema-valid JSON, or does it hallucinate invalid values?)
- Latency (does the prompt length affect LLM inference time meaningfully?)

**When to stop iterating:**

- Diminishing returns on recall@10 (per-version improvement < 0.02 for two versions in a row)
- Latency budget exceeded (>2s p95 per `docs/roadmap/phase-4-hyde-and-prefilter.md` line 41)
- The prompt has grown to >2000 tokens — beyond that, few-shot examples start hurting via position bias

**Common failure modes to watch for:**

- **Over-constraining filters.** The LLM adds a color or type constraint the query didn't explicitly mention. Recall craters.
- **Hallucinated card names in `hypothetical_card`.** The LLM writes "Lightning Bolt deals 3 damage" instead of "Deal 3 damage to any target." This actually might work for recall (lexical overlap with the real card) but is stylistically wrong and could inflate results in ways that don't generalize.
- **Empty output on complex queries.** LLM refuses or returns `{}` when the query is ambiguous. Prompt should explicitly instruct: partial output is better than no output.
- **Schema violations.** LLM invents filter values like `"cheap"` or `"medium mana cost"` instead of numeric CMC comparisons. Prompt should show this-is-not-valid via at least one negative example.

---

## 8. Where this lives in the pipeline

1. User submits NL query
2. **HyDE (this prompt)** produces JSON output
3. `filters` → SQL WHERE clause construction → pre-filtered candidate set of cards
4. `hypothetical_card` → embedding via `format_for_nomic_query(...)` + Nomic Embed v1.5 → 768-dim query vector
5. `pgvector <=> ` cosine search over the pre-filtered candidate set → top-K
6. Return ranked list to user

Reference implementation lives in `src/query_rewriter.py` (to be written) and `src/search.py` (to be written) per `docs/roadmap/phase-4-hyde-and-prefilter.md`.
