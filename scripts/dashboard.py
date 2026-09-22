"""Local results dashboard — human eyes on the cascade's output.

A single-page web UI over :class:`src.search.Searcher`, for reviewing results
visually and for judging the evaluation set's annotation holes
(journal 2026-09-18 §5e). Stdlib HTTP server, no extra dependencies, binds to
localhost only.

Flow:

1. Type a query (or pick an eval query). **Rewrite** calls the Stage 1 rewriter
   and fills the two editable boxes: the text that will be embedded, and the
   filter JSON. Rewrite needs ``mlx_lm.server``; everything else works without it.
2. Edit either box by hand. **Search** re-runs Stage 2 + 3 on exactly what is
   in the boxes — no rewriter call — so filter tweaks are instant.
3. Results render as a card-image grid (left to right, top down) with a match
   score (cosine similarity as a percentage). When an eval query is selected,
   each card shows whether the eval set already judges it; when a tag slug is
   given, whether the card is in that tag's pool.
4. Judgment buttons under each card append to
   ``data/eval/judgments_pending.jsonl`` — raw material for eval-set v2. The
   curated YAML is never modified from here.

Images are hot-linked from ``cards.scryfall.io`` (Scryfall's image file origin,
which is exempt from API rate limits); nothing is fetched from the Scryfall API.

Every embedder with vectors in ``card_embeddings`` is loaded at startup and
selectable per search, so the same query can be compared across checkpoints
without re-embedding anything (migration 0004).

Usage::

    PYTHONPATH="$PWD" .venv/bin/python scripts/dashboard.py            # http://localhost:8765
    PYTHONPATH="$PWD" .venv/bin/python scripts/dashboard.py --port 9000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import yaml

import src.utils.quiet  # noqa: F401
from src.config import settings
from src.query_rewriter import HyDEError, HyDEFilters, rewrite_query
from src.search import FilterError, FilterPolicy, Searcher

PAGE = Path(__file__).with_name("dashboard.html")
JUDGMENTS = settings.eval_dir / "judgments_pending.jsonl"
EVAL_SET = settings.eval_dir / "queries_v1_draft.yaml"
ORACLE = settings.eval_dir / "tag_label_oracle_v1.yaml"

_CARD_META_SQL = """
    SELECT DISTINCT ON (oracle_id) oracle_id::text,
           COALESCE(raw->'image_uris'->>'normal',
                    raw->'card_faces'->0->'image_uris'->>'normal') AS image,
           raw->>'scryfall_uri', (raw->>'edhrec_rank')::int
    FROM cards WHERE oracle_id::text = ANY(%s) ORDER BY oracle_id, face_index
"""

_POOL_SQL = """
    SELECT DISTINCT ct.oracle_id::text
    FROM oracle_tags t
    JOIN oracle_tag_closure cl ON cl.ancestor_id = t.id
    JOIN card_tags ct ON ct.tag_id = cl.descendant_id
    WHERE t.slug = %s AND ct.oracle_id::text = ANY(%s)
"""


_VERSIONS_SQL = "SELECT DISTINCT embedding_version FROM card_embeddings ORDER BY 1"


class App:
    """Holds one searcher per available embedder and the eval set."""

    def __init__(self) -> None:
        import psycopg

        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute(_VERSIONS_SQL)
            versions = [r[0] for r in cur.fetchall()]
        self.searchers: dict[str, Searcher] = {}
        for v in versions:
            model = v.split("|preproc=")[0]
            print(f"  Loading embedder {model} ...", flush=True)
            self.searchers[v] = Searcher(embedding_model=model)
        default = settings.embedding_version
        self.default_version = default if default in self.searchers else versions[0]
        self.searcher = self.searchers[self.default_version]
        ev = yaml.safe_load(EVAL_SET.read_text(encoding="utf-8"))
        oracle = (
            yaml.safe_load(ORACLE.read_text(encoding="utf-8"))["queries"] if ORACLE.exists() else {}
        )
        self.eval_version = ev.get("version", "v1-draft")
        self.eval: dict[str, dict[str, Any]] = {}
        for q in ev["queries"]:
            self.eval[q["id"]] = {
                "id": q["id"],
                "query": q["query"],
                "category": q.get("category"),
                "relevant": {r["id"] for r in q.get("relevant") or []},
                "borderline": {r["id"] for r in q.get("borderline") or []},
                "tag": ((oracle.get(q["id"]) or {}).get("tags") or [None])[0],
            }

    # ---- endpoints ----

    def _pick(self, body: dict[str, Any]) -> Searcher:
        v = body.get("embedding_version") or self.default_version
        if v not in self.searchers:
            raise ValueError(f"unknown embedder {v!r}; available: {sorted(self.searchers)}")
        return self.searchers[v]

    def meta(self) -> dict[str, Any]:
        return {
            "embedding_version": self.default_version,
            "embedding_versions": sorted(self.searchers),
            "hyde_model": settings.hyde_model,
            "prompt": self.searcher.prompt_version,
            "eval_version": self.eval_version,
            "eval_queries": [
                {k: v for k, v in q.items() if k in ("id", "query", "category", "tag")}
                | {"relevant_count": len(q["relevant"])}
                for q in self.eval.values()
            ],
        }

    def rewrite(self, body: dict[str, Any]) -> dict[str, Any]:
        result = rewrite_query(body["query"], prompt_path=self.searcher.prompt_path)
        return {
            "filters": result.filters.model_dump(exclude_none=True) if result.filters else {},
            "hypothetical_card": result.hypothetical_card or "",
        }

    def search(self, body: dict[str, Any]) -> dict[str, Any]:
        text = (body.get("embed_text") or "").strip()
        if not text:
            raise ValueError("Nothing to embed: the 'text to embed' box is empty.")
        raw_filters = body.get("filters") or {}
        filters = HyDEFilters.model_validate(raw_filters) if raw_filters else None
        policy = FilterPolicy(keywords=bool(body.get("keywords_filter")))
        k = max(1, min(int(body.get("k") or 20), 200))
        searcher = self._pick(body)
        res = searcher.search_prepared(text, filters, k=k, policy=policy)

        ids = [h.oracle_id for h in res.hits]
        ev = self.eval.get(body.get("eval_id") or "")
        tag = (body.get("tag_slug") or "").strip()
        with searcher.conn.cursor() as cur:
            cur.execute(_CARD_META_SQL, (ids,))
            meta = {r[0]: r[1:] for r in cur.fetchall()}
            pool: set[str] = set()
            if tag:
                cur.execute(_POOL_SQL, (tag, ids))
                pool = {r[0] for r in cur.fetchall()}
        prior = _load_judgments(body.get("eval_id"), text)

        hits = []
        for rank, h in enumerate(res.hits, 1):
            image, uri, edhrec = meta.get(h.oracle_id, (None, None, None))
            judged = None
            if ev:
                judged = (
                    "relevant"
                    if h.oracle_id in ev["relevant"]
                    else "borderline"
                    if h.oracle_id in ev["borderline"]
                    else "unjudged"
                )
            hits.append(
                {
                    "rank": rank,
                    "oracle_id": h.oracle_id,
                    "name": h.name,
                    "type_line": h.type_line,
                    "mana_cost": h.mana_cost,
                    "oracle_text": h.oracle_text,
                    "score": round((1 - h.distance) * 100, 1),
                    "image": image,
                    "scryfall_uri": uri,
                    "edhrec_rank": edhrec,
                    "eval_judgment": judged,
                    "in_tag_pool": (h.oracle_id in pool) if tag else None,
                    "my_judgment": prior.get(h.oracle_id),
                }
            )
        summary: dict[str, Any] = {}
        if ev:
            summary["judged_relevant"] = sum(h["eval_judgment"] == "relevant" for h in hits)
            summary["unjudged"] = sum(h["eval_judgment"] == "unjudged" for h in hits)
        if tag:
            summary["in_tag_pool"] = sum(bool(h["in_tag_pool"]) for h in hits)
        return {
            "hits": hits,
            "candidate_count": res.candidate_count,
            "where_sql": res.where_sql,
            "timings_ms": res.timings_ms,
            "warnings": res.warnings,
            "summary": summary,
            "embedding_version": searcher.embedding_version,
        }

    def judge(self, body: dict[str, Any]) -> dict[str, Any]:
        label = body["label"]
        if label not in ("relevant", "borderline", "not_relevant"):
            raise ValueError(f"unknown label {label!r}")
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "eval_id": body.get("eval_id"),
            "query": body.get("query"),
            "embed_text": body.get("embed_text"),
            "oracle_id": body["oracle_id"],
            "name": body.get("name"),
            "label": label,
            "rank": body.get("rank"),
            "score": body.get("score"),
            "embedding_version": body.get("embedding_version") or self.default_version,
        }
        with JUDGMENTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True}


def _load_judgments(eval_id: str | None, embed_text: str) -> dict[str, str]:
    """Latest pending judgment per card for this eval query (or this exact text)."""
    out: dict[str, str] = {}
    if not JUDGMENTS.exists():
        return out
    for line in JUDGMENTS.open(encoding="utf-8"):
        r = json.loads(line)
        same = (eval_id and r.get("eval_id") == eval_id) or (
            not eval_id and not r.get("eval_id") and r.get("embed_text") == embed_text
        )
        if same:
            out[r["oracle_id"]] = r["label"]
    return out


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    routes = {"/api/rewrite": app.rewrite, "/api/search": app.search, "/api/judge": app.judge}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/meta":
                self._json(200, app.meta())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            fn = routes.get(self.path)
            if fn is None:
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
                )
                self._json(200, fn(body))
            except HyDEError as exc:
                self._json(
                    502, {"error": f"Rewriter unavailable — is mlx_lm.server running? ({exc})"}
                )
            except (FilterError, ValueError, KeyError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                for s_ in app.searchers.values():
                    s_.conn.rollback()
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            if "/api/" in (args[0] if args else ""):
                sys.stderr.write(f"  {self.log_date_time_string()}  {fmt % args}\n")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print("  Loading embedder ...", flush=True)
    app = App()
    # Single-threaded on purpose: one model, one DB connection, one reviewer.
    server = HTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"  Embedders: {', '.join(sorted(app.searchers))}  (default {app.default_version})")
    print(f"  Dashboard: http://localhost:{args.port}   (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        for s_ in app.searchers.values():
            s_.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
