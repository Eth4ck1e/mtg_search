"""Generate the results report (tables + figures) from ``experiment_runs``.

The paper's Results section and the presentation are built from this output,
not from numbers copied by hand (CLAUDE.md §12). Everything here is derived
from logged rows; nothing is re-measured.

Row selection: for each ``(config_name, embedder)`` the **newest** eval row
since ``--since`` wins — re-running a config supersedes the old row, and the
superseded ids are listed in the appendix. Scryfall-comparator rows are joined
through their ``cascade_row``; only comparisons against a selected eval row
are reported.

Outputs (``<out>/<YYYY-MM-DD>/``, local date)::

    report.md                     all tables, figure links, provenance
    tables/*.csv                  one CSV per table (for slides / LaTeX)
    figures/*.svg                 dependency-free SVG, light + dark aware

Figures follow the project's chart conventions: the form is chosen by the
data's job (grouped bars to tell base from tuned apart; emphasis bars where
one configuration is the point; a single-hue ranking per query), two
categorical slots validated for CVD separation in both modes, thin bars with
a rounded data-end, values at the bar tip in text ink, a legend whenever two
series share a plot, and a ``<title>`` on every mark for hover.

Usage::

    python scripts/generate_report.py --since 2026-09-01 --out docs/reports/
    python scripts/generate_report.py --since 2026-10-03 --headline cascade_v2_concepts_kw
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, date, datetime
from html import escape
from pathlib import Path
from typing import Any

import psycopg

from src.config import settings
from src.logging_utils import PipelineRun

BASE_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Display order + labels for the grid. Unknown configs are appended alphabetically.
CONFIG_LABELS: dict[str, str] = {
    "raw_dense": "Raw query, no rewriter, no filter",
    "cascade_passthrough": "v1 filters + raw query",
    "cascade_hyde_v1_nosql": "v1 hypothetical text, no filter",
    "cascade_hyde_v1": "v1 hypothetical text + filters",
    "cascade_v2_concepts_nosql": "v2 concepts, no filter",
    "cascade_v2_concepts": "v2 concepts + filters",
    "cascade_v2_concepts_kw": "v2 concepts + filters + keywords",
    "cascade_v2_query_plus_concepts": "v2 query + concepts + filters + keywords",
    "cascade_v2_hyde": "v2 one-sentence hypothetical + filters",
    "oracle_tag_label": "Tag-label oracle + v1 filters",
    "oracle_tag_label_nosql": "Tag-label oracle, no filter",
}
CATEGORIES = ("natural", "jargon", "fragmented", "hybrid", "constrained", "mechanical")


# ---- data ----------------------------------------------------------------


def embedder_label(model: str | None) -> str:
    if not model:
        return "—"
    return "base" if model == BASE_MODEL else f"tuned ({Path(model).name})"


def load_rows(since: date) -> list[dict[str, Any]]:
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, created_at, eval_set_version, config, metrics, per_query, notes "
            "FROM experiment_runs WHERE created_at >= %s ORDER BY id",
            (since,),
        )
        cols = ("id", "created_at", "eval_set_version", "config", "metrics", "per_query", "notes")
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def select(rows: list[dict[str, Any]]):
    """Newest eval row per (config, embedder); comparator rows keyed by cascade row."""
    evals: dict[tuple[str, str], dict[str, Any]] = {}
    superseded: dict[tuple[str, str], list[int]] = defaultdict(list)
    finetunes, comparators = [], defaultdict(dict)
    for r in rows:
        kind = r["config"].get("kind", "eval")
        if kind == "finetune_embedder":
            finetunes.append(r)
        elif kind == "scryfall_comparator":
            comparators[int(r["config"]["cascade_row"])][r["config"]["variant"]] = r
        elif r["config"].get("config_name"):
            key = (r["config"]["config_name"], r["config"].get("embedding_model") or "")
            if key in evals:
                superseded[key].append(evals[key]["id"])
            evals[key] = r
    return evals, superseded, finetunes, comparators


def config_order(names: set[str]) -> list[str]:
    known = [c for c in CONFIG_LABELS if c in names]
    return known + sorted(names - set(known))


# ---- formatting ------------------------------------------------------------


def f3(v: Any) -> str:
    return "—" if v is None else f"{float(v):.3f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)


# ---- SVG charts ------------------------------------------------------------

# Light-mode values are written as presentation ATTRIBUTES so renderers without
# CSS-variable support (Word, some slide tools, pandoc's converters) still get the
# right colours; dark mode is a CSS override on the same classes.
_LIGHT = {
    "bg": 'fill="#fcfcfb"',
    "t": 'fill="#0b0b0b" font-size="15" font-weight="600"',
    "st": 'fill="#52514e" font-size="12"',
    "lab": 'fill="#52514e" font-size="12"',
    "val": 'fill="#0b0b0b" font-size="11.5"',
    "tick": 'fill="#898781" font-size="11"',
    "grid": 'stroke="#e1e0d9" stroke-width="1"',
    "axis": 'stroke="#c3c2b7" stroke-width="1"',
    "s1": 'fill="#2a78d6"',
    "s2": 'fill="#eb6834"',
    "ctx": 'fill="#b9b7ae"',
}
_FONT = "system-ui,-apple-system,Segoe UI,sans-serif"
_STYLE = """
  .val,.tick{font-variant-numeric:tabular-nums}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .t,.val{fill:#ffffff} .st,.lab{fill:#c3c2b7} .tick{fill:#898781}
    .grid{stroke:#2c2c2a} .axis{stroke:#383835}
    .s1{fill:#3987e5} .s2{fill:#d95926} .ctx{fill:#5a5955}}
"""


def _a(cls: str) -> str:
    """class + light-mode presentation attributes for one element."""
    return f'class="{cls}" {_LIGHT[cls]}'


def _bar(x0: float, y: float, w: float, h: float, cls: str, tip: str) -> str:
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    r = min(4.0, h / 2, max(w, 0))
    if w <= r:
        d = f"M{x0:.1f},{y:.1f}h{max(w, 1):.1f}v{h:.1f}h{-max(w, 1):.1f}z"
    else:
        d = (
            f"M{x0:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z"
        )
    return f'<path {_a(cls)} d="{d}"><title>{escape(tip)}</title></path>'


def hbar_chart(
    *,
    title: str,
    subtitle: str,
    groups: list[tuple[str, list[tuple[str, float | None, str]]]],
    series: list[tuple[str, str]] | None = None,
    x_max: float = 1.0,
    label_values: bool = True,
    bar_h: int = 14,
) -> str:
    """Horizontal (optionally grouped) bars.

    ``groups``: ``[(row label, [(css class, value, hover text), ...]), ...]``.
    ``series``: ``[(css class, legend label), ...]`` — legend drawn when >= 2.
    """
    label_w = min(330, max(120, int(max(len(g[0]) for g in groups) * 6.4) + 12))
    plot_w, right, gap = 430, 56, 2
    top = 62 + (22 if series and len(series) >= 2 else 0)
    band_pad = 10
    y = top
    body: list[str] = []
    for label, bars in groups:
        n = len(bars)
        band = n * bar_h + (n - 1) * gap
        body.append(
            f'<text {_a("lab")} x="{label_w - 8}" y="{y + band / 2 + 4:.1f}" text-anchor="end">'
            f"{escape(label)}</text>"
        )
        for i, (cls, value, tip) in enumerate(bars):
            by = y + i * (bar_h + gap)
            if value is None:
                body.append(
                    f'<text {_a("tick")} x="{label_w + 4}" y="{by + bar_h - 3}">not run</text>'
                )
                continue
            w = plot_w * min(max(value, 0), x_max) / x_max
            body.append(_bar(label_w, by, w, bar_h, cls, tip))
            if label_values:
                body.append(
                    f'<text {_a("val")} x="{label_w + w + 5:.1f}" y="{by + bar_h - 3}">'
                    f"{value:.2f}</text>"
                )
        y += band + band_pad
    bottom = y + 4
    width, height = label_w + plot_w + right, bottom + 26
    grid = []
    for k in range(5):
        gx = label_w + plot_w * k / 4
        cls = "axis" if k == 0 else "grid"
        grid.append(f'<line {_a(cls)} x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" y2="{bottom}"/>')
        grid.append(
            f'<text {_a("tick")} x="{gx:.1f}" y="{bottom + 16}" text-anchor="middle">'
            f"{x_max * k / 4:.2f}</text>"
        )
    legend = []
    if series and len(series) >= 2:
        lx = label_w
        for cls, name in series:
            legend.append(f'<rect {_a(cls)} x="{lx}" y="46" width="10" height="10" rx="2"/>')
            legend.append(f'<text {_a("lab")} x="{lx + 15}" y="55">{escape(name)}</text>')
            lx += 15 + int(len(name) * 6.6) + 18
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" font-family="{_FONT}" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">'
        f"<style>{_STYLE}</style>"
        f'<rect {_a("bg")} width="{width}" height="{height}"/>'
        f'<text {_a("t")} x="16" y="24">{escape(title)}</text>'
        f'<text {_a("st")} x="16" y="41">{escape(subtitle)}</text>'
        + "".join(legend)
        + "".join(grid)
        + "".join(body)
        + "</svg>\n"
    )


# ---- report ----------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=settings.repo_root, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def build(args: argparse.Namespace, run: PipelineRun) -> Path:
    rows = load_rows(args.since)
    evals, superseded, finetunes, comparators = select(rows)
    # Folder named by LOCAL date (a human-facing artifact); the header timestamp stays UTC.
    out = args.out / datetime.now().strftime("%Y-%m-%d")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    embedders = sorted({k[1] for k in evals}, key=lambda m: (m != BASE_MODEL, m))
    tuned = next((m for m in embedders if m != BASE_MODEL), None)
    configs = config_order({k[0] for k in evals})

    counter = iter(range(1, 100))

    def sec() -> int:
        return next(counter)

    def ev(cfg: str, model: str | None) -> dict[str, Any] | None:
        return evals.get((cfg, model or ""))

    head = ev(args.headline, tuned)
    ctrl = ev(args.control, BASE_MODEL)
    eval_versions = sorted({r["eval_set_version"] for r in evals.values()})

    md += [
        "# Results report",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} at commit `{_git_sha()}` "
        f"by `scripts/generate_report.py` from `experiment_runs` rows since {args.since} "
        f"({len(rows)} rows read, {len(evals)} eval cells selected). Eval set: "
        f"{', '.join(eval_versions)}. Newest row per (configuration, embedder) wins; "
        "superseded ids are in the appendix.",
        "",
        f"**Headline configuration:** `{args.headline}` on the tuned embedder"
        + (f" (row {head['id']})" if head else " — NOT FOUND")
        + f". **Control:** `{args.control}` on the base embedder"
        + (f" (row {ctrl['id']})." if ctrl else " — NOT FOUND."),
        "",
    ]

    # -- Table 1: the grid
    h1 = ["Configuration", "Embedder", "Row", "P@10", "MRR", "R-prec", "R-prec (reachable)",
          "Reachable", "Depth to 90% (x set)", "Stage 1 ms", "Stage 1 out tokens"]  # fmt: skip
    t1: list[list[Any]] = []
    for cfg in configs:
        for m in embedders:
            r = ev(cfg, m)
            if not r:
                continue
            x = r["metrics"]
            t1.append([
                CONFIG_LABELS.get(cfg, cfg), embedder_label(m), r["id"],
                f3(x.get("precision_at_10")), f3(x.get("mrr")), f3(x.get("set_r_precision")),
                f3(x.get("set_r_precision_reachable")), f3(x.get("set_reachable_frac")),
                "—" if x.get("set_depth90_x_target_median") is None else f"{x['set_depth90_x_target_median']:.2f}",
                "—" if x.get("stage1_ms_mean") is None else f"{x['stage1_ms_mean']:.0f}",
                "—" if x.get("stage1_completion_tokens_mean") is None else f"{x['stage1_completion_tokens_mean']:.0f}",
            ])  # fmt: skip
    write_csv(out / "tables" / "01_grid.csv", h1, t1)
    md += [
        f"## {sec()}. Configuration grid",
        "",
        "P@10 and MRR are against the hand-curated judgments (archetype precision). "
        "R-precision, reachable, and depth are set-retrieval measures against each query's tag pool "
        "inside the Stage 2 candidate set (21 tag-mapped queries): *R-prec* charges the filter's "
        "exclusions to the cascade, *R-prec (reachable)* scores Stage 3's ranking of what the filter "
        "admitted, *Reachable* is the share of the pool the filter admitted.",
        "",
        md_table(h1, t1),
        "",
    ]

    # -- Figure 1: rewriter x embedder
    fig_cfgs = [c for c in ("raw_dense", "cascade_passthrough", "cascade_hyde_v1",
                            "cascade_v2_concepts", "cascade_v2_concepts_kw") if c in configs]  # fmt: skip
    if tuned and fig_cfgs:
        groups = []
        for cfg in fig_cfgs:
            bars = []
            for cls, m in (("s1", BASE_MODEL), ("s2", tuned)):
                r = ev(cfg, m)
                v = r["metrics"].get("set_r_precision") if r else None
                tip = f"{CONFIG_LABELS.get(cfg, cfg)} — {embedder_label(m)}: " + (
                    f"{v:.3f} (row {r['id']})" if v is not None else "not run"
                )
                bars.append((cls, v, tip))
            groups.append((CONFIG_LABELS.get(cfg, cfg), bars))
        svg = hbar_chart(
            title="Set retrieval by rewriter configuration and embedder",
            subtitle="R-precision against tag pools (21 queries). Filters shrink the reachable pool, so unfiltered rows can score higher.",
            groups=groups,
            series=[("s1", "base embedder"), ("s2", "tuned embedder")],
        )
        (out / "figures" / "fig1_rewriter_x_embedder.svg").write_text(svg, encoding="utf-8")
        md += ["![Rewriter x embedder](figures/fig1_rewriter_x_embedder.svg)", ""]

    # -- Table 2: per-category
    if head and ctrl:
        h2 = ["Category", "n", "Control P@10", "Headline P@10", "Control R-prec", "Headline R-prec"]
        t2 = []
        for cat in CATEGORIES:
            cells = []
            n = 0
            for r in (ctrl, head):
                qs = [q for q in r["per_query"] if q.get("category") == cat]
                n = len(qs)
                p = sum(q.get("precision_at_10", 0) for q in qs) / n if n else None
                sm = [q["set_metrics"]["r_precision"] for q in qs if q.get("set_metrics")]
                cells.append((p, sum(sm) / len(sm) if sm else None))
            t2.append([cat, n, f3(cells[0][0]), f3(cells[1][0]), f3(cells[0][1]), f3(cells[1][1])])
        write_csv(out / "tables" / "02_by_category.csv", h2, t2)
        md += [f"## {sec()}. By query category (control vs headline)", "", md_table(h2, t2), ""]

    # -- Table 3: fine-tuning runs
    if finetunes:
        h3 = ["Run", "Row", "Pairs", "Steps", "Train min", "Held-out ndcg@10 base → tuned",
              "Held-out recall@100 base → tuned", "NanoBEIR mean ndcg@10 base → tuned"]  # fmt: skip
        t3 = []
        for r in finetunes:
            x, c = r["metrics"], r["config"]

            def g(pre: str, suf: str, x: dict[str, Any] = x) -> Any:
                return next(
                    (v for k, v in x.items() if k.startswith(pre) and k.endswith(suf)), None
                )

            nb_base = None
            nb_file = Path(c.get("output_dir", "")) / "nanobeir_base.json"
            if nb_file.exists():
                nb_base = json.loads(nb_file.read_text()).get("NanoBEIR_mean_cosine_ndcg@10")
            nb_tuned = g("nanobeir_tuned_", "NanoBEIR_mean_cosine_ndcg@10")
            t3.append([
                c.get("run_name"), r["id"], f"{c.get('pairs_used', 0):,}", x.get("steps"),
                f"{x.get('train_seconds', 0) / 60:.0f}",
                f"{f3(g('holdout_base_', 'cosine_ndcg@10'))} → {f3(g('holdout_tuned_', 'cosine_ndcg@10'))}",
                f"{f3(g('holdout_base_', 'cosine_recall@100'))} → {f3(g('holdout_tuned_', 'cosine_recall@100'))}",
                f"{f3(nb_base)} → {f3(nb_tuned)}",
            ])  # fmt: skip
        write_csv(out / "tables" / "03_finetune.csv", h3, t3)
        md += [
            f"## {sec()}. Embedder fine-tuning",
            "",
            "Held-out = tags withheld from training, queried by label against the full corpus. "
            "NanoBEIR is the general-retrieval forgetting probe.",
            "",
            md_table(h3, t3),
            "",
        ]

    # -- Table 4 + Figures 2/3: Scryfall comparator
    comp_rows = []
    for (cfg, m), r in evals.items():
        for variant, c in comparators.get(r["id"], {}).items():
            comp_rows.append((cfg, m, variant, r, c))
    if comp_rows:
        h4 = ["Configuration", "Embedder", "Cascade row", "Expert variant", "Parity R-prec",
              "Jaccard @|S|", "Depth to 90% (x set)", "n"]  # fmt: skip
        t4 = []
        order = {c: i for i, c in enumerate(configs)}
        comp_rows.sort(key=lambda t: (order.get(t[0], 99), t[1] != BASE_MODEL, t[2]))
        for cfg, m, variant, r, c in comp_rows:
            x = c["metrics"]
            t4.append([
                CONFIG_LABELS.get(cfg, cfg), embedder_label(m), r["id"],
                "tags allowed" if variant == "expert" else "no tags",
                f3(x.get("parity_r_precision")), f3(x.get("parity_jaccard_at_n_mean")),
                f"{x.get('parity_depth90_x_set_median', 0):.2f}", x.get("n_queries"),
            ])  # fmt: skip
        write_csv(out / "tables" / "04_scryfall_parity.csv", h4, t4)
        md += [
            f"## {sec()}. Parity with expert Scryfall queries",
            "",
            "For each eval query an expert Scryfall search was run and its result set restricted to "
            "the corpus. *Parity R-prec* is the fraction of the cascade's first |S| results that are "
            "in the expert's set S; depth below 1.0x means 90% of S is seen before scrolling past |S| cards.",
            "",
            md_table(h4, t4),
            "",
        ]
        expert = [t for t in comp_rows if t[2] == "expert"]
        if expert:
            groups = []
            for cfg, m, _v, r, c in expert:
                is_head = head is not None and r["id"] == head["id"]
                v = c["metrics"].get("parity_r_precision")
                label = f"{CONFIG_LABELS.get(cfg, cfg)} · {embedder_label(m).split(' ')[0]}"
                groups.append(
                    (label, [("s1" if is_head else "ctx", v, f"{label}: {v:.3f} (row {r['id']})")])
                )
            groups.sort(key=lambda g: -(g[1][0][1] or 0))
            svg = hbar_chart(
                title="Result parity with an expert's Scryfall query",
                subtitle="R-precision of the cascade ranking against the expert result set (tags allowed); headline configuration highlighted",
                groups=groups,
                bar_h=16,
            )
            (out / "figures" / "fig2_scryfall_parity.svg").write_text(svg, encoding="utf-8")
            md += ["![Scryfall parity](figures/fig2_scryfall_parity.svg)", ""]
        hc = comparators.get(head["id"], {}).get("expert") if head else None
        if hc:
            pq = [q for q in hc["per_query"] if q.get("parity")]
            pq.sort(key=lambda q: -q["parity"]["r_precision"])
            groups = [
                (q["query"][:44] + ("…" if len(q["query"]) > 44 else ""),
                 [("s1", q["parity"]["r_precision"],
                   f"{q['query']}: parity {q['parity']['r_precision']:.2f}; expert query {q['scryfall_query']} "
                   f"({q['scryfall_set_size']} cards)")])
                for q in pq
            ]  # fmt: skip
            svg = hbar_chart(
                title="Parity by query — headline configuration",
                subtitle="R-precision against each expert Scryfall result set; hover a bar for the expert query",
                groups=groups,
                label_values=False,
                bar_h=12,
            )
            (out / "figures" / "fig3_parity_per_query.svg").write_text(svg, encoding="utf-8")
            h5 = [
                "Query",
                "Expert Scryfall query",
                "Set size",
                "Parity R-prec",
                "Jaccard",
                "Operators",
                "Needs otag",
            ]
            t5 = [[q["query"], f"`{q['scryfall_query']}`", q["scryfall_set_size"],
                   f3(q["parity"]["r_precision"]), f3(q["parity"]["jaccard_at_n"]),
                   q["complexity"]["operators"], "yes" if q["complexity"]["uses_otag"] else ""] for q in pq]  # fmt: skip
            write_csv(out / "tables" / "05_parity_per_query.csv", h5, t5)
            md += [
                "![Parity per query](figures/fig3_parity_per_query.svg)",
                "",
                md_table(h5, t5),
                "",
            ]

            # -- Table 6: complexity
            h6 = ["", "Plain language (this work)", "Scryfall, tags allowed", "Scryfall, no tags"]
            nt = comparators.get(head["id"], {}).get("no_tag")
            e, n = hc["metrics"], (nt["metrics"] if nt else {})

            def g(d: dict[str, Any], k: str, fmt: str) -> str:
                return "—" if d.get(k) is None else fmt.format(d[k])

            t6 = [
                ["Length", f"{e.get('plain_words_mean', 0):.1f} words",
                 g(e, "complexity_chars_mean", "{:.0f} chars"), g(n, "complexity_chars_mean", "{:.0f} chars")],
                ["Operators per query", "0", g(e, "complexity_operators_mean", "{:.1f}"),
                 g(n, "complexity_operators_mean", "{:.1f}")],
                ["Queries needing `otag:`", "—", g(e, "complexity_uses_otag_frac", "{:.0%}"),
                 g(n, "complexity_uses_otag_frac", "{:.0%}")],
                ["Expert set precision vs judgments", "—", g(e, "scryfall_precision_vs_judgments_mean", "{:.2f}"),
                 g(n, "scryfall_precision_vs_judgments_mean", "{:.2f}")],
                ["Expert set recall vs judgments", "—", g(e, "scryfall_recall_vs_judgments_mean", "{:.2f}"),
                 g(n, "scryfall_recall_vs_judgments_mean", "{:.2f}")],
            ]  # fmt: skip
            write_csv(out / "tables" / "06_query_complexity.csv", h6, t6)
            md += [f"## {sec()}. What the user had to type", "", md_table(h6, t6), ""]

    # -- Per query: control vs headline
    per_query_md = ""
    if head and ctrl:
        cpar = {
            q["id"]: q
            for q in (comparators.get(ctrl["id"], {}).get("expert") or {"per_query": []})[
                "per_query"
            ]
        }
        hpar = {
            q["id"]: q
            for q in (comparators.get(head["id"], {}).get("expert") or {"per_query": []})[
                "per_query"
            ]
        }
        cq = {q["id"]: q for q in ctrl["per_query"]}

        def par(d: dict[str, Any], qid: str) -> str:
            return f3(d[qid]["parity"]["r_precision"]) if d.get(qid, {}).get("parity") else "—"

        def setp(q: dict[str, Any] | None) -> str:
            return f3(q["set_metrics"]["r_precision"]) if q and q.get("set_metrics") else "—"

        h7 = ["ID", "Query", "Category", "Control P@10", "Headline P@10", "Control R-prec",
              "Headline R-prec", "Control parity", "Headline parity"]  # fmt: skip
        t7 = [[q["id"], q["query"], q.get("category"), f3(cq.get(q["id"], {}).get("precision_at_10")),
               f3(q.get("precision_at_10")), setp(cq.get(q["id"])), setp(q), par(cpar, q["id"]), par(hpar, q["id"])]
              for q in sorted(head["per_query"], key=lambda q: q["id"])]  # fmt: skip
        write_csv(out / "tables" / "07_per_query.csv", h7, t7)
        per_query_md = md_table(h7, t7)
        md += [
            f"## {sec()}. Per query: control vs headline",
            "",
            "P@10 against the hand-curated judgments; R-prec against the query's tag pool (— where no tag is "
            "mapped); parity against the expert Scryfall result set (tags allowed).",
            "",
            per_query_md,
            "",
        ]

    # -- Appendix
    ha = [
        "Configuration",
        "Embedder",
        "Selected row",
        "Date",
        "Prompt",
        "Prompt sha",
        "Superseded rows",
    ]
    ta = []
    for cfg in configs:
        for m in embedders:
            r = ev(cfg, m)
            if r:
                c = r["config"]
                ta.append([cfg, embedder_label(m), r["id"], r["created_at"].date(), c.get("hyde_prompt") or "—",
                           c.get("hyde_prompt_sha") or "—", ", ".join(map(str, superseded.get((cfg, m), []))) or "—"])  # fmt: skip
    write_csv(out / "tables" / "A_provenance.csv", ha, ta)
    md += ["## Appendix — provenance", "", md_table(ha, ta), ""]

    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    if args.paper:
        update_paper_appendices(out, per_query_md, head, ctrl)
    run.processed(len(evals))
    run.note(out=str(out), rows_read=len(rows), eval_cells=len(evals), finetunes=len(finetunes),
             comparator_rows=sum(len(v) for v in comparators.values()))  # fmt: skip
    return out


def update_paper_appendices(
    out: Path, per_query_md: str, head: dict | None, ctrl: dict | None
) -> None:
    """Regenerate Appendices A-C of the paper draft from files and logged rows.

    Appendix A embeds the current rewriter prompt, B summarises the evaluation
    set (queries, judgment counts, mapped tag, expert Scryfall query), C is the
    per-query results table. Appendix D and everything else are left alone.
    """
    import yaml

    paper = settings.repo_root / "docs" / "thesis" / "paper-draft.md"
    text = paper.read_text(encoding="utf-8")
    a, d = text.index("### Appendix A"), text.index("### Appendix D")
    prompt = yaml.safe_load((settings.prompts_dir / "hyde_v2.yaml").read_text(encoding="utf-8"))
    ev = yaml.safe_load((settings.eval_dir / "queries_v1_draft.yaml").read_text(encoding="utf-8"))
    oracle = yaml.safe_load(
        (settings.eval_dir / "tag_label_oracle_v1.yaml").read_text(encoding="utf-8")
    )["queries"]
    expert = yaml.safe_load(
        (settings.eval_dir / "scryfall_expert_queries_v1.yaml").read_text(encoding="utf-8")
    )["queries"]
    hb = [
        "ID",
        "Query",
        "Category",
        "Relevant",
        "Borderline",
        "Mapped tag",
        "Expert Scryfall query",
        "Without tags",
    ]
    tb = [[q["id"], q["query"], q.get("category"), len(q.get("relevant") or []), len(q.get("borderline") or []),
           (oracle.get(q["id"], {}).get("tags") or ["—"])[0], f"`{expert[q['id']]['expert']}`",
           f"`{expert[q['id']]['no_tag']}`"] for q in sorted(ev["queries"], key=lambda q: q["id"])]  # fmt: skip
    examples = "\n\n".join(
        f"Query: {e['query']}\nOutput: {e['expected_output'].strip()}" for e in prompt["examples"]
    )
    gen = f"*Generated by `scripts/generate_report.py --paper` on {datetime.now().strftime('%Y-%m-%d')}; do not edit by hand.*"
    rows = (
        f"rows {ctrl['id']} (control) and {head['id']} (headline)"
        if head and ctrl
        else "no rows selected"
    )
    block = "\n".join(
        [
            "### Appendix A — Query-rewriter prompt (v2)",
            "",
            gen + " The original v1 prompt is `prompts/hyde_v1.yaml` in the repository.",
            "",
            "**System prompt**",
            "",
            "````text",
            prompt["system"].rstrip(),
            "````",
            "",
            f"**Few-shot examples ({len(prompt['examples'])})**",
            "",
            "````text",
            examples,
            "````",
            "",
            "### Appendix B — Evaluation set",
            "",
            gen
            + f" {len(tb)} queries (`data/eval/queries_v1_draft.yaml`, version {ev.get('version')}); the full relevance "
            "judgments (card ids with a one-line reason each) are in that file. *Mapped tag* is the Scryfall oracle tag used as "
            "the set-retrieval target; the expert queries are the Scryfall comparator inputs.",
            "",
            md_table(hb, tb),
            "",
            "### Appendix C — Per-query results",
            "",
            gen
            + f" Source: `{out.relative_to(settings.repo_root)}/tables/07_per_query.csv`, `experiment_runs` {rows}. "
            "P@10 is against the hand-curated judgments, R-prec against the query's tag pool, parity against the expert "
            "Scryfall result set (tags allowed).",
            "",
            per_query_md or "*No headline/control rows selected.*",
            "",
            "",
        ]
    )
    paper.write_text(text[:a] + block + text[d:], encoding="utf-8")
    write_csv(out / "tables" / "B_eval_set.csv", hb, tb)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--since", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--out", type=Path, default=settings.repo_root / "docs" / "reports")
    parser.add_argument(
        "--headline", default="cascade_v2_concepts_kw", help="Headline config (tuned embedder)."
    )
    parser.add_argument(
        "--control", default="cascade_hyde_v1", help="Control config (base embedder)."
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Also regenerate Appendices A-C of docs/thesis/paper-draft.md.",
    )
    args = parser.parse_args()
    with PipelineRun("generate_report", inputs={"since": str(args.since), "headline": args.headline,
                                                "control": args.control}) as run:  # fmt: skip
        out = build(args, run)
    print(f"  Wrote {out / 'report.md'}")
    for p in sorted((out / "figures").glob("*.svg")):
        print(f"        {p.relative_to(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
