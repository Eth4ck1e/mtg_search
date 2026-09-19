"""Set-retrieval probe: if results are 'every match in descending order', how good
is the ordering against a tag's full pool? Reports P@100, R-precision (precision at
depth = pool size), R@500, and the depth needed to collect 90% of the pool.
No filters; in-memory cosine; dedupe by oracle_id. One-off analysis (journal §5f)."""

import json
from pathlib import Path

import numpy as np
import psycopg
import yaml
from sentence_transformers import SentenceTransformer

import src.utils.quiet  # noqa: F401
from src.config import settings
from src.preprocess_text import (
    build_embedding_text,
    format_for_nomic_document,
    format_for_nomic_query,
    load_keyword_dict,
)
from src.utils.device import select_device

TUNED = "models/nomic-mtg-v1"
spec = yaml.safe_load(Path("data/eval/tag_label_oracle_v1.yaml").read_text())["queries"]
with psycopg.connect(settings.database_url) as conn:
    cur = conn.cursor()
    cur.execute(
        "SELECT oracle_id::text, oracle_text, keywords FROM cards WHERE oracle_text <> '' ORDER BY oracle_id, face_index"
    )
    rows = cur.fetchall()
    cur.execute("SELECT per_query FROM experiment_runs WHERE id=11")
    ctrl = {q["id"]: q for q in cur.fetchone()[0]}
    pools, labels = {}, {}
    for qid, e in spec.items():
        if not e.get("tags"):
            continue
        slug = e["tags"][0]
        cur.execute(
            """SELECT t.label, array_agg(DISTINCT ct.oracle_id::text) FROM oracle_tags t
          JOIN oracle_tag_closure cl ON cl.ancestor_id=t.id JOIN card_tags ct ON ct.tag_id=cl.descendant_id
          JOIN (SELECT DISTINCT oracle_id FROM cards) c ON c.oracle_id=ct.oracle_id WHERE t.slug=%s GROUP BY t.label""",
            (slug,),
        )
        lab, ids = cur.fetchone()
        pools[qid] = set(ids)
        labels[qid] = (slug, lab)
oids = np.array([r[0] for r in rows])
kd = load_keyword_dict()
texts = [format_for_nomic_document(build_embedding_text(r[1] or "", r[2] or [], kd)) for r in rows]
dev = str(select_device())
uniq, inv = np.unique(oids, return_inverse=True)
models, docs = {}, {}
for name, path in [("tuned", TUNED), ("base", "nomic-ai/nomic-embed-text-v1.5")]:
    models[name] = SentenceTransformer(path, device=dev, trust_remote_code=True)
    docs[name] = (
        models[name]
        .encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        .astype(np.float32)
    )


def ranked(m, text):
    q = models[m].encode([format_for_nomic_query(text)], normalize_embeddings=True)[0]
    best = np.full(len(uniq), -2.0, dtype=np.float32)
    np.maximum.at(best, inv, docs[m] @ q)
    return uniq[np.argsort(-best)]


def stats(order, pool):
    pool = pool & set(uniq)
    hit = np.isin(order, list(pool))
    n = len(pool)
    pos = np.flatnonzero(hit)
    return dict(
        n=n,
        rprec=float(hit[:n].mean()),
        r500=float(hit[:500].sum() / n),
        p100=float(hit[:100].mean()),
        depth90=int(pos[int(np.ceil(0.9 * len(pos))) - 1]) + 1,
    )


def hyp(q):
    return (ctrl[q].get("stage1") or {}).get("hypothetical_card") or ctrl[q]["query"]


systems = {
    "label/tuned": ("tuned", lambda q: labels[q][1]),
    "label/base": ("base", lambda q: labels[q][1]),
    "hypoth/base": ("base", hyp),
    "hypoth/tuned": ("tuned", hyp),
    "rawquery/tuned": ("tuned", lambda q: ctrl[q]["query"]),
    "rawquery/base": ("base", lambda q: ctrl[q]["query"]),
}
per = {(q, s): stats(ranked(m, fn(q)), pools[q]) for q in pools for s, (m, fn) in systems.items()}
print(
    f"{'system':16s} {'P@100':>6s} {'R-prec':>7s} {'R@500':>6s} {'median depth to 90% (x pool size)':>36s}"
)
for s in systems:
    L = [per[(q, s)] for q in pools]
    print(
        f"{s:16s} {np.mean([x['p100'] for x in L]):6.2f} {np.mean([x['rprec'] for x in L]):7.2f} {np.mean([x['r500'] for x in L]):6.2f} {np.median([x['depth90'] / x['n'] for x in L]):36.1f}"
    )
print(
    "\nper tag   R-precision: label/tuned | label/base | hypoth/base     depth-to-90%: label/tuned vs hypoth/base"
)
for q in pools:
    a, b, c = (per[(q, s)] for s in ["label/tuned", "label/base", "hypoth/base"])
    print(
        f"  {labels[q][0]:20s} pool={a['n']:5d}   {a['rprec']:.2f} | {b['rprec']:.2f} | {c['rprec']:.2f}        {a['depth90']:6,d}  vs {c['depth90']:6,d}"
    )
Path(f"{TUNED}/set_retrieval_probe.json").write_text(
    json.dumps({f"{q}|{s}": v for (q, s), v in per.items()}, indent=1)
)
