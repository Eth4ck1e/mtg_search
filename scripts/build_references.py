"""Build the paper's reference list and ``references.bib`` from source metadata.

Inputs:

* ``docs/sources/arxiv_metadata.json`` — title / authors / year for every arXiv
  source, as returned by the arXiv API (``--refresh`` re-fetches it in one
  request for the ids found in ``docs/sources/README.md`` plus ``--extra-id``).
* ``docs/sources/extra_references.yaml`` — non-arXiv works, published venues
  (each copied from the arXiv entry's own comment field), in-text aliases, and
  citations that could not be resolved to a publication.

Outputs:

* ``docs/thesis/references.bib`` — every source held.
* The ``## References`` section of ``docs/thesis/paper-draft.md`` — only the
  works the paper text actually cites, detected by "<first-author surname> …
  <year>" (or an alias) appearing in the body. Re-run after editing the paper;
  the section is regenerated, never hand-edited.

Author names and titles come from the metadata, never from memory — the first
run of this script corrected three citations that had been written by hand.

Usage::

    python scripts/build_references.py
    python scripts/build_references.py --refresh --extra-id 2106.09685
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from typing import Any

import requests
import yaml

from src.config import settings

SOURCES = settings.repo_root / "docs" / "sources"
META = SOURCES / "arxiv_metadata.json"
EXTRA = SOURCES / "extra_references.yaml"
PAPER = settings.repo_root / "docs" / "thesis" / "paper-draft.md"
BIB = settings.repo_root / "docs" / "thesis" / "references.bib"
ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def fetch_arxiv(ids: list[str]) -> list[dict[str, Any]]:
    """One arXiv API request for all ids (their guidance: batch, don't loop)."""
    time.sleep(3)
    resp = requests.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": ",".join(ids), "max_results": len(ids) + 5},
        headers={"User-Agent": "mtg_search-research/0.2 (academic use)"},
        timeout=60,
    )
    resp.raise_for_status()
    out = []
    for e in ET.fromstring(resp.text).findall("a:entry", ATOM):

        def opt(tag: str, e: ET.Element = e) -> str | None:
            node = e.find(tag, ATOM)
            return node.text if node is not None else None

        out.append(
            {
                "id": re.sub(r"v\d+$", "", e.find("a:id", ATOM).text.rsplit("/", 1)[-1]),
                "title": " ".join(e.find("a:title", ATOM).text.split()),
                "authors": [a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)],
                "year": e.find("a:published", ATOM).text[:4],
                "journal_ref": opt("x:journal_ref"),
                "comment": " ".join(opt("x:comment").split()) if opt("x:comment") else None,
            }
        )
    return out


def surname(full: str) -> str:
    return full.split()[-1]


def ascii_key(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", norm.lower())


def initials(full: str) -> str:
    parts = full.split()
    return f"{parts[-1]}, " + " ".join(f"{p[0]}." for p in parts[:-1])


def author_list(authors: list[str]) -> str:
    names = [initials(a) for a in authors]
    if len(names) > 5:
        return ", ".join(names[:3]) + ", et al."
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def load_entries() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extra = yaml.safe_load(EXTRA.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    for m in json.loads(META.read_text(encoding="utf-8")):
        first = m["title"].split(":")[0].split()[0]
        entries.append(
            {
                "key": f"{ascii_key(surname(m['authors'][0]))}{m['year']}{ascii_key(first)}",
                "authors": m["authors"],
                "year": int(m["year"]),
                "title": m["title"],
                "venue": extra["venues"].get(m["id"]),
                "arxiv": m["id"],
                "aliases": extra.get("aliases", {}).get(m["id"], []),
                "bibtype": "inproceedings" if extra["venues"].get(m["id"]) else "misc",
            }
        )
    for m in extra["manual"]:
        entries.append({"arxiv": None, "aliases": [], **m})
    return entries, extra


def is_cited(entry: dict[str, Any], body: str) -> bool:
    names = [surname(entry["authors"][0]), *entry["aliases"]]
    year = str(entry["year"])
    # "Karpukhin et al., 2020", "Reimers and Gurevych (2019)", "CustomIR (Paull, 2025)"
    return any(re.search(rf"\b{re.escape(n)}\b[^\n]{{0,45}}?\b{year}\b", body) for n in names)


def reference_line(e: dict[str, Any]) -> str:
    where = e["venue"] or "arXiv preprint"
    tail = f" arXiv:{e['arxiv']}." if e["arxiv"] else (f" doi:{e['doi']}." if e.get("doi") else "")
    return f"{author_list(e['authors'])} ({e['year']}). {e['title'].rstrip('.')}. *{where}*.{tail}"


def bibtex(e: dict[str, Any]) -> str:
    fields = {
        "author": " and ".join(e["authors"]),
        "title": "{" + e["title"] + "}",
        "year": str(e["year"]),
    }
    if e["venue"]:
        fields[
            "booktitle"
            if e["bibtype"] == "inproceedings"
            else "howpublished"
            if e["bibtype"] == "misc"
            else "note"
        ] = e["venue"]
    if e["arxiv"]:
        fields |= {
            "eprint": e["arxiv"],
            "archivePrefix": "arXiv",
            "url": f"https://arxiv.org/abs/{e['arxiv']}",
        }
    if e.get("doi"):
        fields["doi"] = e["doi"]
    body = ",\n".join(f"  {k:13s} = {{{v}}}" for k, v in fields.items())
    return f"@{e['bibtype']}{{{e['key']},\n{body}\n}}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="Re-fetch arXiv metadata.")
    parser.add_argument(
        "--extra-id", action="append", default=[], help="arXiv id not in the sources README."
    )
    parser.add_argument("--check", action="store_true", help="Report only; write nothing.")
    args = parser.parse_args()

    if args.refresh:
        readme = (SOURCES / "README.md").read_text(encoding="utf-8")
        have = {m["id"] for m in json.loads(META.read_text())} if META.exists() else set()
        ids = sorted(
            set(re.findall(r"arxiv\.org/abs/([0-9.]+)", readme)) | have | set(args.extra_id)
        )
        META.write_text(
            json.dumps(fetch_arxiv(ids), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"  Refreshed {len(ids)} arXiv entries → {META.name}")

    entries, extra = load_entries()
    paper = PAPER.read_text(encoding="utf-8")
    start = paper.index("\n## References\n")
    end = paper.index("\n---\n\n## Appendices", start)
    body = paper[:start]

    cited = sorted(
        (e for e in entries if is_cited(e, body)),
        key=lambda e: (ascii_key(surname(e["authors"][0])), e["year"]),
    )
    uncited = [e for e in entries if e not in cited]
    lines = [
        "",
        "## References",
        "",
        "*Generated by `scripts/build_references.py` from `docs/sources/arxiv_metadata.json` and "
        "`extra_references.yaml`; lists the works cited in the text above. Do not edit by hand — "
        "re-run the script after changing citations. BibTeX for every source held: `docs/thesis/references.bib`.*",
        "",
    ]
    lines += [f"{reference_line(e)}\n" for e in cited]
    if extra.get("unresolved"):
        lines += [
            "**Cited but unresolved** (no publication identified — resolve or remove before submission):",
            "",
        ]
        lines += [
            f"- {u['cite']} — {u['where']}. {' '.join(u['note'].split())}"
            for u in extra["unresolved"]
        ]
        lines.append("")
    section = "\n".join(lines)

    print(
        f"  Sources held: {len(entries)}   cited in paper: {len(cited)}   not cited: {len(uncited)}"
    )
    for u in extra.get("unresolved", []):
        print(f"  UNRESOLVED: {u['cite']} ({u['where']})")
    if args.check:
        for e in uncited:
            print(f"    not cited: {surname(e['authors'][0])} {e['year']} — {e['title'][:60]}")
        return 0
    PAPER.write_text(paper[:start] + section + paper[end:], encoding="utf-8")
    BIB.write_text(
        "\n\n".join(bibtex(e) for e in sorted(entries, key=lambda e: e["key"])) + "\n",
        encoding="utf-8",
    )
    print(
        f"  Wrote {BIB.relative_to(settings.repo_root)} and the References section of {PAPER.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
