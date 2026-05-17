"""Download the Scryfall ``oracle_cards`` bulk JSON.

Hits Scryfall's bulk-data index, locates the ``oracle_cards`` entry, and
stream-downloads it to ``data/raw/oracle-cards-<UTC-date>.json``. The
file is staged through a ``.partial`` tempfile and atomically renamed on
success, so a half-finished download cannot masquerade as a complete
file. SHA-256 is computed while streaming and recorded in the run log.

Idempotent within a day — if today's file already exists, the script
no-ops unless ``--force`` is passed. Re-running on a later date always
produces a fresh dated file.

Usage::

    python scripts/download_scryfall.py
    python scripts/download_scryfall.py --force
    python scripts/download_scryfall.py --out-dir /tmp
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.config import settings
from src.logging_utils import PipelineRun

BULK_INDEX_URL = "https://api.scryfall.com/bulk-data"
USER_AGENT = "mtg_search/0.2.0 (+https://github.com/Eth4ck1e/mtg_search)"
ACCEPT = "application/json"
CHUNK_SIZE = 1024 * 1024  # 1 MiB
HTTP_TIMEOUT_S = 30


def _http_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": ACCEPT}


def _find_oracle_cards(entries: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in entries:
        if entry.get("type") == "oracle_cards":
            return entry
    available = sorted({str(e.get("type")) for e in entries})
    raise RuntimeError(
        f"No 'oracle_cards' entry in Scryfall bulk-data index. Available: {available}"
    )


def _stream_download(url: str, dest: Path, expected_size: int) -> tuple[int, str]:
    """Stream ``url`` to ``dest`` via a ``.partial`` tempfile.

    Returns ``(bytes_written, sha256_hex)``. The tempfile is atomically
    renamed to ``dest`` only after the full transfer succeeds.
    """
    tmp = dest.with_suffix(dest.suffix + ".partial")
    sha = hashlib.sha256()
    bytes_written = 0

    with requests.get(url, stream=True, headers=_http_headers(), timeout=HTTP_TIMEOUT_S) as resp:
        resp.raise_for_status()
        with (
            tmp.open("wb") as fh,
            tqdm(total=expected_size, unit="B", unit_scale=True, desc=dest.name) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                sha.update(chunk)
                bytes_written += len(chunk)
                pbar.update(len(chunk))

    tmp.replace(dest)
    return bytes_written, sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=settings.raw_data_dir,
        help="Directory where the bulk file lands (default: data/raw/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if today's file already exists.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = args.out_dir / f"oracle-cards-{today}.json"

    with PipelineRun(
        "download_scryfall",
        inputs={"index_url": BULK_INDEX_URL, "out_path": str(out_path)},
    ) as run:
        resp = requests.get(BULK_INDEX_URL, headers=_http_headers(), timeout=HTTP_TIMEOUT_S)
        resp.raise_for_status()
        entries = resp.json()["data"]
        run.event("bulk_index_fetched", entries=len(entries))

        entry = _find_oracle_cards(entries)
        run.note(
            scryfall_updated_at=entry["updated_at"],
            size_bytes_expected=entry["size"],
            download_uri=entry["download_uri"],
        )

        if out_path.exists() and not args.force:
            existing_size = out_path.stat().st_size
            run.note(action="skipped_existing", existing_size=existing_size)
            print(
                f"Already have {out_path.name} ({existing_size:,} bytes). "
                "Use --force to re-download."
            )
            return 0

        bytes_written, sha256 = _stream_download(entry["download_uri"], out_path, entry["size"])
        run.note(
            size_bytes_actual=bytes_written,
            sha256=sha256,
            output_path=str(out_path),
        )
        run.processed()
        print(
            f"Wrote {out_path}\n"
            f"  size:   {bytes_written:,} bytes\n"
            f"  sha256: {sha256}\n"
            f"  scryfall updated_at: {entry['updated_at']}"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
