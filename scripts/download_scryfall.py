"""Download a Scryfall bulk-data file (``oracle_cards`` or ``oracle_tags``).

Hits Scryfall's typed bulk-data endpoint for the requested dataset
(returns metadata for that dataset directly, no filtering required), then
stream-downloads the referenced ``.jsonl.gz`` bulk file to
``data/raw/<dataset>-<UTC-date>.jsonl.gz``. The file is staged through a
``.partial`` tempfile and atomically renamed on success, so a half-finished
download cannot masquerade as a complete file. SHA-256 is computed while
streaming and recorded in the run log.

Scryfall API contract (verified 2026-09-11 for oracle-cards, 2026-09-18 for
oracle-tags):
    Endpoint returns a single bulk_data object (not wrapped in ``data``):
        {
          "object": "bulk_data",
          "type": "oracle_cards" | "oracle_tags",
          "updated_at": "...",
          "jsonl_download_uri": "https://data.scryfall.io/...jsonl.gz",
          "compressed_size": <bytes>,
          ...
        }
    The download is gzipped JSON-Lines; downstream ingestion must decompress
    and iterate line by line. ``data.scryfall.io`` file origins are exempt
    from Scryfall's API rate limits (https://scryfall.com/docs/api/rate-limits);
    the single metadata call to ``api.scryfall.com`` is the only rate-limited
    request this script makes.

Idempotent within a day — if today's file already exists, the script
no-ops unless ``--force`` is passed. Re-running on a later date always
produces a fresh dated file.

Usage::

    python scripts/download_scryfall.py                        # oracle-cards
    python scripts/download_scryfall.py --dataset oracle-tags  # Tagger oracle tags
    python scripts/download_scryfall.py --force
    python scripts/download_scryfall.py --out-dir /tmp
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import settings
from src.logging_utils import PipelineRun

CHUNK_SIZE = 1024 * 1024  # 1 MiB
HTTP_TIMEOUT_S = 30

# CLI dataset name -> (bulk-data endpoint, expected Scryfall ``type`` field).
# The CLI name doubles as the output filename prefix.
DATASETS: dict[str, tuple[str, str]] = {
    "oracle-cards": (settings.scryfall_bulk_endpoint, "oracle_cards"),
    "oracle-tags": (settings.scryfall_oracle_tags_endpoint, "oracle_tags"),
}


def _http_headers() -> dict[str, str]:
    return {
        "User-Agent": settings.scryfall_user_agent,
        "Accept": "application/json",
    }


def _stream_download(url: str, dest: Path, expected_size: int) -> tuple[int, str]:
    """Stream ``url`` to ``dest`` via a ``.partial`` tempfile.

    Returns ``(bytes_written, sha256_hex)``. The tempfile is atomically
    renamed to ``dest`` only after the full transfer succeeds. Progress is
    reported via a tqdm bar sized to ``expected_size`` (Scryfall's
    ``compressed_size`` field).
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
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="oracle-cards",
        help="Which Scryfall bulk dataset to fetch (default: oracle-cards).",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = args.out_dir / f"{args.dataset}-{today}.jsonl.gz"
    endpoint, expected_type = DATASETS[args.dataset]

    with PipelineRun(
        "download_scryfall",
        inputs={"dataset": args.dataset, "endpoint": endpoint, "out_path": str(out_path)},
    ) as run:
        resp = requests.get(endpoint, headers=_http_headers(), timeout=HTTP_TIMEOUT_S)
        resp.raise_for_status()
        meta = resp.json()

        # Field names verified 2026-09-11 against the live API.
        try:
            download_uri = meta["jsonl_download_uri"]
            expected_size = int(meta["compressed_size"])
            updated_at = meta["updated_at"]
            bulk_type = meta.get("type", "unknown")
        except KeyError as exc:
            raise RuntimeError(
                f"Unexpected Scryfall bulk-data response shape (missing {exc}). "
                f"Response keys: {sorted(meta.keys())}"
            ) from exc

        if bulk_type != expected_type:
            raise RuntimeError(
                f"Endpoint returned type={bulk_type!r}, expected {expected_type!r}. "
                f"Wrong endpoint configured?"
            )

        run.note(
            scryfall_updated_at=updated_at,
            compressed_size_expected=expected_size,
            download_uri=download_uri,
        )

        if out_path.exists() and not args.force:
            existing_size = out_path.stat().st_size
            run.note(action="skipped_existing", existing_size=existing_size)
            print(
                f"Already have {out_path.name} ({existing_size:,} bytes). "
                "Use --force to re-download."
            )
            return 0

        bytes_written, sha256 = _stream_download(download_uri, out_path, expected_size)
        run.note(
            compressed_size_actual=bytes_written,
            sha256=sha256,
            output_path=str(out_path),
        )
        run.processed()
        print(
            f"Wrote {out_path}\n"
            f"  size:   {bytes_written:,} bytes\n"
            f"  sha256: {sha256}\n"
            f"  scryfall updated_at: {updated_at}"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
