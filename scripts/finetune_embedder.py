"""Fine-tune the embedder on tag-derived contrastive pairs (M6, step 2).

Implements the recipe in ``docs/journal/2026-09-18-fine-tuning-pivot-oracle-tags-and-recipe.md``
§4a on the pairs written by ``scripts/build_training_pairs.py``:

* **Full fine-tune** of ``settings.embedding_model`` (Nomic Embed v1.5), not
  adapters — every sub-300M encoder in the literature is trained this way,
  Nomic's own supervised stage included.
* **Loss:** ``CachedMultipleNegativesRankingLoss`` — InfoNCE with in-batch
  negatives, cosine similarity, scale 20 (τ = 0.05). The cached variant
  reaches a large effective batch on a memory-bound Mac by mini-batching
  the forward pass (Sentence Transformers ≥ 5.7 required: earlier versions
  corrupted gradients in the cached losses).
* **Prefixes stay on:** ``search_query: `` on anchors, ``search_document: ``
  on positives, applied through ``prompts=`` so the trained model matches
  what ``scripts/embed.py`` and ``src/search.py`` feed it.
* **Tag-aware batching** (the structural false-negative guard): a custom
  batch sampler builds each batch so that (a) no two pairs share an anchor
  tag and (b) no positive card in the batch carries another pair's anchor
  tag (closure membership from ``card_tags_<version>.json``). Without this,
  a batch holding "sweeper → Wrath of God" and "board wipe → Damnation"
  would train Damnation as a negative for "sweeper".
* **Hyperparameters:** lr 2e-5, AdamW, weight decay 0.01, 5 % linear warmup
  then linear decay, **one epoch** (Nussbaum et al.: more epochs hurt),
  grad clip 1.0, fp32 on MPS.
* **Evaluation:** an in-training dev evaluator on a slice of *training* tags
  (sanity: is the loss doing anything?), and after training the
  **held-out-tag probe** — retrieval over the full corpus for tags the
  trainer never saw, queried by their label. Optional ``--nanobeir`` runs
  the general-retrieval forgetting probe (downloads small BEIR subsets).
  All scores go to one ``experiment_runs`` row.

The tuned checkpoint lands in ``models/<run-name>/``. To evaluate it through
the cascade, point ``EMBEDDING_MODEL`` at that directory in ``.env``,
re-run ``scripts/embed.py`` (the ``embedding_version`` string follows the
model path, so the base vectors stay untouched), then run the configs in
``configs/``.

Usage::

    python scripts/finetune_embedder.py --smoke                # 5 steps on 2k pairs, MPS sanity
    python scripts/finetune_embedder.py                        # full run, pairs_v1
    python scripts/finetune_embedder.py --batch-size 128 --nanobeir
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import psycopg
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import DefaultBatchSampler
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss

import src.utils.quiet  # noqa: F401
from src.config import settings
from src.db.experiment_log import log_experiment
from src.logging_utils import PipelineRun
from src.preprocess_text import (
    NOMIC_DOCUMENT_PREFIX,
    NOMIC_QUERY_PREFIX,
    build_embedding_text,
    load_keyword_dict,
)
from src.utils.device import select_device

PROMPTS = {"anchor": NOMIC_QUERY_PREFIX, "positive": NOMIC_DOCUMENT_PREFIX}


# ---- Tag-aware batch sampler ---------------------------------------------


class TagDisjointBatchSampler(DefaultBatchSampler):
    """Greedy batches with no shared anchor tag and no positive that carries another
    pair's anchor tag.

    Pairs that cannot be placed within ``lookahead`` candidates are deferred; at
    the end of the epoch the deferred remainder is packed loosely (logged) rather
    than dropped, so every pair is seen once.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
        *,
        pair_tags: list[str],
        card_tag_sets: list[frozenset[str]],
        lookahead: int = 4096,
    ) -> None:
        super().__init__(dataset, batch_size, drop_last, valid_label_columns, generator, seed)
        self.n_items = len(dataset)
        self.pair_tags = pair_tags
        self.card_tag_sets = card_tag_sets
        self.lookahead = lookahead
        self.loose_pairs = 0

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator and self.seed:
            self.generator.manual_seed(self.seed + getattr(self, "epoch", 0))
        order = torch.randperm(self.n_items, generator=self.generator).tolist()
        pending: list[int] = order
        self.loose_pairs = 0
        while pending:
            batch: list[int] = []
            anchor_tags: set[str] = set()
            card_tags: set[str] = set()
            deferred: list[int] = []
            scanned = 0
            for idx in pending:
                if len(batch) >= self.batch_size:
                    deferred.append(idx)
                    continue
                if scanned >= self.lookahead:
                    deferred.append(idx)
                    continue
                scanned += 1
                t = self.pair_tags[idx]
                ctags = self.card_tag_sets[idx]
                if t in card_tags or t in anchor_tags or (ctags & anchor_tags):
                    deferred.append(idx)
                    continue
                batch.append(idx)
                anchor_tags.add(t)
                card_tags |= ctags
            if len(batch) < self.batch_size and deferred and scanned < self.lookahead:
                # Constraint-starved tail: pack loosely so nothing is dropped.
                need = self.batch_size - len(batch)
                batch.extend(deferred[:need])
                self.loose_pairs += min(need, len(deferred))
                deferred = deferred[need:]
            pending = deferred
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self) -> int:
        n = self.n_items
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


class TagAwareTrainer(SentenceTransformerTrainer):
    """SentenceTransformerTrainer with the tag-disjoint batch sampler plugged in."""

    pair_tags: ClassVar[list[str]] = []
    card_tag_sets: ClassVar[list[frozenset[str]]] = []
    last_sampler: ClassVar[TagDisjointBatchSampler | None] = None

    def get_batch_sampler(
        self, dataset, batch_size, drop_last, valid_label_columns=None, generator=None, seed=0
    ):  # type: ignore[override]
        if len(dataset) != len(self.pair_tags):  # eval datasets etc.
            return super().get_batch_sampler(
                dataset, batch_size, drop_last, valid_label_columns, generator, seed
            )
        sampler = TagDisjointBatchSampler(
            dataset,
            batch_size,
            drop_last,
            valid_label_columns,
            generator,
            seed,
            pair_tags=self.pair_tags,
            card_tag_sets=self.card_tag_sets,
        )
        TagAwareTrainer.last_sampler = sampler
        return sampler


# ---- Data loading ----------------------------------------------------------


def _load_pairs(path: Path, limit: int | None, rng: random.Random) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if limit and len(rows) > limit:
        rows = rng.sample(rows, limit)
    return rows


def _dev_evaluator(
    pairs: list[dict[str, Any]], n_queries: int, rng: random.Random
) -> InformationRetrievalEvaluator:
    """Small in-training sanity probe on TRAINING tags: query = anchor, corpus = cards
    of the sampled tags (capped), relevant = the anchor's own tag pool."""
    by_anchor: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, str] = {}
    for p in pairs:
        by_anchor[p["anchor"]].add(p["oracle_id"])
        texts.setdefault(p["oracle_id"], p["positive"])
    anchors = rng.sample(sorted(by_anchor), min(n_queries, len(by_anchor)))
    corpus_ids: set[str] = set()
    for a in anchors:
        corpus_ids |= by_anchor[a]
    # Pad the corpus with distractors so precision means something.
    distractors = [c for c in texts if c not in corpus_ids]
    corpus_ids |= set(rng.sample(distractors, min(2000, len(distractors))))
    return InformationRetrievalEvaluator(
        queries={f"q{i}": a for i, a in enumerate(anchors)},
        corpus={c: texts[c] for c in corpus_ids},
        relevant_docs={f"q{i}": by_anchor[a] for i, a in enumerate(anchors)},
        name="dev-train-tags",
        query_prompt=NOMIC_QUERY_PREFIX,
        corpus_prompt=NOMIC_DOCUMENT_PREFIX,
        show_progress_bar=False,
        precision_recall_at_k=[10],
        map_at_k=[10],
        ndcg_at_k=[10],
        mrr_at_k=[10],
        accuracy_at_k=[1, 10],
    )


def _holdout_evaluator(
    manifest: dict[str, Any], holdout_path: Path
) -> InformationRetrievalEvaluator:
    """Held-out-tag probe: query = held-out tag LABEL, corpus = the whole ingested
    corpus (face 0 text, reminder-augmented), relevant = that tag's closure pool."""
    relevant: dict[str, set[str]] = defaultdict(set)
    label_by_tag: dict[str, str] = {}
    for line in holdout_path.open(encoding="utf-8"):
        p = json.loads(line)
        if p["anchor_kind"] != "label":
            continue
        label_by_tag[p["tag_id"]] = p["anchor"]
        relevant[p["tag_id"]].add(p["oracle_id"])
    keyword_dict = load_keyword_dict()
    corpus: dict[str, str] = {}
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT oracle_id::text, oracle_text, keywords FROM cards "
            "WHERE face_index = 0 AND oracle_text <> ''"
        )
        for oid, text, kws in cur:
            corpus[oid] = build_embedding_text(text, kws or [], keyword_dict)
    queries = {t: label_by_tag[t] for t in relevant}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs={t: relevant[t] & set(corpus) for t in relevant},
        name="holdout-tags",
        query_prompt=NOMIC_QUERY_PREFIX,
        corpus_prompt=NOMIC_DOCUMENT_PREFIX,
        show_progress_bar=True,
        batch_size=64,
        precision_recall_at_k=[10, 100],
        map_at_k=[10],
        ndcg_at_k=[10],
        mrr_at_k=[10],
        accuracy_at_k=[1, 10],
    )


# ---- Main ------------------------------------------------------------------


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs-version", default="v1")
    parser.add_argument("--run-name", default=None, help="Output dir name under models/.")
    parser.add_argument("--batch-size", type=int, default=256, help="Effective (cached) batch.")
    parser.add_argument("--mini-batch-size", type=int, default=32, help="CachedMNRL forward chunk.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=20.0, help="MNRL scale (1/temperature).")
    parser.add_argument(
        "--max-seq-length", type=int, default=256, help="Token cap during training."
    )
    parser.add_argument("--limit", type=int, default=None, help="Subsample training pairs.")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--dev-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--nanobeir", action="store_true", help="Run the NanoBEIR forgetting probe."
    )
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2k pairs, 5 steps, no probes.")
    parser.add_argument("--dry-run", action="store_true", help="No experiment_runs row.")
    args = parser.parse_args()

    if args.smoke:
        args.limit, args.max_steps, args.skip_holdout = 2000, 5, True
        args.batch_size, args.eval_steps, args.dev_queries = 64, 3, 20
        args.run_name = args.run_name or "smoke"

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()
    run_name = args.run_name or f"nomic-mtg-{args.pairs_version}-{time.strftime('%Y%m%d-%H%M')}"
    out_dir = settings.repo_root / "models" / run_name
    training_dir = settings.training_dir
    pairs_path = training_dir / f"pairs_{args.pairs_version}.jsonl"
    holdout_path = training_dir / f"holdout_{args.pairs_version}.jsonl"
    manifest = json.loads((training_dir / f"manifest_{args.pairs_version}.json").read_text())
    card_tags = json.loads((training_dir / f"card_tags_{args.pairs_version}.json").read_text())

    with PipelineRun(
        "finetune_embedder",
        inputs={
            "base_model": settings.embedding_model,
            "pairs_version": args.pairs_version,
            "run_name": run_name,
            "batch_size": args.batch_size,
            "mini_batch_size": args.mini_batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "scale": args.scale,
            "limit": args.limit,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "device": str(device),
            "smoke": args.smoke,
        },
    ) as run:
        pairs = _load_pairs(pairs_path, args.limit, rng)
        run.event("pairs_loaded", count=len(pairs), manifest_train_pairs=manifest["train_pairs"])
        print(f"\n  Run:         {run_name}")
        print(f"  Base model:  {settings.embedding_model}")
        print(f"  Pairs:       {len(pairs):,}  (from {pairs_path.name})")
        print(
            f"  Batch:       {args.batch_size} (mini {args.mini_batch_size})  lr={args.lr}  epochs={args.epochs}"
        )
        print(f"  Device:      {device}\n")

        dataset = Dataset.from_dict(
            {"anchor": [p["anchor"] for p in pairs], "positive": [p["positive"] for p in pairs]}
        )
        TagAwareTrainer.pair_tags = [p["tag_id"] for p in pairs]
        TagAwareTrainer.card_tag_sets = [
            frozenset(card_tags.get(p["oracle_id"], ())) for p in pairs
        ]

        model = SentenceTransformer(
            settings.embedding_model, device=str(device), trust_remote_code=True
        )
        # Pairs are short (positive text p95 ~470 chars ≈ 120 tokens); a tight cap
        # bounds padding and memory during training. Inference keeps settings.max_length.
        model.max_seq_length = args.max_seq_length
        loss = CachedMultipleNegativesRankingLoss(
            model, scale=args.scale, mini_batch_size=args.mini_batch_size
        )
        dev_eval = _dev_evaluator(pairs, args.dev_queries, rng)

        targs = SentenceTransformerTrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_steps=args.warmup_ratio,  # float in [0,1) = ratio (Transformers v5 convention)
            weight_decay=args.weight_decay,
            max_grad_norm=1.0,
            lr_scheduler_type="linear",
            fp16=False,
            bf16=False,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            logging_steps=10,
            save_strategy="no",
            seed=args.seed,
            report_to="none",
            dataloader_drop_last=False,
            prompts=PROMPTS,
            run_name=run_name,
        )
        trainer = TagAwareTrainer(
            model=model,
            args=targs,
            train_dataset=dataset,
            loss=loss,
            evaluator=dev_eval,
        )

        print("  Base-model dev probe (before training):")
        base_dev = dev_eval(model)
        for k_, v in sorted(base_dev.items()):
            if "ndcg" in k_ or "map" in k_ or "mrr" in k_:
                print(f"    {k_:45s} {v:.4f}")
        run.note(**{f"base_{k}": v for k, v in base_dev.items()})

        t0 = time.perf_counter()
        train_out = trainer.train()
        train_s = time.perf_counter() - t0
        sampler = TagAwareTrainer.last_sampler
        run.note(
            train_seconds=round(train_s, 1),
            train_loss=train_out.training_loss,
            steps=train_out.global_step,
        )
        print(
            f"\n  Trained {train_out.global_step} steps in {train_s / 60:.1f} min, loss {train_out.training_loss:.4f}"
        )

        model.save_pretrained(str(out_dir))
        run.note(saved_to=str(out_dir))

        print("\n  Tuned-model dev probe:")
        tuned_dev = dev_eval(model)
        for k_, v in sorted(tuned_dev.items()):
            if "ndcg" in k_ or "map" in k_ or "mrr" in k_:
                print(f"    {k_:45s} {v:.4f}   (base {base_dev.get(k_, float('nan')):.4f})")

        metrics: dict[str, Any] = {
            "train_loss": train_out.training_loss,
            "steps": train_out.global_step,
            "train_seconds": round(train_s, 1),
            **{f"dev_base_{k}": v for k, v in base_dev.items()},
            **{f"dev_tuned_{k}": v for k, v in tuned_dev.items()},
        }

        if not args.skip_holdout:
            print("\n  Held-out-tag probe (full corpus) — base then tuned:")
            hold_eval = _holdout_evaluator(manifest, holdout_path)
            base_model = SentenceTransformer(
                settings.embedding_model, device=str(device), trust_remote_code=True
            )
            hold_base = hold_eval(base_model)
            del base_model
            hold_tuned = hold_eval(model)
            for k_, v in sorted(hold_tuned.items()):
                if any(s in k_ for s in ("ndcg", "map", "mrr", "recall@100")):
                    print(f"    {k_:45s} {v:.4f}   (base {hold_base.get(k_, float('nan')):.4f})")
            metrics.update({f"holdout_base_{k}": v for k, v in hold_base.items()})
            metrics.update({f"holdout_tuned_{k}": v for k, v in hold_tuned.items()})

        if args.nanobeir:
            from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator

            nb = NanoBEIREvaluator(
                dataset_names=["scifact", "nfcorpus", "fiqa2018"],
                query_prompts=NOMIC_QUERY_PREFIX,
                corpus_prompts=NOMIC_DOCUMENT_PREFIX,
                show_progress_bar=False,
            )
            print("\n  NanoBEIR forgetting probe (tuned):")
            nb_tuned = nb(model)
            for k_, v in sorted(nb_tuned.items()):
                if "ndcg@10" in k_:
                    print(f"    {k_:45s} {v:.4f}")
            metrics.update({f"nanobeir_tuned_{k}": v for k, v in nb_tuned.items()})

        run.note(**{k: v for k, v in metrics.items() if isinstance(v, int | float)})
        (out_dir / "training_summary.json").write_text(
            json.dumps({"args": vars(args), "metrics": metrics}, indent=1, default=str) + "\n"
        )

        if args.dry_run or args.smoke:
            print("\n  No experiment_runs row (smoke/dry-run).")
            return 0
        record = log_experiment(
            eval_set_version=f"holdout-tags-{args.pairs_version}",
            config={
                "kind": "finetune_embedder",
                "run_name": run_name,
                "base_model": settings.embedding_model,
                "pairs_version": args.pairs_version,
                "pairs_used": len(pairs),
                "loss": "CachedMultipleNegativesRankingLoss",
                "scale": args.scale,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "tag_aware_batching": True,
                "loose_pairs_last_epoch": getattr(sampler, "loose_pairs", None),
                "output_dir": str(out_dir),
            },
            metrics={k: v for k, v in metrics.items() if isinstance(v, int | float)},
            notes=f"Embedder fine-tune {run_name}; probes: dev(train tags), holdout tags"
            + (", NanoBEIR" if args.nanobeir else ""),
        )
        run.note(experiment_run_id=record.id)
        print(f"\n  Logged as experiment_runs id={record.id}")
        print(
            f"  Next: EMBEDDING_MODEL={out_dir.relative_to(settings.repo_root)} in .env → scripts/embed.py → configs/"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
