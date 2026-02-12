#!/usr/bin/env python3
"""
Evaluate frozen Jina v4 final-layer (layer 36) embeddings on the VDR training set
using InfoNCE loss with hard negatives and/or in-batch negatives.

Reports:
  - Mean InfoNCE loss
  - Top-1 accuracy (positive is ranked first)
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_from_disk


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InfoNCE evaluation on VDR training set (final layer 36)."
    )
    p.add_argument(
        "--dataset-dir",
        default="/root/autodl-tmp/hf/processed/vdr_multilingual_train_internal_arrow/en",
        help="Path to the Arrow dataset directory (load_from_disk).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    p.add_argument(
        "--negatives-mode",
        choices=("hard", "in-batch", "both"),
        default="both",
        help=(
            "'hard'     = use the 16 mined hard negatives per query;\n"
            "'in-batch' = CLIP-style, all other images in the batch are negatives;\n"
            "'both'     = report both metrics side by side."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature (tau) for InfoNCE logits.",
    )
    p.add_argument(
        "--text-col",
        default="text_layer_36",
        help="Column name for text embeddings.",
    )
    p.add_argument(
        "--image-col",
        default="image_layer_36",
        help="Column name for image embeddings.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu'.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap rows for quick testing.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def build_id_to_index(ids: List[str]) -> Dict[str, int]:
    """Map every sample id to its row index."""
    return {sample_id: idx for idx, sample_id in enumerate(ids)}


def load_all_image_embeddings(
    ds, image_col: str, device: str, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Load entire image embedding matrix into a single tensor on *device*."""
    # ds[image_col] returns a list of lists; convert to tensor once.
    return torch.tensor(ds[image_col], dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# InfoNCE with hard negatives
# ---------------------------------------------------------------------------

def eval_hard_negatives(
    query_embs: torch.Tensor,
    all_image_embs: torch.Tensor,
    negatives_ids: List[List[str]],
    row_indices: List[int],
    id_to_index: Dict[str, int],
    temperature: float,
) -> Dict[str, float]:
    """
    For each query i:
      positives = image at same row
      negatives = 16 hard-negative images looked up by id
      logits = query @ [pos, neg_1, ..., neg_16].T / tau
      label  = 0
    """
    device = query_embs.device
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for local_idx, global_idx in enumerate(row_indices):
        q = query_embs[local_idx]                # (D,)
        pos = all_image_embs[global_idx]          # (D,)

        neg_ids = negatives_ids[local_idx]
        neg_global = [id_to_index[nid] for nid in neg_ids if nid in id_to_index]
        if not neg_global:
            continue

        neg = all_image_embs[neg_global]          # (K, D)
        candidates = torch.cat([pos.unsqueeze(0), neg], dim=0)  # (1+K, D)

        logits = (q @ candidates.T) / temperature  # (1+K,)
        label = torch.tensor(0, device=device)

        loss = F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))
        total_loss += loss.item()
        total_correct += int(logits.argmax().item() == 0)
        total_count += 1

    if total_count == 0:
        return {"hard/loss": float("nan"), "hard/acc": float("nan"), "hard/n": 0}

    return {
        "hard/loss": total_loss / total_count,
        "hard/acc": total_correct / total_count,
        "hard/n": total_count,
    }


# ---------------------------------------------------------------------------
# InfoNCE with in-batch negatives (CLIP-style)
# ---------------------------------------------------------------------------

def eval_in_batch(
    query_embs: torch.Tensor,
    image_embs: torch.Tensor,
    temperature: float,
) -> Dict[str, float]:
    """
    Standard symmetric InfoNCE over the batch.
    query_embs, image_embs: (B, D) — same-index pairs are positives.
    """
    B = query_embs.shape[0]
    if B == 0:
        return {"in-batch/loss": float("nan"), "in-batch/acc": float("nan"), "in-batch/n": 0}

    logits = (query_embs @ image_embs.T) / temperature  # (B, B)
    labels = torch.arange(B, device=query_embs.device)

    loss_q2i = F.cross_entropy(logits, labels)
    loss_i2q = F.cross_entropy(logits.T, labels)
    loss = (loss_q2i + loss_i2q) / 2.0

    acc_q2i = (logits.argmax(dim=1) == labels).float().mean()
    acc_i2q = (logits.T.argmax(dim=1) == labels).float().mean()
    acc = (acc_q2i + acc_i2q) / 2.0

    return {
        "in-batch/loss": loss.item(),
        "in-batch/acc": acc.item(),
        "in-batch/n": B,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Negatives mode: {args.negatives_mode}")
    print(f"Temperature: {args.temperature}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    ds_full = load_from_disk(args.dataset_dir)
    print(f"Total rows (incl. empty-query): {len(ds_full)}")

    # Build id -> row_index over ALL rows (needed for hard-negative lookup).
    all_ids: List[str] = ds_full["id"]
    id_to_index = build_id_to_index(all_ids)

    # Pre-load all image embeddings (including empty-query rows).
    print("Loading all image embeddings into memory...")
    t0 = time.time()
    all_image_embs = load_all_image_embeddings(ds_full, args.image_col, device)
    # L2-normalise (layer 36 is unnormalised).
    all_image_embs = F.normalize(all_image_embs, p=2, dim=-1)
    print(f"  shape={all_image_embs.shape}  took {time.time() - t0:.1f}s")

    # Filter to rows with real queries.
    query_mask = [not empty for empty in ds_full["query_is_empty"]]
    query_indices = [i for i, keep in enumerate(query_mask) if keep]
    print(f"Rows with query: {len(query_indices)}")

    if args.max_samples is not None:
        query_indices = query_indices[: args.max_samples]
        print(f"Capped to: {len(query_indices)}")

    # Subset for iteration.
    ds_queries = ds_full.select(query_indices)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    do_hard = args.negatives_mode in ("hard", "both")
    do_inbatch = args.negatives_mode in ("in-batch", "both")

    agg_hard: Dict[str, float] = {"hard/loss": 0.0, "hard/acc": 0.0, "hard/n": 0}
    agg_inbatch: Dict[str, float] = {"in-batch/loss": 0.0, "in-batch/acc": 0.0, "in-batch/n": 0}

    n_batches = (len(ds_queries) + args.batch_size - 1) // args.batch_size
    print(f"\nRunning evaluation ({n_batches} batches, batch_size={args.batch_size})...")
    t0 = time.time()

    for batch_start in range(0, len(ds_queries), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(ds_queries))
        batch = ds_queries.select(range(batch_start, batch_end))

        # Text embeddings for this batch.
        q_embs = torch.tensor(batch[args.text_col], dtype=torch.float32, device=device)
        q_embs = F.normalize(q_embs, p=2, dim=-1)

        # Global indices of rows in this batch (for image lookup).
        batch_global_indices = query_indices[batch_start:batch_end]

        # --- Hard negatives ---
        if do_hard:
            has_negatives = "negatives" in batch.column_names
            if has_negatives:
                neg_ids: List[List[str]] = batch["negatives"]
                res = eval_hard_negatives(
                    query_embs=q_embs,
                    all_image_embs=all_image_embs,
                    negatives_ids=neg_ids,
                    row_indices=batch_global_indices,
                    id_to_index=id_to_index,
                    temperature=args.temperature,
                )
                n = res["hard/n"]
                agg_hard["hard/loss"] += res["hard/loss"] * n
                agg_hard["hard/acc"] += res["hard/acc"] * n
                agg_hard["hard/n"] += n

        # --- In-batch negatives ---
        if do_inbatch:
            batch_image_embs = all_image_embs[batch_global_indices]
            res = eval_in_batch(
                query_embs=q_embs,
                image_embs=batch_image_embs,
                temperature=args.temperature,
            )
            n = res["in-batch/n"]
            agg_inbatch["in-batch/loss"] += res["in-batch/loss"] * n
            agg_inbatch["in-batch/acc"] += res["in-batch/acc"] * n
            agg_inbatch["in-batch/n"] += n

        if (batch_start // args.batch_size) % 50 == 0:
            print(f"  batch {batch_start // args.batch_size + 1}/{n_batches}")

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS — InfoNCE evaluation (layer 36, frozen model)")
    print("=" * 60)

    if do_hard and agg_hard["hard/n"] > 0:
        n = agg_hard["hard/n"]
        print(f"\n  Hard-negative InfoNCE:")
        print(f"    Samples evaluated : {int(n)}")
        print(f"    Loss              : {agg_hard['hard/loss'] / n:.4f}")
        print(f"    Top-1 accuracy    : {agg_hard['hard/acc'] / n:.4f}")

    if do_inbatch and agg_inbatch["in-batch/n"] > 0:
        n = agg_inbatch["in-batch/n"]
        print(f"\n  In-batch InfoNCE (batch_size={args.batch_size}):")
        print(f"    Samples evaluated : {int(n)}")
        print(f"    Loss              : {agg_inbatch['in-batch/loss'] / n:.4f}")
        print(f"    Top-1 accuracy    : {agg_inbatch['in-batch/acc'] / n:.4f}")

    print(f"\n  Temperature         : {args.temperature}")
    print(f"  Wall time           : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
