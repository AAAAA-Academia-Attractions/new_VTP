#!/usr/bin/env python3
"""
Evaluate frozen Jina v4 final-layer (layer 36) embeddings on the VDR test set
using retrieval metrics (NDCG@k, Recall@k, MRR) per language.

Each language subset is evaluated independently:
  - Queries  = rows where query_is_empty == False
  - Corpus   = ALL image embeddings in the subset (including empty-query rows)
  - Ground truth: each query's relevant image is the one in its own row (matched by id)
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_from_disk


LANGUAGES = ("de", "en", "es", "fr", "it")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NDCG / Recall / MRR evaluation on VDR test set (final layer 36)."
    )
    p.add_argument(
        "--dataset-dir",
        default="/root/autodl-tmp/hf/processed/vdr_multilingual_test_internal_arrow",
        help="Root directory containing per-language subdirectories.",
    )
    p.add_argument(
        "--languages",
        default="all",
        help="Comma-separated language codes or 'all'.",
    )
    p.add_argument(
        "--k-values",
        default="1,3,5,10,20",
        help="Comma-separated k values for NDCG@k and Recall@k.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for chunked query-corpus matmul (memory control).",
    )
    p.add_argument(
        "--text-col",
        default="text_layer_36",
        help="Column name for text (query) embeddings.",
    )
    p.add_argument(
        "--image-col",
        default="image_layer_36",
        help="Column name for image (corpus) embeddings.",
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
        help="Cap query rows per language for quick testing.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def parse_languages(raw: str) -> Sequence[str]:
    if raw.strip().lower() == "all":
        return LANGUAGES
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def parse_k_values(raw: str) -> List[int]:
    return sorted(int(x.strip()) for x in raw.split(",") if x.strip())


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    sim: torch.Tensor,
    gt_indices: torch.Tensor,
    k_values: List[int],
) -> Dict[str, float]:
    """
    Args:
        sim:        (N_queries, N_corpus) similarity matrix
        gt_indices: (N_queries,) index into corpus for the single relevant doc per query
        k_values:   list of k for NDCG@k / Recall@k

    Returns:
        dict with keys like 'ndcg@1', 'recall@5', 'mrr', etc.
    """
    N = sim.shape[0]
    max_k = max(k_values)
    device = sim.device

    # Top-k indices per query.
    topk_vals, topk_idx = sim.topk(min(max_k, sim.shape[1]), dim=1)  # (N, max_k)

    # Relevance matrix: 1 where topk_idx matches the ground-truth, else 0.
    gt_expanded = gt_indices.unsqueeze(1).expand_as(topk_idx)  # (N, max_k)
    rel = (topk_idx == gt_expanded).float()  # (N, max_k)

    results: Dict[str, float] = {}

    # --- NDCG@k ---
    # Since there is exactly 1 relevant doc, ideal DCG = 1/log2(2) = 1.0
    # DCG@k = rel[rank] / log2(rank+2)  (rank is 0-indexed)
    log_discount = torch.log2(
        torch.arange(2, max_k + 2, dtype=torch.float32, device=device)
    )  # (max_k,)

    dcg_per_pos = rel / log_discount.unsqueeze(0)  # (N, max_k)
    ideal_dcg = 1.0  # single relevant doc at rank 0

    for k in k_values:
        dcg_at_k = dcg_per_pos[:, :k].sum(dim=1)  # (N,)
        ndcg_at_k = dcg_at_k / ideal_dcg
        results[f"ndcg@{k}"] = ndcg_at_k.mean().item()

    # --- Recall@k ---
    for k in k_values:
        hits = rel[:, :k].sum(dim=1).clamp(max=1.0)  # (N,)  — 0 or 1
        results[f"recall@{k}"] = hits.mean().item()

    # --- MRR ---
    # Find rank of first relevant doc (1-indexed); 0 if not found in top-max_k.
    first_hit = rel.argmax(dim=1)  # index of first 1 (or 0 if none)
    has_hit = rel.sum(dim=1) > 0
    reciprocal_rank = torch.where(
        has_hit,
        1.0 / (first_hit.float() + 1.0),
        torch.zeros(1, device=device),
    )
    results["mrr"] = reciprocal_rank.mean().item()

    return results


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

def evaluate_language(
    dataset_dir: str,
    lang: str,
    text_col: str,
    image_col: str,
    k_values: List[int],
    batch_size: int,
    device: str,
    max_samples: int | None,
) -> Tuple[Dict[str, float], int, int]:
    """
    Returns:
        metrics dict, n_queries, n_corpus
    """
    ds = load_from_disk(f"{dataset_dir}/{lang}")
    n_total = len(ds)
    all_ids = ds["id"]

    # --- Corpus: all image embeddings ---
    corpus = torch.tensor(ds[image_col], dtype=torch.float32, device=device)
    corpus = F.normalize(corpus, p=2, dim=-1)
    n_corpus = corpus.shape[0]

    # Build id -> corpus_index (same as row index).
    id_to_corpus_idx = {sid: i for i, sid in enumerate(all_ids)}

    # --- Queries: rows with real text ---
    query_mask = [not empty for empty in ds["query_is_empty"]]
    query_row_indices = [i for i, keep in enumerate(query_mask) if keep]
    if max_samples is not None:
        query_row_indices = query_row_indices[:max_samples]

    ds_queries = ds.select(query_row_indices)
    n_queries = len(ds_queries)

    queries = torch.tensor(ds_queries[text_col], dtype=torch.float32, device=device)
    queries = F.normalize(queries, p=2, dim=-1)

    # Ground-truth: each query's relevant image is in the same original row.
    query_ids = ds_queries["id"]
    gt_indices = torch.tensor(
        [id_to_corpus_idx[qid] for qid in query_ids],
        dtype=torch.long,
        device=device,
    )

    # --- Chunked similarity + metrics ---
    # For large corpora we compute sim in chunks over queries to save memory.
    max_k = max(k_values)
    all_topk_vals = []
    all_topk_idx = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_chunk = queries[start:end]               # (chunk, D)
        sim_chunk = q_chunk @ corpus.T              # (chunk, N_corpus)
        vals, idx = sim_chunk.topk(min(max_k, n_corpus), dim=1)
        all_topk_vals.append(vals)
        all_topk_idx.append(idx)

    topk_vals = torch.cat(all_topk_vals, dim=0)   # (N_queries, max_k)
    topk_idx = torch.cat(all_topk_idx, dim=0)

    # Reconstruct a "virtual" sim matrix that only has the top-k columns.
    # compute_retrieval_metrics needs full sim, but we can be smart:
    # build rel from topk_idx vs gt_indices directly.
    metrics = _compute_metrics_from_topk(topk_idx, gt_indices, k_values, device)

    return metrics, n_queries, n_corpus


def _compute_metrics_from_topk(
    topk_idx: torch.Tensor,
    gt_indices: torch.Tensor,
    k_values: List[int],
    device: str,
) -> Dict[str, float]:
    """Compute NDCG@k, Recall@k, MRR from precomputed top-k indices."""
    N, max_k = topk_idx.shape

    gt_expanded = gt_indices.unsqueeze(1).expand_as(topk_idx)
    rel = (topk_idx == gt_expanded).float()

    log_discount = torch.log2(
        torch.arange(2, max_k + 2, dtype=torch.float32, device=device)
    )
    dcg_per_pos = rel / log_discount.unsqueeze(0)
    ideal_dcg = 1.0

    results: Dict[str, float] = {}
    for k in k_values:
        kk = min(k, max_k)
        dcg_at_k = dcg_per_pos[:, :kk].sum(dim=1)
        results[f"ndcg@{k}"] = (dcg_at_k / ideal_dcg).mean().item()
        hits = rel[:, :kk].sum(dim=1).clamp(max=1.0)
        results[f"recall@{k}"] = hits.mean().item()

    first_hit = rel.argmax(dim=1)
    has_hit = rel.sum(dim=1) > 0
    reciprocal_rank = torch.where(
        has_hit,
        1.0 / (first_hit.float() + 1.0),
        torch.zeros(1, device=device),
    )
    results["mrr"] = reciprocal_rank.mean().item()

    return results


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_results_table(
    all_results: Dict[str, Dict[str, float]],
    k_values: List[int],
    counts: Dict[str, Tuple[int, int]],
) -> None:
    """Print a nicely formatted table of per-language + average metrics."""
    metric_keys = [f"ndcg@{k}" for k in k_values] + [f"recall@{k}" for k in k_values] + ["mrr"]

    # Header
    header = f"{'lang':>6s}  {'queries':>8s}  {'corpus':>8s}"
    for mk in metric_keys:
        header += f"  {mk:>10s}"
    print(header)
    print("-" * len(header))

    # Per-language rows
    avg: Dict[str, float] = {mk: 0.0 for mk in metric_keys}
    n_langs = 0
    for lang in sorted(all_results.keys()):
        metrics = all_results[lang]
        nq, nc = counts[lang]
        row = f"{lang:>6s}  {nq:>8d}  {nc:>8d}"
        for mk in metric_keys:
            val = metrics.get(mk, float("nan"))
            row += f"  {val:>10.4f}"
            avg[mk] += val
        print(row)
        n_langs += 1

    # Average row
    if n_langs > 0:
        print("-" * len(header))
        row = f"{'AVG':>6s}  {'':>8s}  {'':>8s}"
        for mk in metric_keys:
            row += f"  {avg[mk] / n_langs:>10.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    languages = parse_languages(args.languages)
    k_values = parse_k_values(args.k_values)

    print(f"Device     : {device}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Languages  : {', '.join(languages)}")
    print(f"k values   : {k_values}")
    print()

    all_results: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Tuple[int, int]] = {}

    t0 = time.time()
    for lang in languages:
        print(f"Evaluating {lang}...")
        metrics, nq, nc = evaluate_language(
            dataset_dir=args.dataset_dir,
            lang=lang,
            text_col=args.text_col,
            image_col=args.image_col,
            k_values=k_values,
            batch_size=args.batch_size,
            device=device,
            max_samples=args.max_samples,
        )
        all_results[lang] = metrics
        counts[lang] = (nq, nc)
        print(f"  {lang}: queries={nq}, corpus={nc}, mrr={metrics['mrr']:.4f}")

    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("RESULTS — Retrieval evaluation (layer 36, frozen model)")
    print("=" * 80 + "\n")
    print_results_table(all_results, k_values, counts)
    print(f"\nWall time: {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
