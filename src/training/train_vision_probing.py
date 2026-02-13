#!/usr/bin/env python3
"""
Train one InternalRepresentationProbing per image layer (1-36) using InfoNCE
with hard negatives.

Setup:
  - Text query  = text_layer_36 (L2-normalised, frozen, NOT probed)
  - Positive img = probe_L(image_layer_L) of the same row
  - Neg images   = probe_L(image_layer_L) of 16 hard-negative rows
  - Loss: InfoNCE  =>  text_query should be closest to probed positive image
  - L1 regularisation on probe.p

Evaluation (periodic):
  - Per-language NDCG/Recall/MRR on VDR test set
  - Query  = normalised text_layer_36 (no probe)
  - Corpus = probe_L(image_layer_L) for each layer L

Best checkpoint saved per layer independently based on avg NDCG@5.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

import wandb

# ---------------------------------------------------------------------------
# Allow importing the probing model from model_definition
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from model_definition.internal_representation_probing import InternalRepresentationProbing  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LANGUAGES = ("de", "en", "es", "fr", "it")
TEXT_ANCHOR_COL = "text_layer_36"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vision-side probing models.")
    # data
    p.add_argument("--train-dataset-dir",
                    default="/root/autodl-tmp/hf/processed/vdr_multilingual_train_internal_arrow/en")
    p.add_argument("--test-dataset-dir",
                    default="/root/autodl-tmp/hf/processed/vdr_multilingual_test_internal_arrow")
    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-l1", type=float, default=1e-4,
                    help="L1 regularisation strength on probe.p")
    p.add_argument("--seed", type=int, default=42)
    # eval
    p.add_argument("--eval-every-n-epochs", type=int, default=1)
    p.add_argument("--k-values", default="1,3,5,10,20")
    p.add_argument("--eval-languages", default="all")
    # infra
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", default="/root/autodl-tmp/new_VTP/checkpoints/vision_probing")
    p.add_argument("--run-name", default=None)
    # wandb
    p.add_argument("--wandb-entity", default="5a-academia-attractions")
    p.add_argument("--wandb-project", default="MMIR")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def parse_k_values(raw: str) -> List[int]:
    return sorted(int(x.strip()) for x in raw.split(",") if x.strip())


def parse_languages(raw: str) -> Sequence[str]:
    if raw.strip().lower() == "all":
        return LANGUAGES
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_image_layer_cols(ds) -> List[str]:
    """Return sorted list of image_layer_XX columns present in the dataset."""
    return sorted([c for c in ds.column_names if c.startswith("image_layer_")])


def layer_index_from_col(col: str) -> int:
    """'image_layer_03' -> 3"""
    return int(col.split("_")[-1])


class VDRTrainDataset(TorchDataset):
    """
    Wraps the HF Arrow dataset.  Each __getitem__ returns:
      - text_anchor : (D,) float32   – normalised text_layer_36
      - image_layers: dict[col_name -> (D,)] float32  – raw image per layer
      - neg_image_layers: dict[col_name -> (K, D)] float32 – raw neg images per layer
    """

    def __init__(
        self,
        ds: Dataset,
        id_to_index: Dict[str, int],
        image_layer_cols: List[str],
        full_ds: Dataset,
    ):
        # ds is the filtered (non-empty query) subset
        # full_ds is the complete dataset (for negative lookups)
        self.ds = ds
        self.id_to_index = id_to_index
        self.image_layer_cols = image_layer_cols
        self.full_ds = full_ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]

        text_anchor = torch.tensor(row[TEXT_ANCHOR_COL], dtype=torch.float32)
        text_anchor = F.normalize(text_anchor, p=2, dim=-1)

        # Positive image layers
        pos_layers = {}
        for col in self.image_layer_cols:
            pos_layers[col] = torch.tensor(row[col], dtype=torch.float32)

        # Hard-negative image layers
        neg_ids = row["negatives"]
        neg_global_indices = [self.id_to_index[nid] for nid in neg_ids if nid in self.id_to_index]
        K = len(neg_global_indices)

        neg_layers = {}
        if K > 0:
            #neg_rows = self.full_ds.select(neg_global_indices)
            neg_rows = self.full_ds[neg_global_indices]
            for col in self.image_layer_cols:
                neg_layers[col] = torch.tensor(neg_rows[col], dtype=torch.float32)  # (K, D)
        else:
            D = pos_layers[self.image_layer_cols[0]].shape[-1]
            for col in self.image_layer_cols:
                neg_layers[col] = torch.zeros(0, D, dtype=torch.float32)

        return text_anchor, pos_layers, neg_layers, K



def collate_fn(batch):
    """
    Custom collate that pads negative counts to the max K in the batch.
    Returns:
      text_anchors: (B, D)
      pos_images:   dict[col -> (B, D)]
      neg_images:   dict[col -> (B, K_max, D)]
      neg_mask:     (B, K_max) bool – True where the negative is real
    """
    text_anchors = torch.stack([b[0] for b in batch])
    B = len(batch)

    col_names = list(batch[0][1].keys())
    D = batch[0][1][col_names[0]].shape[-1]
    K_max = max(b[3] for b in batch)
    if K_max == 0:
        K_max = 1  # avoid zero-size tensors

    pos_images = {}
    neg_images = {}
    for col in col_names:
        pos_images[col] = torch.stack([b[1][col] for b in batch])  # (B, D)
        neg_tensor = torch.zeros(B, K_max, D, dtype=torch.float32)
        for i, b in enumerate(batch):
            k = b[3]
            if k > 0:
                neg_tensor[i, :k] = b[2][col]
        neg_images[col] = neg_tensor

    neg_mask = torch.zeros(B, K_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b[3]
        neg_mask[i, :k] = True

    return text_anchors, pos_images, neg_images, neg_mask


# ---------------------------------------------------------------------------
# InfoNCE loss for one layer
# ---------------------------------------------------------------------------
def infonce_loss_one_layer(
    text_query: torch.Tensor,       # (B, D) normalised
    probed_pos: torch.Tensor,       # (B, D)
    probed_neg: torch.Tensor,       # (B, K, D)
    neg_mask: torch.Tensor,         # (B, K) bool
    logit_scale: torch.Tensor,      # scalar
) -> Tuple[torch.Tensor, float]:
    """
    Returns (loss, accuracy).
    """
    B, K, D = probed_neg.shape
    # candidates: (B, 1+K, D)
    candidates = torch.cat([probed_pos.unsqueeze(1), probed_neg], dim=1)

    # similarity: (B, 1+K)
    scale = logit_scale.exp()
    logits = (text_query.unsqueeze(1) * candidates).sum(dim=-1) * scale  # (B, 1+K)

    # Mask out padded negatives with large negative value
    # mask shape: (B, 1+K) – first column (positive) is always valid
    full_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=neg_mask.device), neg_mask], dim=1)
    logits = logits.masked_fill(~full_mask, -1e9)

    labels = torch.zeros(B, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)

    acc = (logits.argmax(dim=1) == 0).float().mean().item()
    return loss, acc


# ---------------------------------------------------------------------------
# NDCG / Recall / MRR  (reused from eval script logic)
# ---------------------------------------------------------------------------
def compute_metrics_from_topk(
    topk_idx: torch.Tensor,
    gt_indices: torch.Tensor,
    k_values: List[int],
    device: str,
) -> Dict[str, float]:
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


@torch.no_grad()
def evaluate_probes_on_test(
    probes: Dict[str, InternalRepresentationProbing],
    test_dataset_dir: str,
    image_layer_cols: List[str],
    languages: Sequence[str],
    k_values: List[int],
    device: str,
    eval_batch_size: int = 256,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns: {layer_col: {lang: {metric: value}}}
    """
    for probe in probes.values():
        probe.eval()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for col in image_layer_cols:
        results[col] = {}

    for lang in tqdm(languages, desc="Eval languages"):
        ds = load_from_disk(f"{test_dataset_dir}/{lang}")
        all_ids = ds["id"]
        id_to_idx = {sid: i for i, sid in enumerate(all_ids)}

        # Queries: non-empty text
        query_mask = [not empty for empty in ds["query_is_empty"]]
        query_row_indices = [i for i, keep in enumerate(query_mask) if keep]
        ds_queries = ds.select(query_row_indices)
        n_queries = len(ds_queries)

        # Text queries (normalised, no probe)
        queries = torch.tensor(ds_queries[TEXT_ANCHOR_COL], dtype=torch.float32, device=device)
        queries = F.normalize(queries, p=2, dim=-1)

        # Ground truth
        query_ids = ds_queries["id"]
        gt_indices = torch.tensor(
            [id_to_idx[qid] for qid in query_ids], dtype=torch.long, device=device
        )

        max_k = max(k_values)

        for col in image_layer_cols:
            probe = probes[col]
            # Build probed corpus from ALL rows
            all_image_raw = torch.tensor(ds[col], dtype=torch.float32, device=device)
            corpus = probe(all_image_raw)  # already normalised by the probe
            n_corpus = corpus.shape[0]

            # Chunked topk
            all_topk_idx = []
            for start in range(0, n_queries, eval_batch_size):
                end_idx = min(start + eval_batch_size, n_queries)
                q_chunk = queries[start:end_idx]
                sim_chunk = q_chunk @ corpus.T
                _, idx = sim_chunk.topk(min(max_k, n_corpus), dim=1)
                all_topk_idx.append(idx)

            topk_idx = torch.cat(all_topk_idx, dim=0)
            metrics = compute_metrics_from_topk(topk_idx, gt_indices, k_values, device)
            results[col][lang] = metrics

    for probe in probes.values():
        probe.train()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    k_values = parse_k_values(args.k_values)
    eval_languages = parse_languages(args.eval_languages)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------
    run_name = args.run_name or f"vision_probing_lr{args.lr}_l1{args.lambda_l1}"
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    print(f"Loading training data from {args.train_dataset_dir} ...")
    ds_full = load_from_disk(args.train_dataset_dir)
    print(f"  Total rows: {len(ds_full)}")

    image_layer_cols = discover_image_layer_cols(ds_full)
    print(f"  Image layer columns ({len(image_layer_cols)}): {image_layer_cols[0]} .. {image_layer_cols[-1]}")

    # id -> row index for negative lookups
    all_ids = ds_full["id"]
    id_to_index = {sid: i for i, sid in enumerate(all_ids)}

    # Filter to non-empty query rows for training
    query_mask = [not empty for empty in ds_full["query_is_empty"]]
    query_indices = [i for i, keep in enumerate(query_mask) if keep]
    ds_train = ds_full.select(query_indices)
    print(f"  Training rows (non-empty query): {len(ds_train)}")

    train_dataset = VDRTrainDataset(
        ds=ds_train,
        id_to_index=id_to_index,
        image_layer_cols=image_layer_cols,
        full_ds=ds_full,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Create probes (one per layer)
    # ------------------------------------------------------------------
    D = len(ds_full[0][image_layer_cols[0]])
    probes: Dict[str, InternalRepresentationProbing] = {}
    for col in image_layer_cols:
        probes[col] = InternalRepresentationProbing(input_dim=D).to(device)

    # Single optimiser over all probes
    all_params = []
    for probe in probes.values():
        all_params.extend(probe.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    print(f"\nDevice: {device}")
    print(f"Probes: {len(probes)}  |  params per probe: p({D},) + logit_scale(1,)")
    print(f"Epochs: {args.epochs}  |  batch_size: {args.batch_size}  |  lr: {args.lr}")
    print(f"L1 lambda: {args.lambda_l1}")
    print(f"Eval every {args.eval_every_n_epochs} epoch(s) on languages: {eval_languages}")

    # ------------------------------------------------------------------
    # Best-model tracking
    # ------------------------------------------------------------------
    best_ndcg5: Dict[str, float] = {col: 0.0 for col in image_layer_cols}
    best_epoch: Dict[str, int] = {col: -1 for col in image_layer_cols}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        for probe in probes.values():
            probe.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for text_anchors, pos_images, neg_images, neg_mask in pbar:
            print("here")
            text_anchors = text_anchors.to(device)     # (B, D) already normalised
            neg_mask = neg_mask.to(device)              # (B, K)

            total_loss = torch.tensor(0.0, device=device)
            log_dict: Dict[str, float] = {}

            for col in image_layer_cols:
                probe = probes[col]
                pos_raw = pos_images[col].to(device)      # (B, D)
                neg_raw = neg_images[col].to(device)      # (B, K, D)

                # Probe positive
                probed_pos = probe(pos_raw)                # (B, D)

                # Probe negatives: reshape to (B*K, D), probe, reshape back
                B, K, D_dim = neg_raw.shape
                neg_flat = neg_raw.reshape(B * K, D_dim)
                probed_neg_flat = probe(neg_flat)
                probed_neg = probed_neg_flat.reshape(B, K, D_dim)

                logit_scale = probe.get_logit_scale()

                loss_layer, acc_layer = infonce_loss_one_layer(
                    text_query=text_anchors,
                    probed_pos=probed_pos,
                    probed_neg=probed_neg,
                    neg_mask=neg_mask,
                    logit_scale=logit_scale,
                )

                # L1 on p
                l1_reg = args.lambda_l1 * probe.p.abs().mean()

                layer_total = loss_layer + l1_reg
                total_loss = total_loss + layer_total

                layer_idx = layer_index_from_col(col)
                log_dict[f"train/loss_layer_{layer_idx:02d}"] = loss_layer.item()
                #log_dict[f"train/l1_reg_layer_{layer_idx:02d}"] = l1_reg.item()
                log_dict[f"train/acc_layer_{layer_idx:02d}"] = acc_layer
                #log_dict[f"train/logit_scale_layer_{layer_idx:02d}"] = logit_scale.exp().item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            log_dict["train/total_loss"] = total_loss.item()
            log_dict["train/epoch"] = epoch
            wandb.log(log_dict, step=global_step)

            epoch_loss += total_loss.item()
            epoch_steps += 1
            global_step += 1

            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t_epoch
        print(f"Epoch {epoch} done — avg_loss={avg_epoch_loss:.4f}  time={elapsed:.1f}s")

        # --------------------------------------------------------------
        # Log |p| mean absolute value per probe
        # --------------------------------------------------------------
        p_log: Dict[str, float] = {}
        print("\n  Probe |p| mean absolute values:")
        for col in image_layer_cols:
            layer_idx = layer_index_from_col(col)
            p_abs_mean = probes[col].p.abs().mean().item()
            p_log[f"probe/p_abs_mean_layer_{layer_idx:02d}"] = p_abs_mean
            print(f"    layer {layer_idx:02d}: |p|_mean = {p_abs_mean:.6f}")
        wandb.log(p_log, step=global_step)

        # --------------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------------
        if epoch % args.eval_every_n_epochs == 0:
            print(f"\n--- Evaluation after epoch {epoch} ---")
            t_eval = time.time()
            eval_results = evaluate_probes_on_test(
                probes=probes,
                test_dataset_dir=args.test_dataset_dir,
                image_layer_cols=image_layer_cols,
                languages=eval_languages,
                k_values=k_values,
                device=device,
            )

            eval_log: Dict[str, float] = {}
            for col in image_layer_cols:
                layer_idx = layer_index_from_col(col)
                avg_metrics: Dict[str, float] = {}
                for lang, metrics in eval_results[col].items():
                    for mk, val in metrics.items():
                        key = f"eval/{lang}/{mk}_layer_{layer_idx:02d}"
                        eval_log[key] = val
                        avg_metrics[mk] = avg_metrics.get(mk, 0.0) + val

                n_langs = len(eval_results[col])
                for mk in avg_metrics:
                    avg_val = avg_metrics[mk] / n_langs
                    eval_log[f"eval/avg/{mk}_layer_{layer_idx:02d}"] = avg_val

                # Best-model check on avg NDCG@5
                avg_ndcg5 = avg_metrics.get("ndcg@5", 0.0) / max(n_langs, 1)
                prev_best = best_ndcg5[col]
                if avg_ndcg5 > prev_best:
                    best_ndcg5[col] = avg_ndcg5
                    best_epoch[col] = epoch
                    ckpt_path = save_dir / f"best_probe_layer_{layer_idx:02d}.pt"
                    torch.save(probes[col].state_dict(), ckpt_path)
                    print(f"  NEW BEST layer {layer_idx:02d}: ndcg@5={avg_ndcg5:.4f} (prev={prev_best:.4f}, epoch {epoch}) -> {ckpt_path}")
                else:
                    print(f"  layer {layer_idx:02d}: ndcg@5={avg_ndcg5:.4f} (best={prev_best:.4f} @ epoch {best_epoch[col]})")

                eval_log[f"eval/best_ndcg5_layer_{layer_idx:02d}"] = best_ndcg5[col]

            wandb.log(eval_log, step=global_step)
            print(f"  Eval took {time.time() - t_eval:.1f}s\n")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — Best NDCG@5 per layer")
    print("=" * 70)
    for col in image_layer_cols:
        layer_idx = layer_index_from_col(col)
        print(f"  layer {layer_idx:02d}: ndcg@5={best_ndcg5[col]:.4f}  (epoch {best_epoch[col]})")
    print("=" * 70)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
