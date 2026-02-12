#!/usr/bin/env python3
"""
Replace `query` and `image` fields with internal-layer activations in Arrow format.

This script:
- loads `llamaindex/vdr-multilingual-train` from HF cache
- computes `encode_text_internal_representation` and
  `encode_image_internal_representation`
- writes a transformed Arrow dataset (via `save_to_disk`) where:
  - `query` is replaced by text layer columns
  - `image` is replaced by image layer columns
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from datasets import Dataset, Features, Sequence as HFSequence, Value, load_dataset
from transformers import AutoModel

DEFAULT_DATASET_NAME = "llamaindex/vdr-multilingual-train"
DEFAULT_SUBSETS = ("en", "es", "it", "de", "fr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VDR query/image fields to per-layer internal activations in Arrow dataset format."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--subsets",
        default="all",
        help="Comma-separated subsets (en,es,it,de,fr) or 'all'.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/hf")
    parser.add_argument(
        "--model-dir", default="/root/autodl-tmp/new_VTP/jina-embeddings-v4-local"
    )
    parser.add_argument("--task", default="retrieval")
    parser.add_argument("--prompt-name", default="query")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument(
        "--layer-selection",
        default="all",
        help="Layer(s) to keep: 'all' or comma-separated indices like '35' or '30,31,35'.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=8,
        help="Model encode batch size for dataset.map.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
        help="Output dtype for saved embedding tensors.",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/hf/processed/vdr_multilingual_train_internal_arrow",
        help="Where to save transformed Arrow dataset(s).",
    )
    parser.add_argument(
        "--max-samples-per-subset",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    parser.add_argument(
        "--keep-empty-query",
        action="store_true",
        help="If set, keeps rows with empty/null query instead of filtering them out.",
    )
    parser.add_argument(
        "--empty-query-text-policy",
        choices=("zero-vector", "encode-empty"),
        default="zero-vector",
        help=(
            "How to handle text embeddings for empty queries when --keep-empty-query is enabled: "
            "'zero-vector' skips text encoding for empty rows and fills zeros; "
            "'encode-empty' encodes empty-string query like regular rows."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu', or explicit CUDA device like 'cuda:0'.",
    )
    parser.add_argument(
        "--writer-batch-size",
        type=int,
        default=32,
        help="Arrow writer batch size for dataset.map.",
    )
    return parser.parse_args()


def resolve_subsets(raw: str) -> Sequence[str]:
    if raw.strip().lower() == "all":
        return DEFAULT_SUBSETS
    subsets = [s.strip() for s in raw.split(",") if s.strip()]
    if not subsets:
        raise ValueError("No valid subset provided.")
    return subsets


def resolve_device(raw: str) -> str:
    requested = "cuda" if raw == "auto" else raw
    if "cuda" not in requested:
        return requested

    if not torch.cuda.is_available():
        if raw == "auto":
            print("CUDA not available. Falling back to CPU.")
            return "cpu"
        raise RuntimeError(
            f"Requested device '{raw}' but CUDA is not available in this environment."
        )

    # Guard against architectures unsupported by the current PyTorch build.
    # Example: device capability sm_120 with a build that only has kernels up to sm_90.
    cuda_index = 0
    if ":" in requested:
        try:
            cuda_index = int(requested.split(":", 1)[1])
        except ValueError:
            cuda_index = 0
    major, minor = torch.cuda.get_device_capability(cuda_index)
    current_sm = f"sm_{major}{minor}"
    supported_sms = set(torch.cuda.get_arch_list())
    if supported_sms and current_sm not in supported_sms:
        if raw == "auto":
            print(
                f"CUDA device capability {current_sm} is not supported by this PyTorch build "
                f"(supports: {', '.join(sorted(supported_sms))}). Falling back to CPU."
            )
            return "cpu"
        raise RuntimeError(
            f"Requested device '{raw}' with capability {current_sm}, but this PyTorch build "
            f"supports only: {', '.join(sorted(supported_sms))}."
        )

    return requested


def output_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def output_dtype_name(dtype_name: str) -> str:
    if dtype_name == "float16":
        return "float16"
    if dtype_name == "bfloat16":
        # Hugging Face Datasets does not support bfloat16 feature dtype directly.
        # Persist as float32 to avoid implicit float64 upcast.
        return "float32"
    return "float32"


def parse_requested_layers(spec: str, available_layers: Iterable[int]) -> List[int]:
    available_sorted = sorted(available_layers)
    if spec.strip().lower() == "all":
        return available_sorted

    requested = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        requested.append(int(token))
    if not requested:
        raise ValueError("No layer index parsed from --layer-selection.")

    missing = [x for x in requested if x not in available_sorted]
    if missing:
        raise ValueError(
            f"Requested layers {missing} not in available layers {available_sorted}."
        )
    return sorted(set(requested))


def maybe_limit_dataset(ds: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return ds
    if max_samples <= 0:
        raise ValueError("--max-samples-per-subset must be > 0")
    return ds.select(range(min(max_samples, len(ds))))


def filter_query_rows(ds: Dataset) -> Dataset:
    return ds.filter(
        lambda x: isinstance(x["query"], str) and x["query"].strip() != "",
        desc="Filtering out rows with empty/null query",
    )


def to_rowwise_lists(layer_tensor: torch.Tensor, dtype: torch.dtype) -> List[List[float]]:
    return layer_tensor.to(dtype=dtype).cpu().tolist()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    subsets = resolve_subsets(args.subsets)
    device = resolve_device(args.device)
    save_dtype = output_dtype(args.dtype)
    save_dtype_name = output_dtype_name(args.dtype)

    model_load_dtype = torch.bfloat16 if "cuda" in device else torch.float32
    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=model_load_dtype,
    )
    model = model.to(device)
    model.eval()

    print(f"Dataset: {args.dataset_name}")
    print(f"Subsets: {', '.join(subsets)}")
    print(f"Cache dir: {cache_dir}")
    print(f"Output root: {output_root}")
    print(f"Device: {device}")
    print(f"Saved dtype: {save_dtype}")

    for subset in subsets:
        print(f"\n=== Processing subset: {subset} ===")
        ds = load_dataset(
            args.dataset_name,
            subset,
            split=args.split
        )
        ds = maybe_limit_dataset(ds, args.max_samples_per_subset)
        if not args.keep_empty_query:
            ds = filter_query_rows(ds)

        total = len(ds)
        print(f"Rows to embed: {total}")
        if total == 0:
            print("No rows in subset after filtering; skipping.")
            continue

        # Probe first row to discover available layer indices.
        probe = ds.select(range(min(1, total)))
        probe_text = model.encode_text_internal_representation(
            texts=["probe query"],
            task=args.task,
            prompt_name=args.prompt_name,
            batch_size=1,
            max_length=args.max_length,
        )
        probe_image = model.encode_image_internal_representation(
            images=probe["image"],
            task=args.task,
            batch_size=1,
            max_pixels=args.max_pixels,
        )
        selected_layers = parse_requested_layers(args.layer_selection, probe_text.keys())
        print(f"Selected layers: {selected_layers}")
        if args.keep_empty_query:
            print(f"Empty-query text policy: {args.empty_query_text_policy}")

        # Force Arrow schema for embedding columns so `--dtype` is preserved on disk.
        # Without explicit features, Python float lists are often inferred as float64.
        base_features = {
            key: value
            for key, value in ds.features.items()
            if key not in {"query", "image"}
        }
        base_features["query_is_empty"] = Value("bool")
        layer_dims: Dict[int, int] = {}
        for layer_idx in selected_layers:
            text_dim = int(probe_text[layer_idx].shape[-1])
            image_dim = int(probe_image[layer_idx].shape[-1])
            layer_dims[layer_idx] = text_dim
            base_features[f"text_layer_{layer_idx:02d}"] = HFSequence(
                Value(save_dtype_name), length=text_dim
            )
            base_features[f"image_layer_{layer_idx:02d}"] = HFSequence(
                Value(save_dtype_name), length=image_dim
            )
        output_features = Features(base_features)

        def encode_batch(batch: Dict[str, List]) -> Dict[str, List[List[float]]]:
            raw_queries = batch["query"]
            query_is_empty = [
                not isinstance(q, str) or q.strip() == ""
                for q in raw_queries
            ]
            non_empty_queries = [
                q for q, is_empty in zip(raw_queries, query_is_empty) if not is_empty
            ]
            images = batch["image"]
            image_layers = model.encode_image_internal_representation(
                images=images,
                task=args.task,
                batch_size=args.encode_batch_size,
                max_pixels=args.max_pixels,
            )
            image_rows_by_layer = {
                layer_idx: to_rowwise_lists(image_layers[layer_idx], save_dtype)
                for layer_idx in selected_layers
            }

            # Text side: either encode all rows (including empties as ""), or skip empties.
            if args.keep_empty_query and args.empty_query_text_policy == "zero-vector":
                if non_empty_queries:
                    encoded_non_empty = model.encode_text_internal_representation(
                        texts=non_empty_queries,
                        task=args.task,
                        prompt_name=args.prompt_name,
                        batch_size=args.encode_batch_size,
                        max_length=args.max_length,
                    )
                    encoded_non_empty = {
                        layer_idx: to_rowwise_lists(encoded_non_empty[layer_idx], save_dtype)
                        for layer_idx in selected_layers
                    }
                else:
                    encoded_non_empty = {layer_idx: [] for layer_idx in selected_layers}

                text_rows_by_layer: Dict[int, List[List[float]]] = {
                    layer_idx: [] for layer_idx in selected_layers
                }
                non_empty_ptr = 0
                for is_empty in query_is_empty:
                    for layer_idx in selected_layers:
                        if is_empty:
                            text_rows_by_layer[layer_idx].append(
                                [0.0] * layer_dims[layer_idx]
                            )
                        else:
                            text_rows_by_layer[layer_idx].append(
                                encoded_non_empty[layer_idx][non_empty_ptr]
                            )
                    if not is_empty:
                        non_empty_ptr += 1
            else:
                safe_queries = [
                    q if isinstance(q, str) and q.strip() else ""
                    for q in raw_queries
                ]
                text_layers = model.encode_text_internal_representation(
                    texts=safe_queries,
                    task=args.task,
                    prompt_name=args.prompt_name,
                    batch_size=args.encode_batch_size,
                    max_length=args.max_length,
                )
                text_rows_by_layer = {
                    layer_idx: to_rowwise_lists(text_layers[layer_idx], save_dtype)
                    for layer_idx in selected_layers
                }

            out: Dict[str, List] = {"query_is_empty": query_is_empty}
            for layer_idx in selected_layers:
                out[f"text_layer_{layer_idx:02d}"] = text_rows_by_layer[layer_idx]
                out[f"image_layer_{layer_idx:02d}"] = image_rows_by_layer[layer_idx]
            return out

        transformed = ds.map(
            encode_batch,
            batched=True,
            batch_size=args.encode_batch_size,
            writer_batch_size=args.writer_batch_size,
            features=output_features,
            desc=f"Encoding {subset}",
            remove_columns=["query", "image"],
        )

        subset_out = output_root / subset
        subset_out.mkdir(parents=True, exist_ok=True)
        transformed.save_to_disk(str(subset_out))
        print(f"Saved transformed Arrow dataset to: {subset_out}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
