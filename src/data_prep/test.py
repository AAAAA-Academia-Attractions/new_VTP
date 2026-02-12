import argparse
from datasets import load_from_disk
from pathlib import Path
import pyarrow as pa
import torch

from datasets import load_from_disk

ds = load_from_disk("/root/autodl-tmp/hf/processed/vdr_multilingual_train_internal_arrow/en")
print(ds.column_names)
print(len(ds))
# for i in range(len(ds)):
#     print(ds[i]["query_is_empty"])

# parser = argparse.ArgumentParser(description="Inspect Arrow file and estimate per-column sizes.")
# parser.add_argument(
#     "--dataset-dir",
#     default="/root/autodl-tmp/hf/processed/vdr_multilingual_train_internal_arrow/en",
#     help="HF dataset directory (load_from_disk path).",
# )
# parser.add_argument(
#     "--arrow-path",
#     default="/root/autodl-tmp/hf/processed/vdr_multilingual_train_internal_arrow/en/data-00000-of-00001.arrow",
#     help="Path to the Arrow file to inspect.",
# )
# args = parser.parse_args()

# ds = load_from_disk(args.dataset_dir)

# arrow_path = Path(args.arrow_path)

# def human_bytes(n: int) -> str:
#     units = ["B", "KB", "MB", "GB", "TB"]
#     x = float(n)
#     for u in units:
#         if x < 1024 or u == units[-1]:
#             return f"{x:.2f} {u}"
#         x /= 1024

# file_size = arrow_path.stat().st_size

# with pa.memory_map(str(arrow_path), "r") as src:
#     try:
#         reader = pa.ipc.open_file(src)   # random-access file format
#         is_file_reader = True
#     except pa.ArrowInvalid:
#         reader = pa.ipc.open_stream(src) # stream format fallback
#         is_file_reader = False

#     num_rows = 0
#     if is_file_reader:
#         for i in range(reader.num_record_batches):
#             num_rows += reader.get_batch(i).num_rows
#     else:
#         for batch in reader:
#             num_rows += batch.num_rows

# avg = file_size / num_rows if num_rows else 0

# print(f"File: {arrow_path}")
# print(f"Total size on SSD: {file_size} bytes ({human_bytes(file_size)})")
# print(f"Rows: {num_rows}")
# print(f"Average per row (on-disk estimate): {avg:.2f} bytes ({human_bytes(int(avg))})")

# # Column size estimate (in-memory Arrow buffers; useful relative indicator)
# table = ds.data.table
# col_sizes = []
# for name in table.column_names:
#     arr = table[name]
#     size_bytes = arr.nbytes
#     col_sizes.append((name, size_bytes))

# total_col_bytes = sum(size for _, size in col_sizes)
# scale = (file_size / total_col_bytes) if total_col_bytes else 0.0

# print("\nPer-column size estimate:")
# print("(nbytes + scaled-to-file-size estimate)")
# for name, size in sorted(col_sizes, key=lambda x: x[1], reverse=True):
#     scaled = int(size * scale)
#     per_row = (scaled / num_rows) if num_rows else 0.0
#     pct = (scaled / file_size * 100) if file_size else 0.0
#     print(
#         f"{name:20s} "
#         f"raw={size:10d} ({human_bytes(size):>8s}) | "
#         f"disk~={scaled:10d} ({human_bytes(scaled):>8s}) | "
#         f"row~={per_row:9.1f} B | "
#         f"{pct:5.2f}%"
#     )