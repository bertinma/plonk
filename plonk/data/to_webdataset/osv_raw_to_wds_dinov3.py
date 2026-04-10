"""
Convert raw OSV-5M images + CSV → WebDataset tar shards with DINOv3 embeddings.

Usage (single GPU):
    poetry run python plonk/data/to_webdataset/osv_raw_to_wds_dinov3.py \
        --images_dir /mnt/.../osv5m/images/train \
        --csv /mnt/.../osv5m/train.csv \
        --dest /mnt/.../osv5m/wds/train

Usage (multiple GPUs in parallel, e.g. 4 jobs):
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i poetry run python plonk/data/to_webdataset/osv_raw_to_wds_dinov3.py \
            --images_dir /mnt/.../osv5m/images/train \
            --csv /mnt/.../osv5m/train.csv \
            --dest /mnt/.../osv5m/wds/train \
            --num_jobs 4 --job_id $i &
    done
    wait
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EMBEDDING_KEY = "dinov3_vitl16.npy"
NUM_SAMPLES_PER_TAR = 10000


def build_image_index(images_dir: Path) -> dict:
    """Scan all subdirectories once and return {image_id: full_path}."""
    print(f"Scanning {images_dir} ...")
    index = {p.stem: p for p in images_dir.rglob("*.jpg")}
    print(f"Found {len(index)} images")
    return index


def flush_batch(batch_images, batch_rows, model, processor, device, sink, image_index):
    if not batch_images:
        return 0
    inputs = processor(images=batch_images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output.cpu().numpy()

    written = 0
    for emb, row in zip(embeddings, batch_rows):
        img_id = str(row["id"])
        img_path = image_index.get(img_id)
        if img_path is None:
            continue
        with open(img_path, "rb") as f:
            jpg_bytes = f.read()
        sink.write(
            {
                "__key__": img_id,
                "jpg": jpg_bytes,
                EMBEDDING_KEY: emb.astype(np.float32),
                "json": json.dumps(row.to_dict()),
            }
        )
        written += 1
    return written


def main(args):
    dest = Path(args.dest)
    dest.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.csv)
    df = df[df["latitude"].notna() & df["longitude"].notna()].reset_index(drop=True)
    df["id"] = df["id"].astype(str)

    total = len(df)
    chunk_size = math.ceil(total / args.num_jobs)
    start = args.job_id * chunk_size
    end = min(start + chunk_size, total)
    df = df.iloc[start:end].reset_index(drop=True)
    print(f"Job {args.job_id}/{args.num_jobs}: {len(df)}/{total} samples (rows {start}-{end})")

    image_index = build_image_index(Path(args.images_dir))

    print(f"Loading {MODEL_NAME} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval().to(device)
    print(f"Model on {device}")

    # Non-overlapping shard range: each job gets a reserved block of shard numbers
    max_shards_per_job = math.ceil(chunk_size / NUM_SAMPLES_PER_TAR) + 1
    start_shard = args.job_id * max_shards_per_job

    batch_images = []
    batch_rows = []

    with wds.ShardWriter(
        str(dest / "%05d.tar"),
        maxcount=NUM_SAMPLES_PER_TAR,
        start_shard=start_shard,
    ) as sink:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_id = str(row["id"])
            if img_id not in image_index:
                continue
            try:
                img = Image.open(image_index[img_id]).convert("RGB")
            except Exception:
                continue

            batch_images.append(img)
            batch_rows.append(row)

            if len(batch_images) >= args.batch_size:
                flush_batch(batch_images, batch_rows, model, processor, device, sink, image_index)
                batch_images.clear()
                batch_rows.clear()

        # Remaining samples
        flush_batch(batch_images, batch_rows, model, processor, device, sink, image_index)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to images/train (or val/test)")
    parser.add_argument("--csv", required=True, help="Path to train.csv (or test.csv)")
    parser.add_argument("--dest", required=True, help="Output directory for tar shards")
    parser.add_argument("--num_jobs", type=int, default=1, help="Total number of parallel jobs")
    parser.add_argument("--job_id", type=int, default=0, help="Index of this job (0-indexed)")
    parser.add_argument("--batch_size", type=int, default=64, help="GPU batch size")
    args = parser.parse_args()
    main(args)
