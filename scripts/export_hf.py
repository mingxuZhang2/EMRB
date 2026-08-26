#!/usr/bin/env python3
"""Build the Hugging Face Hub package for EMRB.

Two representations of the same 200 problems go into one dataset repo:

  parquet/L{1..5}-00000-of-00001.parquet   tabular view, drives `load_dataset`
                                           and the dataset viewer; I/Q is split
                                           into iq_real / iq_imag float32 lists
                                           because Parquet has no complex type
  raw/L{1..5}/EMRB_*.npy, *.json           canonical files, byte-identical to
                                           data/; `evaluate.py` reads these
                                           after `snapshot_download`

The Parquet rows are lossless with respect to the JSON: `metadata_json` holds
the complete original object, so nothing that only some levels carry (L5's
`verification`, `answer_schema_version`) is dropped.

Usage:
    python scripts/export_hf.py                  # build hf_export/
    python scripts/export_hf.py --check          # build, then load it back
    python scripts/export_hf.py --push mingxuzhang/EMRB
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
LEVELS = ["L1", "L2", "L3", "L4", "L5"]

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("level", pa.string()),
        ("archetype", pa.string()),
        ("archetype_desc", pa.string()),
        ("num_questions", pa.int32()),
        ("total_points", pa.int32()),
        ("prompt", pa.string()),
        ("questions_json", pa.string()),
        ("metadata_json", pa.string()),
        ("sampling_rate_hz", pa.float64()),
        ("num_samples", pa.int32()),
        ("npy_file", pa.string()),
        ("iq_real", pa.list_(pa.float32())),
        ("iq_imag", pa.list_(pa.float32())),
    ]
)


def build_rows(level):
    rows = []
    for json_path in sorted((DATA_DIR / level).glob("EMRB_*.json")):
        meta = json.loads(json_path.read_text())
        npy_path = json_path.with_suffix(".npy")
        iq = np.load(npy_path)
        if iq.dtype != np.complex64:
            raise ValueError(f"{npy_path.name}: expected complex64, got {iq.dtype}")
        params = meta.get("generation_params", {})
        rows.append(
            {
                "sample_id": meta["sample_id"],
                "level": meta["level"],
                "archetype": meta.get("archetype", ""),
                "archetype_desc": meta.get("archetype_desc", ""),
                # L4 problem files omit num_questions; the questions list is authoritative
                "num_questions": int(meta.get("num_questions", len(meta["questions"]))),
                "total_points": int(meta["total_points"]),
                "prompt": meta["question"],
                "questions_json": json.dumps(meta["questions"], ensure_ascii=False),
                "metadata_json": json.dumps(meta, ensure_ascii=False),
                "sampling_rate_hz": float(params.get("fs", float("nan"))),
                "num_samples": int(iq.size),
                "npy_file": f"raw/{level}/{npy_path.name}",
                "iq_real": iq.real.astype(np.float32),
                "iq_imag": iq.imag.astype(np.float32),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no problems found under {DATA_DIR / level}")
    return rows


def write_parquet(out_dir, level, rows):
    columns = {name: [row[name] for row in rows] for name in SCHEMA.names}
    table = pa.table(columns, schema=SCHEMA)
    target = out_dir / "parquet" / f"{level}-00000-of-00001.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target, compression="zstd")
    return target


def copy_raw(out_dir, level):
    dest = out_dir / "raw" / level
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted((DATA_DIR / level).iterdir()):
        if src.suffix in {".npy", ".json"}:
            shutil.copy2(src, dest / src.name)
            count += 1
    return count


def write_card(out_dir, counts):
    card = (REPO_ROOT / "scripts" / "hf_dataset_card.md").read_text()
    total = sum(counts.values())
    per_level = " + ".join(f"{lv} {counts[lv]}" for lv in LEVELS)
    card = card.replace("{{TOTAL}}", str(total)).replace("{{PER_LEVEL}}", per_level)
    (out_dir / "README.md").write_text(card)


def build(out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    counts = {}
    for level in LEVELS:
        rows = build_rows(level)
        target = write_parquet(out_dir, level, rows)
        raw_n = copy_raw(out_dir, level)
        counts[level] = len(rows)
        size_mb = target.stat().st_size / 1e6
        print(f"{level}: {len(rows)} problems -> {target.name} ({size_mb:.1f} MB), {raw_n} raw files")
    write_card(out_dir, counts)
    print(f"total {sum(counts.values())} problems in {out_dir}")
    return counts


def check(out_dir):
    from datasets import load_dataset

    ds = load_dataset(str(out_dir))
    print(ds)
    for level in LEVELS:
        for row in ds[level]:
            iq = np.asarray(row["iq_real"], dtype=np.float32) + 1j * np.asarray(
                row["iq_imag"], dtype=np.float32
            )
            original = np.load(DATA_DIR / level / (row["sample_id"] + ".npy"))
            assert iq.astype(np.complex64).tobytes() == original.tobytes(), (
                f"I/Q round-trip mismatch on {row['sample_id']}"
            )
            meta = json.loads(row["metadata_json"])
            assert meta["sample_id"] == row["sample_id"]
            assert len(json.loads(row["questions_json"])) == row["num_questions"]
            assert (out_dir / row["npy_file"]).exists(), row["npy_file"]
        n = len(ds[level])
        print(f"{level}: {n}/{n} rows round-trip bit-exact against data/{level}")


def push(out_dir, repo_id, private):
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()["name"]
    print(f"authenticated as {who}, pushing to {repo_id} (private={private})")
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add EMRB: 200 raw I/Q signal-analysis problems (Parquet + canonical .npy)",
    )
    print(f"done: https://huggingface.co/datasets/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "hf_export"))
    ap.add_argument("--check", action="store_true", help="load the built package back and verify")
    ap.add_argument("--push", metavar="REPO_ID", help="upload to this Hub dataset repo")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--skip-build", action="store_true", help="reuse an existing --out directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not args.skip_build:
        build(out_dir)
    if args.check:
        check(out_dir)
    if args.push:
        push(out_dir, args.push, args.private)


if __name__ == "__main__":
    main()
