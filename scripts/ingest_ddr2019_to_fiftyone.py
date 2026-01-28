#!/usr/bin/env python3
"""Ingest DDR2019 from data/processed/ddr2019/ into FiftyOne (remote MongoDB).

Data layout: project root/data/processed/ddr2019/images/, labels.csv.

On host (for notebook / host App):
  uv run python scripts/ingest_ddr2019_to_fiftyone.py

In Docker (for Docker App to show images; filepaths must be under /workspace):
  ./docker/ingest-fiftyone.sh
  (sets FIFTYONE_MEDIA_PREFIX=/workspace and runs this script in the fiftyone container)

Requires: .env with FIFTYONE_DATABASE_URI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root (for .env and chdir)
_path = Path(__file__).resolve().parent.parent
if not (_path / "pyproject.toml").exists():
    _path = Path.cwd()
PROJECT_ROOT = _path
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
uri = os.environ.get("FIFTYONE_DATABASE_URI")
if not uri:
    print("FIFTYONE_DATABASE_URI not set in .env", file=sys.stderr)
    sys.exit(1)

import fiftyone as fo

fo.config.database_uri = uri

# Use FIFTYONE_MEDIA_PREFIX when ingesting inside Docker so filepaths are under /workspace.
_base = Path(os.environ.get("FIFTYONE_MEDIA_PREFIX", PROJECT_ROOT))
DATA_DIR = _base / "data" / "processed" / "ddr2019"
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"
SUBSET = 500  # limit; set FIFTYONE_DDR2019_SUBSET=0 for no limit (load all)

if not LABELS_CSV.exists():
    print(f"{LABELS_CSV} not found. Run: uv run preprocess-dataset ddr2019", file=sys.stderr)
    sys.exit(1)
if not IMAGES_DIR.exists():
    print(f"{IMAGES_DIR} not found. Run: uv run preprocess-dataset ddr2019", file=sys.stderr)
    sys.exit(1)

raw = os.environ.get("FIFTYONE_DDR2019_SUBSET")
n = int(raw) if raw else SUBSET

import pandas as pd

df = pd.read_csv(LABELS_CSV)
if n and n > 0:
    df = df.head(n)

samples = []
for _, row in df.iterrows():
    fp = IMAGES_DIR / row["filename"]
    if not fp.exists():
        continue
    s = fo.Sample(filepath=str(fp))
    s["ground_truth"] = fo.Classification(label=str(int(row["label"])))
    samples.append(s)

NAME = "ddr2019"
if fo.dataset_exists(NAME):
    ds = fo.load_dataset(NAME)
    ds.delete_samples(ds.values("id"))
    ds.add_samples(samples)
    print(f"Reloaded {NAME} with {len(samples)} samples")
else:
    ds = fo.Dataset(NAME, overwrite=True)
    ds.add_samples(samples)
    ds.persistent = True
    ds.save()
    print(f"Created {NAME} with {len(samples)} samples")
