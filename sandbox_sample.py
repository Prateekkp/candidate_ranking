#!/usr/bin/env python3
"""
sandbox_sample.py — Generate a 100-candidate sample for HuggingFace Spaces sandbox.

Usage:
    python sandbox_sample.py

Produces:
    sandbox/_sample_candidates.parquet   (100 rows)
    sandbox/_sample_profiles.parquet     (100 rows)
    sandbox/_sample_index.faiss          (FAISS index for 100 candidates)
    sandbox/jd_embedding.npy             (copied from data/processed/)
"""

import shutil
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
SANDBOX = Path("sandbox")
N_SAMPLES = 100


def main():
    SANDBOX.mkdir(parents=True, exist_ok=True)

    # Load full data
    print("Loading data...")
    df = pd.read_parquet(PROCESSED / "candidates.parquet")
    profiles = pd.read_parquet(PROCESSED / "candidates_with_profiles.parquet")
    embeddings = np.load(PROCESSED / "candidate_embeddings.npy")

    # Take first N candidates (deterministic sample)
    sample = df.head(N_SAMPLES).copy()
    sample_ids = set(sample["candidate_id"].tolist())

    # Filter profiles to match sample
    sample_profiles = profiles[profiles["candidate_id"].isin(sample_ids)].copy()

    # Build FAISS index from sample embeddings
    # Find indices of sample candidates in the full embeddings array
    profile_indices = profiles[profiles["candidate_id"].isin(sample_ids)].index.tolist()
    sample_emb = embeddings[profile_indices].astype(np.float32)

    dim = sample_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(sample_emb)

    # Save with exact names that rank.py expects
    sample.to_parquet(SANDBOX / "candidates.parquet", index=False)
    sample_profiles.to_parquet(SANDBOX / "candidates_with_profiles.parquet", index=False)
    faiss.write_index(index, str(SANDBOX / "candidate_index.faiss"))
    shutil.copy2(PROCESSED / "jd_embedding.npy", SANDBOX / "jd_embedding.npy")

    # Summary
    for f in sorted(SANDBOX.iterdir()):
        sz = f.stat().st_size
        unit = "KB" if sz < 1048576 else "MB"
        val = sz / 1024 if unit == "KB" else sz / 1048576
        print(f"  {f.name:40s} {val:6.1f} {unit}")

    print(f"\nSample generated: {N_SAMPLES} candidates in {SANDBOX}/")
    print("Upload the entire sandbox/ folder to your HuggingFace Space.")


if __name__ == "__main__":
    main()
