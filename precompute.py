#!/usr/bin/env python3
"""
precompute.py — One-time pre-computation of embeddings and FAISS index.

This script generates:
  - candidates_with_profiles.parquet (preprocessed candidate profiles)
  - candidate_embeddings.npy (100K x 384)
  - candidate_index.faiss (FAISS inner-product index)
  - jd_embedding.npy (JD embedding)

Usage:
    python precompute.py --candidates ./data/processed/candidates.parquet

This step takes ~100-120 minutes on CPU for 100K candidates.
It only needs to be run once. Subsequent runs skip if outputs exist.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def build_profile_text(row: dict) -> str:
    """Build a text representation of a candidate for embedding."""
    parts = []

    profile = row.get("profile", {})
    if not isinstance(profile, dict):
        return ""

    headline = profile.get("headline", "") or ""
    if headline:
        parts.append(headline)

    years = profile.get("years_of_experience")
    if years is not None:
        parts.append(f"Experience: {years} years")

    career = row.get("career_history", [])
    if isinstance(career, np.ndarray):
        career = career.tolist()
    if isinstance(career, list):
        for job in career[:3]:
            if isinstance(job, dict):
                title = job.get("title", "")
                company = job.get("company", "")
                desc = (job.get("description", "") or "")[:200]
                if title or company:
                    parts.append(f"{title} at {company}: {desc}")

    summary = profile.get("summary", "") or ""
    if summary:
        parts.append(summary[:300])

    skills = row.get("skills", [])
    if isinstance(skills, (list, np.ndarray)):
        skill_names = [
            s.get("name", "")
            for s in skills
            if isinstance(s, dict) and s.get("name")
        ]
        if skill_names:
            parts.append(f"Skills: {', '.join(skill_names[:15])}")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings and FAISS index")
    parser.add_argument("--candidates", required=True, help="Path to candidates.parquet or candidates.jsonl")
    args = parser.parse_args()

    start = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load and preprocess candidates
    # ------------------------------------------------------------------
    log.info("Step 1: Loading candidates...")
    candidates_path = Path(args.candidates)
    if candidates_path.suffix == ".parquet":
        df = pd.read_parquet(candidates_path)
    else:
        df = pd.read_json(str(candidates_path), lines=True)
    log.info(f"Loaded {len(df):,} candidates")

    profile_path = PROCESSED_DIR / "candidates_with_profiles.parquet"
    if profile_path.exists():
        log.info("Loading pre-computed profiles...")
        profiles_df = pd.read_parquet(profile_path)
    else:
        log.info("Building profile texts...")
        df["profile_text"] = df.apply(lambda r: build_profile_text(r.to_dict()), axis=1)
        profiles_df = df[["candidate_id", "profile_text"]].copy()
        profiles_df.to_parquet(profile_path, index=False)
        log.info(f"Saved {profile_path}")

    # ------------------------------------------------------------------
    # Step 2: Load model
    # ------------------------------------------------------------------
    log.info("Step 2: Loading sentence-transformer model...")
    model_name = "BAAI/bge-small-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    model = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------
    # Step 3: Generate candidate embeddings
    # ------------------------------------------------------------------
    embeddings_path = PROCESSED_DIR / "candidate_embeddings.npy"
    if embeddings_path.exists():
        log.info(f"Embeddings already exist at {embeddings_path}, skipping.")
    else:
        log.info(f"Step 3: Encoding {len(profiles_df):,} candidates...")
        texts = profiles_df["profile_text"].fillna("").astype(str).tolist()

        embeddings = []
        batch_size = 512
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            batch_emb = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings.append(batch_emb)

        candidate_embeddings = np.vstack(embeddings).astype(np.float32)
        np.save(embeddings_path, candidate_embeddings)
        log.info(f"Saved {embeddings_path} shape={candidate_embeddings.shape}")

    # ------------------------------------------------------------------
    # Step 4: Build FAISS index
    # ------------------------------------------------------------------
    index_path = PROCESSED_DIR / "candidate_index.faiss"
    if index_path.exists():
        log.info(f"FAISS index already exists at {index_path}, skipping.")
    else:
        log.info("Step 4: Building FAISS index...")
        candidate_embeddings = np.load(embeddings_path)
        dimension = candidate_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(candidate_embeddings)
        faiss.write_index(index, str(index_path))
        log.info(f"Saved {index_path} with {index.ntotal:,} vectors")

    # ------------------------------------------------------------------
    # Step 5: Generate JD embedding
    # ------------------------------------------------------------------
    jd_embedding_path = PROCESSED_DIR / "jd_embedding.npy"
    if jd_embedding_path.exists():
        log.info(f"JD embedding already exists at {jd_embedding_path}, skipping.")
    else:
        log.info("Step 5: Generating JD embedding...")
        import docx
        jd_path = Path("docs/job_description.docx")
        if not jd_path.exists():
            log.error(f"Job description not found: {jd_path}")
            sys.exit(1)

        jd_doc = docx.Document(str(jd_path))
        jd_text = "\n".join(p.text for p in jd_doc.paragraphs)

        jd_embedding = model.encode(
            jd_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        np.save(jd_embedding_path, jd_embedding)
        log.info(f"Saved {jd_embedding_path}")

    elapsed = time.time() - start
    log.info(f"Pre-computation complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info("Now run: python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv")


if __name__ == "__main__":
    main()
