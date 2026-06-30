"""
retrieve.py

Pipeline:
1. Load Job Description (.docx)
2. Generate JD Embedding
3. Load FAISS Index
4. Retrieve Top-K Candidates
5. Save top2000_candidates.parquet

Author: Prateek
"""

from pathlib import Path

import sys
import numpy as np
import pandas as pd
import torch
import docx
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CANDIDATE_FILE,
    MODEL_NAME,
    CANDIDATE_TEXT,
    EMBEDDINGS,
    FAISS_INDEX,
    JD_FILE,
    JD_EMBEDDING,
    TOP2000,
    TOP_K,
)

import faiss


# -----------------------------------------------------
# Load Embedding Model
# -----------------------------------------------------

def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    return SentenceTransformer(
        MODEL_NAME,
        device=device
    )


# -----------------------------------------------------
# Read Job Description
# -----------------------------------------------------

def load_job_description():

    print("Loading Job Description...")

    document = docx.Document(JD_FILE)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

    return text


# -----------------------------------------------------
# Generate JD Embedding
# -----------------------------------------------------

def generate_jd_embedding(model, jd_text):

    if Path(JD_EMBEDDING).exists():

        print("✓ JD embedding already exists.")

        return np.load(JD_EMBEDDING)

    print("Generating JD embedding...")

    embedding = model.encode(
        jd_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embedding = embedding.astype(np.float32)

    np.save(
        JD_EMBEDDING,
        embedding
    )

    return embedding


# -----------------------------------------------------
# Retrieve Candidates
# -----------------------------------------------------

def retrieve_candidates():

    print("=" * 60)
    print("Semantic Candidate Retrieval")
    print("=" * 60)

    if Path(TOP2000).exists():

        print("✓ top2000_candidates.parquet already exists.")
        print("Skipping retrieval.\n")

        return

    model = load_model()

    jd_text = load_job_description()

    jd_embedding = generate_jd_embedding(
        model,
        jd_text
    )

    jd_embedding = jd_embedding.reshape(1, -1)

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(FAISS_INDEX)
    )

    print(f"Indexed Candidates : {index.ntotal:,}")

    print("Searching...")

    scores, indices = index.search(
        jd_embedding,
        TOP_K
    )

    print(f"Retrieved Top {TOP_K:,} candidates.")

    # Candidate profiles
    candidate_text_df = pd.read_parquet(CANDIDATE_TEXT)

    # Original dataset
    full_df = pd.read_parquet(CANDIDATE_FILE)

    # Top-K IDs and scores
    top_candidates = candidate_text_df.iloc[indices[0]].copy()

    top_candidates["embedding_score"] = scores[0]

    # Merge with original metadata
    top_candidates = top_candidates.merge(
        full_df,
        on="candidate_id",
        how="left"
    )

    top_candidates.to_parquet(
        TOP2000,
        index=False
    )

    print()
    print("=" * 60)
    print("Retrieval Completed")
    print("=" * 60)
    print(f"Saved : {TOP2000}")
    print(f"Candidates Retrieved : {len(top_candidates):,}")
    print()


# -----------------------------------------------------
# Run Independently
# -----------------------------------------------------

if __name__ == "__main__":
    retrieve_candidates()