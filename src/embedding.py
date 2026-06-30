"""
embedding.py

Pipeline:
1. Load candidate_text.parquet
2. Generate embeddings
3. Save candidate_embeddings.npy

Author: Prateek
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CANDIDATE_TEXT,
    EMBEDDINGS,
    MODEL_NAME,
    BATCH_SIZE,
)


# -----------------------------------------------------
# Load Embedding Model
# -----------------------------------------------------

def load_model():
    """
    Load SentenceTransformer model.
    Automatically selects GPU if available.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    return model


# -----------------------------------------------------
# Generate Embeddings
# -----------------------------------------------------

def generate_embeddings():

    print("=" * 60)
    print("Generating Candidate Embeddings")
    print("=" * 60)

    if Path(EMBEDDINGS).exists():
        print("✓ candidate_embeddings.npy already exists.")
        print("Skipping embedding generation.\n")
        return

    print("Loading candidate profiles...")

    df = pd.read_parquet(
        CANDIDATE_TEXT,
        engine="pyarrow"
    )

    texts = (
        df["candidate_profile"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(f"Total Candidates : {len(texts):,}")

    model = load_model()

    embeddings = []

    print("\nGenerating embeddings...\n")

    for i in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Embedding"
    ):

        batch = texts[i:i + BATCH_SIZE]

        batch_embeddings = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings.append(batch_embeddings)

    candidate_embeddings = np.vstack(
        embeddings
    ).astype(np.float32)

    np.save(
        EMBEDDINGS,
        candidate_embeddings
    )

    print("\nEmbedding generation completed.")
    print(f"Shape : {candidate_embeddings.shape}")
    print(f"Saved : {EMBEDDINGS}")
    print()


# -----------------------------------------------------
# Run Independently
# -----------------------------------------------------

if __name__ == "__main__":
    generate_embeddings()