"""
faiss_index.py

Pipeline:
1. Load candidate_embeddings.npy
2. Build FAISS Index
3. Save candidate_index.faiss

Author: Prateek
"""

from pathlib import Path

import sys
import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    EMBEDDINGS,
    FAISS_INDEX,
)


# -----------------------------------------------------
# Build FAISS Index
# -----------------------------------------------------

def build_faiss_index():

    print("=" * 60)
    print("Building FAISS Index")
    print("=" * 60)

    # Skip if already created
    if Path(FAISS_INDEX).exists():
        print("✓ candidate_index.faiss already exists.")
        print("Skipping index creation.\n")
        return

    print("Loading candidate embeddings...")

    candidate_embeddings = np.load(
        EMBEDDINGS
    ).astype(np.float32)

    print(f"Embedding Shape : {candidate_embeddings.shape}")

    dimension = candidate_embeddings.shape[1]

    print(f"Embedding Dimension : {dimension}")

    # Inner Product Index
    # Since embeddings are normalized,
    # Inner Product == Cosine Similarity
    index = faiss.IndexFlatIP(dimension)

    print("Adding embeddings to FAISS...")

    index.add(candidate_embeddings)

    print(f"Indexed Candidates : {index.ntotal:,}")

    faiss.write_index(
        index,
        str(FAISS_INDEX)
    )

    print("\nFAISS index created successfully.")
    print(f"Saved : {FAISS_INDEX}\n")


# -----------------------------------------------------
# Load Existing Index
# -----------------------------------------------------

def load_faiss_index():

    if not Path(FAISS_INDEX).exists():
        raise FileNotFoundError(
            "candidate_index.faiss not found.\n"
            "Run faiss_index.py first."
        )

    return faiss.read_index(
        str(FAISS_INDEX)
    )


# -----------------------------------------------------
# Run Independently
# -----------------------------------------------------

if __name__ == "__main__":
    build_faiss_index()