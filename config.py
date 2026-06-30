from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed"
DOCS = BASE_DIR / "docs"
OUTPUT = BASE_DIR / "data" / "output" # not used right now

CANDIDATE_FILE = PROCESSED / "candidates.parquet"
JD_FILE = DOCS / "job_description.docx"

CANDIDATE_TEXT = PROCESSED / "candidate_text.parquet"
EMBEDDINGS = PROCESSED / "candidate_embeddings.npy"
FAISS_INDEX = PROCESSED / "candidate_index.faiss"
JD_EMBEDDING = PROCESSED / "jd_embedding.npy"
TOP2000 = PROCESSED / "top2000_candidates.parquet"

FINAL_RESULT = OUTPUT / "final_top100.csv"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 2000
BATCH_SIZE = 512