"""
preprocessing.py

Pipeline:
1. Read candidates.jsonl
2. Convert to candidates.parquet
3. Load candidates.parquet
4. Extract profile fields
5. Create candidate_profile
6. Save candidate_text.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RAW,
    PROCESSED,
    CANDIDATE_FILE,
    CANDIDATE_TEXT,
)

# -----------------------------------------------------
# Step 1 : Convert JSONL → Parquet
# -----------------------------------------------------

def convert_jsonl_to_parquet():
    """
    Converts the raw JSONL dataset into Parquet.
    """

    jsonl_path = RAW / "candidates.jsonl"

    if CANDIDATE_FILE.exists():
        print("✓ candidates.parquet already exists.")
        return

    print("Loading candidates.jsonl...")

    df = pd.read_json(
        jsonl_path,
        lines=True
    )

    print(f"Loaded {len(df):,} candidates.")

    PROCESSED.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        CANDIDATE_FILE,
        engine="pyarrow",
        index=False
    )

    print("✓ candidates.parquet created.\n")


# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def safe_extract(series, key):
    """
    Safely extract a key from a dictionary column.
    """
    return series.apply(
        lambda x: x.get(key) if isinstance(x, dict) else None
    )


def format_skills(skills):
    """
    Convert skill dictionaries into a comma separated string.
    """

    if not isinstance(skills, (list, np.ndarray)):
        return ""

    names = []

    for skill in skills:
        if isinstance(skill, dict):
            name = skill.get("name")
            if name:
                names.append(name)

    return ", ".join(names)


def format_career_history(history):
    """
    Convert career history into readable text.
    """

    if not isinstance(history, (list, np.ndarray)):
        return ""

    lines = []

    lines.append("Career History")

    for job in history:

        if not isinstance(job, dict):
            continue

        title = job.get("title", "")
        company = job.get("company", "")
        description = job.get("description", "")

        if title or company:
            lines.append(f"\n{title} at {company}")

        if description:

            description = description.replace("\n", " ").strip()

            if "." in description and len(description) > 60:

                sentences = [
                    s.strip()
                    for s in description.split(".")
                    if s.strip()
                ]

                for sentence in sentences:
                    lines.append(f"- {sentence}")

            else:
                lines.append(f"- {description}")

    return "\n".join(lines)


def create_candidate_profile(row):
    """
    Create a single candidate profile string.
    """

    sections = []

    headline = row["headline"]

    if headline:
        sections.append(
            f"Headline:\n{headline}"
        )

    experience = row["years_of_experience"]

    if experience is not None:
        sections.append(
            f"Experience:\n{experience} years"
        )

    career = format_career_history(
        row["career_history"]
    )

    if career:
        sections.append(career)

    summary = row["summary"]

    if summary:

        summary = summary.replace(
            "\n",
            " "
        ).strip()

        sections.append(
            f"Professional Summary:\n{summary}"
        )

    skills = row["skills_text"]

    if skills:
        sections.append(
            f"Skills:\n{skills}"
        )

    return "\n\n".join(sections)


# ----------------------------------------------------------
# Main Function
# ----------------------------------------------------------

def preprocess():

    print("=" * 60)
    print("Candidate Preprocessing")
    print("=" * 60)

    # Convert JSONL → Parquet if needed
    convert_jsonl_to_parquet()

    # Skip if already processed
    if Path(CANDIDATE_TEXT).exists():

        print("✓ candidate_text.parquet already exists.")
        print("Skipping preprocessing.\n")

        return

    print("Loading raw dataset...")

    df = pd.read_parquet(
        CANDIDATE_FILE,
        engine="pyarrow"
    )

    print(f"Loaded {len(df):,} candidates.\n")

    # -----------------------------------
    # Extract profile fields
    # -----------------------------------

    df["headline"] = safe_extract(
        df["profile"],
        "headline"
    )

    df["summary"] = safe_extract(
        df["profile"],
        "summary"
    )

    df["years_of_experience"] = safe_extract(
        df["profile"],
        "years_of_experience"
    )

    # -----------------------------------
    # Format skills
    # -----------------------------------

    print("Formatting skills...")

    df["skills_text"] = df["skills"].apply(
        format_skills
    )

    # -----------------------------------
    # Build profile
    # -----------------------------------

    print("Building candidate profiles...")

    df["candidate_profile"] = df.apply(
        create_candidate_profile,
        axis=1
    )

    # -----------------------------------
    # Save output
    # -----------------------------------

    output = df[
        [
            "candidate_id",
            "candidate_profile",
        ]
    ]

    output.to_parquet(
        CANDIDATE_TEXT,
        engine="pyarrow",
        index=False
    )

    print()
    print("=" * 60)
    print("Preprocessing Complete")
    print("=" * 60)
    print(f"Saved to:\n{CANDIDATE_TEXT}")
    print(f"Total Candidates : {len(output):,}")
    print()


# ----------------------------------------------------------
# Run Independently
# ----------------------------------------------------------

if __name__ == "__main__":
    preprocess()