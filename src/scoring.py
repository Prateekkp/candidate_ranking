"""
scoring.py

Pipeline:
1. Load top2000 candidates from FAISS retrieval
2. Compute individual scores (experience, retrieval, production, etc.)
3. Combine into final_score
4. Save final_top100.csv

Author: Prateek
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TOP2000, FINAL_RESULT

# -----------------------------------------------------
# Keyword Lists
# -----------------------------------------------------

RETRIEVAL_KEYWORDS = [
    "retrieval",
    "search",
    "ranking",
    "ranker",
    "recommendation",
    "recommendation system",
    "matching",
    "semantic search",
    "vector search",
    "dense retrieval",
    "hybrid search",
    "embedding",
    "embeddings",
    "faiss",
    "pinecone",
    "milvus",
    "weaviate",
    "qdrant",
    "elasticsearch",
    "opensearch",
    "bm25",
    "reranking",
    "ndcg",
    "mrr",
    "map",
    "offline evaluation",
    "a/b test",
]

PRODUCTION_WEIGHTS = {
    # Very Strong Signals
    "production": 3,
    "productionized": 3,
    "deployed": 3,
    "deployment": 3,
    "real users": 3,
    "customer-facing": 3,
    "live system": 3,
    "production environment": 3,
    # Infrastructure
    "pipeline": 2,
    "pipelines": 2,
    "serving": 2,
    "inference": 2,
    "monitoring": 2,
    "latency": 2,
    "throughput": 2,
    "scalable": 2,
    "scale": 2,
    "distributed": 2,
    # ML Operations
    "feature pipeline": 2,
    "feature store": 2,
    "online inference": 2,
    "batch inference": 2,
    "evaluation": 2,
    "a/b test": 2,
    "offline evaluation": 2,
    # Reliability
    "schema drift": 2,
    "data quality": 2,
    "alerting": 2,
    "logging": 1,
    "metrics": 1,
}

RESEARCH_TERMS = [
    "research",
    "research scientist",
    "research engineer",
    "paper",
    "publication",
    "published",
    "conference",
    "academic",
    "thesis",
    "phd",
    "university",
    "laboratory",
    "lab",
]

PRODUCT_COMPANIES = {
    "Google",
    "Meta",
    "Uber",
    "PhonePe",
    "Flipkart",
    "Observe.AI",
    "Krutrim",
    "Apple",
    "Amazon",
    "Microsoft",
    "Rephrase.ai",
    "Sarvam AI",
}

SERVICE_COMPANIES = {
    "TCS",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Capgemini",
    "Tech Mahindra",
    "HCL",
}

JD_SKILLS = [
    "Python",
    "Embeddings",
    "Vector Search",
    "Information Retrieval",
    "Learning to Rank",
    "FAISS",
    "Pinecone",
    "Milvus",
    "Qdrant",
    "OpenSearch",
    "Elasticsearch",
    "Sentence Transformers",
    "RAG",
    "LLMs",
    "LoRA",
    "QLoRA",
    "PEFT",
    "Hugging Face Transformers",
    "MLOps",
]

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------


def _career_history_to_list(career_history):
    if isinstance(career_history, np.ndarray):
        return career_history.tolist()
    return career_history


def _career_text(career_history):
    career_history = _career_history_to_list(career_history)
    if not isinstance(career_history, list) or not career_history:
        return ""
    return " ".join(
        job.get("description", "")
        for job in career_history
        if isinstance(job, dict)
    ).lower()


# -----------------------------------------------------
# Scoring Functions
# -----------------------------------------------------


def experience_score(years):
    if years is None:
        return 0.0

    years = float(years)

    if 5 <= years <= 9:
        return 1.0
    elif 4 <= years < 5:
        return 0.8
    elif 9 < years <= 11:
        return 0.8
    elif 3 <= years < 4:
        return 0.5
    elif 11 < years <= 13:
        return 0.5
    else:
        return 0.2


def retrieval_score(career_history):
    text = _career_text(career_history)
    if not text:
        return 0.0

    hits = 0
    for kw in RETRIEVAL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw.lower())}\b", text):
            hits += 1

    return min(hits / 8, 1.0)


def production_score(career_history):
    text = _career_text(career_history)
    if not text:
        return 0.0

    score = 0
    for phrase, weight in PRODUCTION_WEIGHTS.items():
        if re.search(rf"\b{re.escape(phrase.lower())}\b", text):
            score += weight

    return min(score / 20, 1.0)


def research_penalty(career_history):
    text = _career_text(career_history)
    if not text:
        return 0.0

    hits = 0
    for term in RESEARCH_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text):
            hits += 1

    return min(hits / 5, 1.0)


def company_type_score(career_history):
    career_history = _career_history_to_list(career_history)
    if not isinstance(career_history, list) or not career_history:
        return 0.0

    product_company_hits = 0
    service_company_hits = 0

    for job in career_history:
        if isinstance(job, dict) and "company" in job:
            company_name = job["company"]
            if company_name in PRODUCT_COMPANIES:
                product_company_hits += 1
            elif company_name in SERVICE_COMPANIES:
                service_company_hits += 1

    raw_score_difference = product_company_hits - service_company_hits
    normalized_score = raw_score_difference / 5.0

    return max(-1.0, min(normalized_score, 1.0))


def calculate_jd_skill_score(redrob_signals):
    if not isinstance(redrob_signals, dict) or "skill_assessment_scores" not in redrob_signals:
        return 0.0

    skill_scores = redrob_signals["skill_assessment_scores"]
    if not isinstance(skill_scores, dict):
        return 0.0

    relevant_scores = []
    for skill in JD_SKILLS:
        score = skill_scores.get(skill)
        if score is not None:
            relevant_scores.append(score)

    if not relevant_scores:
        return 0.0

    return sum(relevant_scores) / len(relevant_scores)


def availability_score(signals):
    if not isinstance(signals, dict):
        return 0.0

    score = 0.0

    if signals.get("open_to_work_flag"):
        score += 0.25

    notice = signals.get("notice_period_days")
    if notice is not None:
        if notice <= 30:
            score += 0.20
        elif notice <= 60:
            score += 0.10

    response = signals.get("recruiter_response_rate")
    if response is not None:
        score += min(response, 1.0) * 0.20

    interview = signals.get("interview_completion_rate")
    if interview is not None:
        score += interview * 0.15

    offer = signals.get("offer_acceptance_rate")
    if offer is not None:
        score += offer * 0.10

    github = signals.get("github_activity_score")
    if github is not None:
        score += (github / 100) * 0.05

    completeness = signals.get("profile_completeness_score")
    if completeness is not None:
        score += (completeness / 100) * 0.05

    return min(score, 1.0)


# -----------------------------------------------------
# Profile Builder
# -----------------------------------------------------


def create_concise_profile(row):
    headline = row["headline"]
    years_exp = row["years_exp"]
    career_history = row["career_history"]
    summary_text = row["summary"]

    career_summary_parts = []
    career_history = _career_history_to_list(career_history)

    if isinstance(career_history, list) and career_history:
        for job in career_history[:2]:
            if isinstance(job, dict) and "title" in job and "company" in job:
                career_summary_parts.append(f"{job['title']} at {job['company']}")

        career_summary = ", then ".join(career_summary_parts)
        if len(career_history) > 2:
            career_summary += "..."
    else:
        career_summary = ""

    if not career_summary_parts and summary_text:
        career_summary = f"Summary: {summary_text}"
    elif not career_summary_parts:
        career_summary = "N/A"

    return f"{headline} | Experience: {years_exp} years | Career History: {career_summary}"


# -----------------------------------------------------
# Main Pipeline
# -----------------------------------------------------


def score_candidates():
    print("=" * 60)
    print("Candidate Scoring")
    print("=" * 60)

    if FINAL_RESULT.exists():
        print(f"[SKIP] {FINAL_RESULT.name} already exists.")
        print("Skipping scoring.\n")
        return

    top2000 = pd.read_parquet(TOP2000)

    print(f"Loaded {len(top2000):,} candidates from {TOP2000.name}")

    top2000["headline"] = top2000["profile"].apply(lambda p: p.get("headline", "") if isinstance(p, dict) else "")
    top2000["summary"] = top2000["profile"].apply(lambda p: p.get("summary", "") if isinstance(p, dict) else "")
    top2000["years_exp"] = top2000["profile"].apply(lambda p: p.get("years_of_experience") if isinstance(p, dict) else None)

    top2000["experience_score"] = top2000["years_exp"].apply(experience_score)
    top2000["retrieval_score"] = top2000["career_history"].apply(retrieval_score)
    top2000["production_score"] = top2000["career_history"].apply(production_score)
    top2000["research_penalty_score"] = top2000["career_history"].apply(research_penalty)
    top2000["company_type_score"] = top2000["career_history"].apply(company_type_score)
    top2000["skill_assessment_score"] = top2000["redrob_signals"].apply(calculate_jd_skill_score)
    top2000["availability_score"] = top2000["redrob_signals"].apply(availability_score)

    max_skill_score = top2000["skill_assessment_score"].max()
    if max_skill_score > 0:
        top2000["skill_assessment_score"] = top2000["skill_assessment_score"] / max_skill_score

    top2000["final_score"] = (
        0.45 * top2000["embedding_score"]
        + 0.10 * top2000["experience_score"]
        + 0.10 * top2000["retrieval_score"]
        + 0.10 * top2000["production_score"]
        + 0.10 * top2000["company_type_score"]
        + 0.10 * top2000["availability_score"]
        + 0.05 * top2000["skill_assessment_score"]
    )

    top2000["candidate_profile"] = top2000.apply(create_concise_profile, axis=1)

    final_top_100_ranked = top2000.sort_values("final_score", ascending=False).head(100).copy()
    final_top_100_ranked["rank"] = range(1, len(final_top_100_ranked) + 1)

    final_top_100_output = final_top_100_ranked[
        ["candidate_id", "rank", "final_score", "candidate_profile"]
    ].rename(columns={"final_score": "score", "candidate_profile": "reasoning"})

    FINAL_RESULT.parent.mkdir(parents=True, exist_ok=True)
    final_top_100_output.to_csv(FINAL_RESULT, index=False)

    print()
    print("=" * 60)
    print("Scoring Completed")
    print("=" * 60)
    print(f"Saved : {FINAL_RESULT}")
    print(f"Top candidates : {len(final_top_100_output):,}")
    print()
    print(final_top_100_output.head(5).to_string(index=False))
    print(f"\nRest can be seen in CSV file at: {FINAL_RESULT}")


# -----------------------------------------------------
# Run Independently
# -----------------------------------------------------

if __name__ == "__main__":
    score_candidates()
