#!/usr/bin/env python3
"""
rank.py — Fast ranking step using pre-computed embeddings and FAISS index.

Usage:
    python rank.py --candidates ./data/processed/candidates.parquet --out ./submission.csv

Prerequisites:
    Run precompute.py first to generate embeddings and FAISS index:
    python precompute.py --candidates ./data/processed/candidates.parquet

The ranking step (this script) completes in <30 seconds on CPU with 16GB RAM.
"""

import argparse
import csv
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# JD Context (for reasoning generation)
# ---------------------------------------------------------------------------

JD_CORE_SKILLS = {
    "python", "embeddings", "vector search", "information retrieval",
    "learning to rank", "faiss", "pinecone", "milvus", "qdrant",
    "opensearch", "elasticsearch", "sentence transformers", "rag", "llms",
    "lora", "qlora", "peft", "hugging face transformers", "mlops",
    "bm25", "hybrid search", "dense retrieval", "reranking",
}

PRODUCT_COMPANIES = {
    "Google", "Meta", "Uber", "PhonePe", "Flipkart", "Observe.AI",
    "Krutrim", "Apple", "Amazon", "Microsoft", "Rephrase.ai",
    "Sarvam AI", "Netflix", "Salesforce", "Zomato", "Ola", "Meesho",
    "Razorpay", "Dream11", "Unacademy", "Freshworks", "Yellow.ai",
    "Haptik", "Vedantu", "PolicyBazaar", "Nykaa", "Paytm", "Swiggy",
    "CRED", "Groww", "Zepto", "PharmEasy", "Practo", "Lenskart",
    "Mamaearth", "Chargebee", "Postman", "Hasura", "Zerodha",
    "Stripe", "Shopify", "Slack", "Notion", "Figma", "Canva",
    "Spotify", "Airbnb", "Snowflake", "Databricks", "MongoDB",
    "Elastic", "Confluent", "HashiCorp", "Grafana Labs", "Vercel",
    "Cloudflare", "Retool", "Airtable", "Zapier", "Twilio",
    "Plaid", "Brex", "Ramp", "Mercury", "Scale AI", "Harvey AI",
    "Anthropic", "Cohere", "Stability AI", "Mistral AI", "Hugging Face",
    "Weights & Biases", "OpenAI", "Inflection AI", "Adept AI",
    "C3.ai", "DataRobot", "H2O.ai", "Anyscale", "Modal",
    "Pinecone", "Weaviate", "Qdrant", "Chroma", "Milvus", "Zilliz",
    "Jina AI", "Jasper", "Synthesia", "Descript", "Runway",
}

SERVICE_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "Tech Mahindra", "HCL", "LTIMindtree", "Mphasis", "Hexaware",
    "Birlasoft", "KPIT", "Tata Elxsi", "Persistent Systems",
    "Cyient", "Zensar", "Datamatics", "ValueLabs",
    "IBM Consulting", "Deloitte", "PwC", "EY", "KPMG",
}

RETRIEVAL_KEYWORDS = [
    "retrieval", "search", "ranking", "ranker", "recommendation",
    "recommendation system", "matching", "semantic search", "vector search",
    "dense retrieval", "hybrid search", "embedding", "embeddings", "faiss",
    "pinecone", "milvus", "weaviate", "qdrant", "elasticsearch",
    "opensearch", "bm25", "reranking", "ndcg", "mrr",
    "information retrieval", "learning to rank", "recall", "precision",
    "ann search", "approximate nearest neighbor", "inverted index",
]

PRODUCTION_KEYWORDS = {
    "production": 3, "productionized": 3, "deployed": 3,
    "deployment": 3, "real users": 3, "customer-facing": 3,
    "live system": 3, "production environment": 3, "shipped": 3,
    "pipeline": 2, "pipelines": 2, "serving": 2, "inference": 2,
    "monitoring": 2, "latency": 2, "throughput": 2, "scalable": 2,
    "scale": 2, "distributed": 2, "feature pipeline": 2,
    "feature store": 2, "online inference": 2, "batch inference": 2,
    "evaluation": 2, "a/b test": 2, "offline evaluation": 2,
    "ci/cd": 1, "docker": 1, "kubernetes": 1, "airflow": 1, "kafka": 1,
}

RESEARCH_KEYWORDS = [
    "research", "research scientist", "research engineer",
    "paper", "publication", "published", "conference", "academic",
    "thesis", "phd", "university", "laboratory", "lab",
    "arxiv", "icml", "nips", "neurips", "iclr", "acl", "emnlp",
]

LLM_KEYWORDS = [
    "llm", "large language model", "gpt", "bert", "transformer",
    "fine-tuning", "fine tuning", "finetuning", "fine-tuned",
    "rlhf", "dpo", "lora", "qlora", "peft", "prompt engineering",
    "rag", "retrieval augmented", "langchain", "llamaindex",
    "hugging face", "huggingface", "openai", "anthropic", "claude",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _career_history_to_list(ch: Any) -> list:
    if isinstance(ch, np.ndarray):
        return ch.tolist()
    if isinstance(ch, list):
        return ch
    return []


def _career_text(ch: Any) -> str:
    history = _career_history_to_list(ch)
    if not history:
        return ""
    return " ".join(
        job.get("description", "")
        for job in history
        if isinstance(job, dict)
    ).lower()


def _skill_names(skills: Any) -> set:
    if not isinstance(skills, (list, np.ndarray)):
        return set()
    return {
        s.get("name", "").lower()
        for s in skills
        if isinstance(s, dict) and s.get("name")
    }


# ---------------------------------------------------------------------------
# Honeypot Detection
# ---------------------------------------------------------------------------


def detect_honeypots(df: pd.DataFrame) -> pd.DataFrame:
    honeypot_flags = np.zeros(len(df), dtype=bool)

    for idx, row in df.iterrows():
        reasons = []
        profile = row.get("profile", {})
        career = row.get("career_history", [])
        skills = row.get("skills", [])

        if not isinstance(profile, dict):
            continue

        years_exp = profile.get("years_of_experience", 0) or 0

        if isinstance(skills, (list, np.ndarray)):
            num_skills = len(skills)
            if num_skills > 15 and years_exp < 2:
                reasons.append("skill_count_mismatch")
            expert_count = sum(
                1 for s in skills
                if isinstance(s, dict) and s.get("proficiency") == "expert"
            )
            if expert_count > 5 and years_exp < 3:
                reasons.append("expert_proficiency_mismatch")

        if isinstance(career, list):
            total_months = sum(
                j.get("duration_months", 0)
                for j in career if isinstance(j, dict)
            )
            if years_exp > 0 and total_months > years_exp * 12 * 1.5:
                reasons.append("career_duration_exceeds_reported")

        if reasons:
            honeypot_flags[idx] = True

    df["is_honeypot"] = honeypot_flags
    return df


# ---------------------------------------------------------------------------
# Scoring Functions
# ---------------------------------------------------------------------------


def score_experience(years: Any) -> float:
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
    elif 2 <= years < 3:
        return 0.3
    else:
        return 0.2


def score_retrieval(ch: Any) -> float:
    text = _career_text(ch)
    if not text:
        return 0.0
    hits = sum(
        1 for kw in RETRIEVAL_KEYWORDS
        if re.search(rf"\b{re.escape(kw)}\b", text)
    )
    return min(hits / 8, 1.0)


def score_production(ch: Any) -> float:
    text = _career_text(ch)
    if not text:
        return 0.0
    score = sum(
        w for kw, w in PRODUCTION_KEYWORDS.items()
        if re.search(rf"\b{re.escape(kw)}\b", text)
    )
    return min(score / 20, 1.0)


def score_company_type(ch: Any) -> float:
    history = _career_history_to_list(ch)
    if not history:
        return 0.0
    prod = sum(1 for j in history if isinstance(j, dict) and j.get("company", "") in PRODUCT_COMPANIES)
    svc = sum(1 for j in history if isinstance(j, dict) and j.get("company", "") in SERVICE_COMPANIES)
    return max(-1.0, min((prod - svc) / 5.0, 1.0))


def score_skill_match(skills: Any) -> float:
    s = _skill_names(skills)
    if not s:
        return 0.0
    return min(len(s & JD_CORE_SKILLS) / 5, 1.0)


def score_skill_assessment(signals: Any) -> float:
    if not isinstance(signals, dict):
        return 0.0
    sa = signals.get("skill_assessment_scores", {})
    if not isinstance(sa, dict) or not sa:
        return 0.0
    jd_lower = {s.lower() for s in JD_CORE_SKILLS}
    relevant = [
        v for k, v in sa.items()
        if k.lower() in jd_lower and isinstance(v, (int, float))
    ]
    if not relevant:
        return 0.0
    return sum(relevant) / len(relevant)


def score_availability(signals: Any) -> float:
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
        elif notice <= 90:
            score += 0.05
    response = signals.get("recruiter_response_rate")
    if response is not None and response >= 0:
        score += min(response, 1.0) * 0.20
    interview = signals.get("interview_completion_rate")
    if interview is not None and interview >= 0:
        score += interview * 0.15
    offer = signals.get("offer_acceptance_rate")
    if offer is not None and offer >= 0:
        score += offer * 0.10
    github = signals.get("github_activity_score")
    if github is not None and github >= 0:
        score += (github / 100) * 0.05
    completeness = signals.get("profile_completeness_score")
    if completeness is not None and completeness >= 0:
        score += (completeness / 100) * 0.05
    return min(score, 1.0)


def score_research(ch: Any) -> float:
    text = _career_text(ch)
    if not text:
        return 0.0
    hits = sum(
        1 for t in RESEARCH_KEYWORDS
        if re.search(rf"\b{re.escape(t)}\b", text)
    )
    if hits <= 2:
        return hits * 0.15
    elif hits <= 4:
        return 0.3 - (hits - 2) * 0.05
    else:
        return max(0.0, 0.2 - (hits - 4) * 0.05)


def score_llm(ch: Any, skills: Any) -> float:
    text = _career_text(ch)
    skill_set = _skill_names(skills)
    combined = text + " " + " ".join(skill_set)
    hits = sum(
        1 for kw in LLM_KEYWORDS
        if re.search(rf"\b{re.escape(kw)}\b", combined)
    )
    return min(hits / 4, 1.0)


def score_title_relevance(profile: Any) -> float:
    if not isinstance(profile, dict):
        return 0.0
    title = (profile.get("current_title") or "").lower()
    headline = (profile.get("headline") or "").lower()
    combined = title + " " + headline
    high = ["machine learning", "ml engineer", "ai engineer", "data scien",
            "nlp", "deep learning", "search engineer", "ranking",
            "recommendation", "applied scientist", "applied ml"]
    medium = ["software engineer", "backend", "data engineer", "full stack"]
    for t in high:
        if t in combined:
            return 1.0
    for t in medium:
        if t in combined:
            return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Reasoning Generator
# ---------------------------------------------------------------------------


def generate_reasoning(row: pd.Series, rank: int) -> str:
    """Generate concise, JD-aware reasoning (2-3 sentences)."""
    profile = row.get("profile", {})
    if not isinstance(profile, dict):
        return "Insufficient profile data."

    years_exp = profile.get("years_of_experience", 0) or 0
    current_company = profile.get("current_company", "") or ""
    current_title = profile.get("current_title", "") or ""

    skills = row.get("skills", [])
    skill_list = [
        s.get("name", "") for s in skills
        if isinstance(s, dict) and s.get("name")
    ] if isinstance(skills, (list, np.ndarray)) else []

    history = _career_history_to_list(row.get("career_history", []))
    career_highlights = []
    for job in history[:3]:
        if isinstance(job, dict):
            c = job.get("company", "")
            t = job.get("title", "")
            if c and t:
                career_highlights.append((t, c))

    matched_skills = set(s.lower() for s in skill_list) & JD_CORE_SKILLS
    career_text = _career_text(row.get("career_history", []))
    has_retrieval = any(kw in career_text for kw in ["retrieval", "search", "ranking", "vector", "embedding", "reranking"])
    has_production = any(kw in career_text for kw in ["production", "deployed", "shipped", "live system", "real users", "customer-facing"])

    signals = row.get("redrob_signals", {})
    open_to_work = signals.get("open_to_work_flag", False) if isinstance(signals, dict) else False
    notice_period = signals.get("notice_period_days") if isinstance(signals, dict) else None

    # --- Sentence 1: Core fit assessment ---
    # Build concise experience summary
    companies = []
    for t, c in career_highlights[:2]:
        companies.append(f"{t} at {c}")
    company_str = f", {companies[0]}" if companies else ""

    skill_names = ", ".join(sorted(matched_skills)[:3])
    skill_count = len(matched_skills)

    if rank <= 20:
        if matched_skills and has_retrieval:
            s1 = f"{years_exp}yr with retrieval/search systems{company_str}; {skill_count} JD skill matches ({skill_names})"
        elif matched_skills:
            s1 = f"{years_exp}yr, {skill_count} JD skill matches ({skill_names}), but limited retrieval background"
        else:
            s1 = f"{years_exp}yr with retrieval work{company_str}, limited formal skill alignment"
    elif rank <= 50:
        if matched_skills and has_retrieval:
            s1 = f"{years_exp}yr, {skill_count} skill matches ({skill_names}), retrieval background{company_str}"
        elif matched_skills:
            s1 = f"{years_exp}yr, {skill_count} skill matches; limited retrieval experience"
        elif has_retrieval:
            s1 = f"{years_exp}yr with retrieval/search work; weak skill alignment"
        else:
            s1 = f"{years_exp}yr; limited retrieval, ranking, or embeddings signal"
    else:
        if matched_skills and has_retrieval:
            s1 = f"{years_exp}yr, {skill_count} skill overlaps; some retrieval background"
        elif matched_skills:
            s1 = f"{years_exp}yr, {skill_count} JD skill overlaps; limited production retrieval"
        elif has_retrieval:
            s1 = f"{years_exp}yr with some retrieval exposure; few skill endorsements"
        else:
            s1 = f"{years_exp}yr; weak alignment with JD requirements"

    # --- Sentence 2: Strengths and concerns ---
    concerns = []
    strengths = []

    if has_production:
        strengths.append("shipped ML systems to production")
    else:
        concerns.append("no production deployment evidence")

    if current_company in PRODUCT_COMPANIES:
        strengths.append(f"product company ({current_company})")
    elif current_company in SERVICE_COMPANIES:
        concerns.append(f"service company background ({current_company})")

    if notice_period and notice_period > 60:
        concerns.append(f"notice period {notice_period}d")
    if rank <= 20 and not open_to_work:
        concerns.append("not open-to-work")
    if rank <= 50 and years_exp < 4:
        concerns.append(f"below preferred experience band ({years_exp}yr)")
    if rank > 50 and years_exp and years_exp < 3:
        concerns.append(f"significantly below threshold ({years_exp}yr)")

    s2_parts = []
    if strengths:
        s2_parts.append(strengths[0])
    if concerns:
        s2_parts.append(concerns[0])

    s2 = ". ".join(s2_parts) if s2_parts else ""

    # Combine
    if s2:
        return f"{s1}; {s2}."
    return f"{s1}."


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(candidates_path: str, output_path: str) -> None:
    start = time.time()

    log.info("=" * 60)
    log.info("Redrob Candidate Ranking Pipeline (Fast Mode)")
    log.info("=" * 60)

    # Resolve artifact directories: check candidates dir first, then data/processed
    candidates_path = Path(candidates_path)
    candidates_dir = candidates_path.parent.resolve()
    artifact_dirs = [candidates_dir, PROCESSED_DIR.resolve()]

    def resolve_artifact(name):
        for d in artifact_dirs:
            p = d / name
            if p.exists():
                return p
        return None

    profiles_path = resolve_artifact("candidates_with_profiles.parquet")
    index_path = resolve_artifact("candidate_index.faiss")
    jd_emb_path = resolve_artifact("jd_embedding.npy")

    missing = []
    if profiles_path is None:
        missing.append("candidates_with_profiles.parquet")
    if index_path is None:
        missing.append("candidate_index.faiss")
    if jd_emb_path is None:
        missing.append("jd_embedding.npy")

    if missing:
        log.error(f"Pre-computed artifacts missing: {missing}")
        log.error("Run: python precompute.py --candidates ./data/processed/candidates.parquet")
        sys.exit(1)

    log.info(f"Using artifacts from: {profiles_path.parent}")

    # ------------------------------------------------------------------
    # Step 1: Load candidates and profiles
    # ------------------------------------------------------------------
    log.info("Step 1: Loading candidates...")
    if candidates_path.suffix == ".parquet":
        df = pd.read_parquet(candidates_path)
    else:
        df = pd.read_json(str(candidates_path), lines=True)
    log.info(f"Loaded {len(df):,} candidates")

    profiles_df = pd.read_parquet(profiles_path)
    df = df.merge(profiles_df, on="candidate_id", how="left")

    # ------------------------------------------------------------------
    # Step 2: Honeypot detection
    # ------------------------------------------------------------------
    log.info("Step 2: Detecting honeypots...")
    df = detect_honeypots(df)
    honeypot_count = int(df["is_honeypot"].sum())
    log.info(f"Detected {honeypot_count} honeypot candidates")

    # ------------------------------------------------------------------
    # Step 3: Extract profile fields
    # ------------------------------------------------------------------
    log.info("Step 3: Extracting profile fields...")

    def safe_extract(series, key):
        return series.apply(lambda x: x.get(key) if isinstance(x, dict) else None)

    df["headline"] = safe_extract(df["profile"], "headline")
    df["years_exp"] = safe_extract(df["profile"], "years_of_experience")
    df["current_title"] = safe_extract(df["profile"], "current_title")
    df["current_company"] = safe_extract(df["profile"], "current_company")

    # ------------------------------------------------------------------
    # Step 4: FAISS retrieval using pre-computed embeddings
    # ------------------------------------------------------------------
    log.info("Step 4: FAISS retrieval...")

    jd_embedding = np.load(jd_emb_path).astype(np.float32)
    index = faiss.read_index(str(index_path))
    log.info(f"FAISS index: {index.ntotal:,} vectors")

    TOP_K = min(2000, index.ntotal)
    scores, indices = index.search(jd_embedding.reshape(1, -1), TOP_K)

    top_df = df.iloc[indices[0]].copy()
    top_df["embedding_score"] = scores[0]
    log.info(f"Retrieved top {TOP_K} candidates")

    # ------------------------------------------------------------------
    # Step 5: Multi-signal scoring
    # ------------------------------------------------------------------
    log.info("Step 5: Computing multi-signal scores...")

    top_df["experience_score"] = top_df["years_exp"].apply(score_experience)
    top_df["retrieval_score"] = top_df["career_history"].apply(score_retrieval)
    top_df["production_score"] = top_df["career_history"].apply(score_production)
    top_df["company_type_score"] = top_df["career_history"].apply(score_company_type)
    top_df["skill_match_score"] = top_df["skills"].apply(score_skill_match)
    top_df["skill_assessment_raw"] = top_df["redrob_signals"].apply(score_skill_assessment)
    top_df["availability_score"] = top_df["redrob_signals"].apply(score_availability)
    top_df["research_score"] = top_df["career_history"].apply(score_research)
    top_df["llm_score"] = top_df.apply(
        lambda r: score_llm(r.get("career_history"), r.get("skills")), axis=1
    )
    top_df["title_relevance_score"] = top_df["profile"].apply(score_title_relevance)

    max_sa = top_df["skill_assessment_raw"].max()
    top_df["skill_assessment_norm"] = (
        top_df["skill_assessment_raw"] / max_sa if max_sa > 0 else 0.0
    )

    top_df["final_score"] = (
        0.35 * top_df["embedding_score"]
        + 0.10 * top_df["experience_score"]
        + 0.10 * top_df["retrieval_score"]
        + 0.10 * top_df["production_score"]
        + 0.08 * top_df["company_type_score"]
        + 0.07 * top_df["availability_score"]
        + 0.05 * top_df["skill_match_score"]
        + 0.05 * top_df["skill_assessment_norm"]
        + 0.05 * top_df["title_relevance_score"]
        + 0.03 * top_df["llm_score"]
        + 0.02 * top_df["research_score"]
    )

    # Penalize honeypots
    top_df.loc[top_df["is_honeypot"], "final_score"] *= 0.1

    # ------------------------------------------------------------------
    # Step 6: Rank and generate reasoning
    # ------------------------------------------------------------------
    log.info("Step 6: Ranking and generating reasoning...")

    ranked = top_df.sort_values("final_score", ascending=False).head(100).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["reasoning"] = ranked.apply(lambda r: generate_reasoning(r, r["rank"]), axis=1)

    # ------------------------------------------------------------------
    # Step 7: Write CSV
    # ------------------------------------------------------------------
    log.info("Step 7: Writing submission CSV...")

    output = ranked[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    output = output.rename(columns={"final_score": "score"})

    # Ensure non-increasing scores and deterministic tie-breaking
    output = output.sort_values(["score", "candidate_id"], ascending=[False, True])
    output["rank"] = range(1, len(output) + 1)

    # Enforce non-increasing
    for i in range(1, len(output)):
        if output.iloc[i]["score"] > output.iloc[i - 1]["score"]:
            output.iloc[i, output.columns.get_loc("score")] = output.iloc[i - 1]["score"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("Ranking Complete")
    log.info("=" * 60)
    log.info(f"Output     : {output_path}")
    log.info(f"Candidates : {len(output)}")
    log.info(f"Time       : {elapsed:.1f}s")
    log.info(f"Honeypots  : {honeypot_count}")
    log.info("")
    log.info("Top 10:")
    for _, row in output.head(10).iterrows():
        log.info(f"  #{row['rank']:3d} {row['candidate_id']}  score={row['score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking Pipeline")
    parser.add_argument("--candidates", required=True, help="Path to candidates.parquet or candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)


if __name__ == "__main__":
    main()
