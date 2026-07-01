#!/usr/bin/env python3
"""
app.py — Gradio interface for HuggingFace Spaces sandbox.

Runs the Redrob ranking pipeline on 100 sample candidates.
"""

import subprocess
import sys
from pathlib import Path

import gradio as gr


def run_ranking():
    """Execute rank.py and return the output CSV + logs."""
    cmd = [
        sys.executable, "rank.py",
        "--candidates", "./candidates.parquet",
        "--out", "./submission.csv",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    csv_path = Path("submission.csv")
    if csv_path.exists():
        csv_content = csv_path.read_text(encoding="utf-8")
        return csv_content, result.stdout + result.stderr
    else:
        return "", f"Error: submission.csv not created.\n\n{result.stderr}"


demo = gr.Interface(
    fn=run_ranking,
    inputs=[],
    outputs=[
        gr.Textbox(label="Output CSV (first 2000 chars)", lines=20),
        gr.Textbox(label="Pipeline Logs", lines=15),
    ],
    title="Redrob Candidate Ranker",
    description=(
        "Runs the ranking pipeline on 100 sample candidates. "
        "Produces a top-100 ranked CSV with scores and reasoning."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
