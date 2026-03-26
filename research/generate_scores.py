"""
Generate GPTZero slop scores for all draft sections and save as JSON
for the site to consume at build time.

Usage:
    python3 research/generate_scores.py

Outputs:
    site/src/data/scores.json
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).parent.parent / ".env")

WRITING_DIR = Path(__file__).parent.parent / "writing" / "drafts"
OUTPUT_PATH = Path(__file__).parent.parent / "site" / "src" / "data" / "scores.json"


def strip_markdown_heading(text: str) -> str:
    """Remove markdown headings and status comments from draft text."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def get_draft_files() -> list[Path]:
    """Get draft files to analyze."""
    files = sorted(WRITING_DIR.glob("*.md"))
    return [
        f for f in files
        if not f.name.startswith("00_") and not f.name.startswith("three_axis")
    ]


def score_document(text: str) -> dict:
    """Send text to GPTZero and return the full document result."""
    api_key = os.environ.get("GPTZERO_API_KEY")
    if not api_key:
        raise RuntimeError("GPTZERO_API_KEY not set")

    resp = requests.post(
        "https://api.gptzero.me/v2/predict/text",
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={"document": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    files = get_draft_files()
    all_scores: dict[str, dict] = {}

    for path in files:
        text = strip_markdown_heading(path.read_text())
        if not text.strip():
            continue

        print(f"Scoring {path.name}...")
        result = score_document(text)
        doc = result["documents"][0]

        sentences = []
        for s in doc.get("sentences", []):
            sentences.append({
                "text": s["sentence"],
                "score": round(s["generated_prob"], 3),
            })

        all_scores[path.stem] = {
            "overall": round(doc.get("completely_generated_prob", 0), 3),
            "class": doc.get("predicted_class", "unknown"),
            "sentences": sentences,
        }

        time.sleep(0.5)  # rate limit courtesy

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_scores, indent=2))
    print(f"\nWrote scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
