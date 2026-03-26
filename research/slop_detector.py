"""
Slop detector: runs draft sections through AI detection APIs and flags
sentences that read as AI-generated.

Usage:
    python3 research/slop_detector.py                     # all sections
    python3 research/slop_detector.py writing/drafts/02_debt.md  # one file

Requires API keys as environment variables:
    SAPLING_API_KEY   - free tier at https://sapling.ai/docs/api
    GPTZERO_API_KEY   - paid, https://gptzero.me/docs
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import requests

# Load API keys from .env at project root
load_dotenv(Path(__file__).parent.parent / ".env")

WRITING_DIR = Path(__file__).parent.parent / "writing" / "drafts"

# Color codes for terminal output
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def strip_markdown_heading(text: str) -> str:
    """Remove markdown headings and status comments from draft text."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def load_draft(path: Path) -> str:
    """Load a draft file and clean it for analysis."""
    raw = path.read_text()
    return strip_markdown_heading(raw)


def get_draft_files(specific_file: Optional[str] = None) -> list[Path]:
    """Get draft files to analyze."""
    if specific_file:
        p = Path(specific_file)
        if not p.is_absolute():
            p = Path.cwd() / p
        return [p]

    files = sorted(WRITING_DIR.glob("*.md"))
    # Skip the full first draft reference file and three-axis essay
    return [
        f for f in files
        if not f.name.startswith("00_") and not f.name.startswith("three_axis")
    ]


# ---------------------------------------------------------------------------
# Sapling (free tier)
# ---------------------------------------------------------------------------

def check_sapling(text: str) -> Optional[dict]:
    """Check text with Sapling AI detector. Returns per-sentence scores."""
    api_key = os.environ.get("SAPLING_API_KEY")
    if not api_key:
        return None

    try:
        resp = requests.post(
            "https://api.sapling.ai/api/v1/aidetect",
            json={"key": api_key, "text": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  {DIM}Sapling error: {e}{RESET}")
        return None


def format_sapling_results(result: dict) -> None:
    """Print Sapling results with per-sentence highlighting."""
    overall = result.get("score", 0)
    label = score_label(overall)
    print(f"\n  {BOLD}Sapling{RESET}  overall: {label} ({overall:.0%} AI)")

    sentences = result.get("sentence_scores", [])
    if not sentences:
        return

    flagged = [(s, score) for s, score in sentences if score > 0.5]
    if not flagged:
        print(f"  {GREEN}No sentences flagged as AI-sounding.{RESET}")
        return

    print(f"  {len(flagged)} sentence(s) flagged:\n")
    for sentence, score in sorted(flagged, key=lambda x: -x[1]):
        color = RED if score > 0.8 else YELLOW
        truncated = sentence[:120] + "..." if len(sentence) > 120 else sentence
        print(f"  {color}{score:.0%}{RESET}  {truncated}")


# ---------------------------------------------------------------------------
# GPTZero
# ---------------------------------------------------------------------------

def check_gptzero(text: str) -> Optional[dict]:
    """Check text with GPTZero. Returns per-sentence scores."""
    api_key = os.environ.get("GPTZERO_API_KEY")
    if not api_key:
        return None

    try:
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
    except requests.RequestException as e:
        print(f"  {DIM}GPTZero error: {e}{RESET}")
        return None


def format_gptzero_results(result: dict) -> None:
    """Print GPTZero results with per-sentence highlighting."""
    docs = result.get("documents", [])
    if not docs:
        return

    doc = docs[0]
    overall = doc.get("completely_generated_prob", 0)
    label = score_label(overall)
    cls = doc.get("predicted_class", "unknown")
    print(f"\n  {BOLD}GPTZero{RESET}  overall: {label} ({overall:.0%} AI, class: {cls})")

    sentences = doc.get("sentences", [])
    if not sentences:
        return

    flagged = [s for s in sentences if s.get("generated_prob", 0) > 0.5]
    if not flagged:
        print(f"  {GREEN}No sentences flagged as AI-sounding.{RESET}")
        return

    print(f"  {len(flagged)} sentence(s) flagged:\n")
    for s in sorted(flagged, key=lambda x: -x.get("generated_prob", 0)):
        score = s.get("generated_prob", 0)
        text = s.get("sentence", "")
        color = RED if score > 0.8 else YELLOW
        truncated = text[:120] + "..." if len(text) > 120 else text
        print(f"  {color}{score:.0%}{RESET}  {truncated}")


# ---------------------------------------------------------------------------
# Shared formatting
# ---------------------------------------------------------------------------

def score_label(score: float) -> str:
    """Return a colored label for an AI probability score."""
    if score > 0.8:
        return f"{RED}HIGH{RESET}"
    elif score > 0.5:
        return f"{YELLOW}MEDIUM{RESET}"
    else:
        return f"{GREEN}LOW{RESET}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    specific = sys.argv[1] if len(sys.argv) > 1 else None
    files = get_draft_files(specific)

    has_sapling = bool(os.environ.get("SAPLING_API_KEY"))
    has_gptzero = bool(os.environ.get("GPTZERO_API_KEY"))

    if not has_sapling and not has_gptzero:
        print("No API keys found. Set one or more of:")
        print("  export SAPLING_API_KEY=your_key    (free: https://sapling.ai/docs/api)")
        print("  export GPTZERO_API_KEY=your_key    ($10/mo: https://gptzero.me/docs)")
        sys.exit(1)

    active = []
    if has_sapling:
        active.append("Sapling")
    if has_gptzero:
        active.append("GPTZero")
    print(f"Detectors: {', '.join(active)}\n")

    for path in files:
        text = load_draft(path)
        if not text.strip():
            continue

        rel = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
        print(f"{BOLD}{'─' * 60}{RESET}")
        print(f"{BOLD}{rel}{RESET}")

        if has_sapling:
            result = check_sapling(text)
            if result:
                format_sapling_results(result)
            time.sleep(0.5)  # rate limit courtesy

        if has_gptzero:
            result = check_gptzero(text)
            if result:
                format_gptzero_results(result)
            time.sleep(0.5)

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print("Done.")


if __name__ == "__main__":
    main()
