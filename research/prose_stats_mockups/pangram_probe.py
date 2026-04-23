"""
One-off: hit Pangram v3 for each draft and dump the raw response.
Stdlib only. Results go into the raw.html mockup by hand.
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
DRAFTS = REPO / "writing" / "drafts"
SKIP_PREFIXES = ("00_", "three_axis")
ENDPOINT = "https://text.api.pangram.com/v3"


def load_key() -> str:
    """Read PANGRAM_API_KEY from repo-root .env without dotenv."""
    env_path = REPO / ".env"
    if not env_path.exists():
        sys.exit("no .env at repo root")
    for line in env_path.read_text().splitlines():
        if line.startswith("PANGRAM_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("PANGRAM_API_KEY not found in .env")


def strip_markdown(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#")).strip()


def call_pangram(text: str, key: str) -> dict:
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "x-api-key": key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        raise


def main():
    key = load_key()
    files = sorted(DRAFTS.glob("*.md"))
    files = [f for f in files if not f.name.startswith(SKIP_PREFIXES)]

    out_dir = Path(__file__).parent / "pangram_responses"
    out_dir.mkdir(exist_ok=True)

    for path in files:
        text = strip_markdown(path.read_text())
        if not text:
            continue
        print(f"→ {path.name} ({len(text.split())} words)")
        try:
            result = call_pangram(text, key)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        out_path = out_dir / f"{path.stem}.json"
        out_path.write_text(json.dumps(result, indent=2))
        # print summary
        print(
            f"  prediction: {result.get('prediction_short') or result.get('prediction')} | "
            f"ai={result.get('fraction_ai'):.2f} "
            f"assisted={result.get('fraction_ai_assisted'):.2f} "
            f"human={result.get('fraction_human'):.2f} | "
            f"windows={len(result.get('windows', []))}"
        )


if __name__ == "__main__":
    main()
