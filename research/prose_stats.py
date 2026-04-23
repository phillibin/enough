"""
prose_stats — prose quality instrument panel for writing/drafts/.

Runs every metric at its native granularity. No normalization, no blending.
Pangram responses are cached by content hash (sha1 of stripped text) so the
API isn't re-hit on unchanged sections.

Usage:
    python3 research/prose_stats.py                     # all sections
    python3 research/prose_stats.py --no-pangram        # skip API calls
    python3 research/prose_stats.py --open              # open the HTML on finish

Outputs:
    research/prose_stats_runs/<timestamp>/raw.html
    research/prose_stats_runs/<timestamp>/run.json
    research/prose_stats_cache/pangram/<sha1>.json    (cache; auto-written)

Metrics wired: stdlib_stats, pangram, concreteness (Brysbaert norms).
Metrics stubbed: llm-judge, vale.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).parent.parent
DRAFTS_DIR = REPO / "writing" / "drafts"
RUNS_DIR = REPO / "research" / "prose_stats_runs"
CACHE_DIR = REPO / "research" / "prose_stats_cache" / "pangram"
BRYSBAERT_PATH = REPO / "research" / "data" / "brysbaert_concreteness.csv"
SITE_STATS_DIR = REPO / "site" / "public" / "stats"
ENV_PATH = REPO / ".env"

SKIP_PREFIXES = ("00_", "three_axis")
PANGRAM_ENDPOINT = "https://text.api.pangram.com/v3"

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"\b[a-zA-Z'']+\b")
VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)

STOP = set(
    "the a an and or but of to in it is was are be been being that this with for "
    "as at by on from he she they we you i his her its their our your my me us them "
    "not no so if then than also which who what when where why how".split()
)


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------

def load_env_var(name: str) -> Optional[str]:
    """Read a single key from repo-root .env without the dotenv library."""
    if not ENV_PATH.exists():
        return None
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# draft discovery + cleaning
# ---------------------------------------------------------------------------

def strip_markdown(text: str) -> str:
    """Remove markdown headings; keep the prose."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.strip().startswith("#")
    ).strip()


def get_drafts() -> list[Path]:
    files = sorted(DRAFTS_DIR.glob("*.md"))
    return [f for f in files if not f.name.startswith(SKIP_PREFIXES)]


def draft_status(raw_text: str) -> str:
    """Extract the [POLISHED]/[ROUGH]/etc. tag from the first heading if present."""
    m = re.search(r"^#\s+.*?\[([A-Z ]+)\]", raw_text, re.MULTILINE)
    return m.group(1).strip().lower() if m else "unmarked"


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return [p.strip() for p in SENTENCE_END.split(flat) if p.strip()]


def tokenize(text: str) -> list[str]:
    return WORD.findall(text)


def count_syllables(word: str) -> int:
    """Heuristic: vowel groups, minus silent trailing e."""
    word = word.lower().strip("'")
    if not word:
        return 0
    groups = VOWEL_GROUP.findall(word)
    count = len(groups)
    if word.endswith("e") and count > 1 and not word.endswith("le"):
        count -= 1
    return max(1, count)


def mattr(words: list[str], window: int = 100) -> Optional[float]:
    if len(words) < window:
        return None
    ratios = [
        len(set(words[i : i + window])) / window
        for i in range(len(words) - window + 1)
    ]
    return statistics.mean(ratios)


# ---------------------------------------------------------------------------
# metric: stdlib_stats
# ---------------------------------------------------------------------------

def stdlib_stats(text: str) -> dict:
    sentences = split_sentences(text)
    words = tokenize(text)
    lowered = [w.lower() for w in words]
    word_count = len(words)
    sentence_count = len(sentences)

    word_lengths = [len(w) for w in words]
    sentence_word_counts = [len(tokenize(s)) for s in sentences]
    syllables = [count_syllables(w) for w in words]
    total_syllables = sum(syllables)
    complex_words = sum(1 for s in syllables if s >= 3)

    avg_sen = statistics.mean(sentence_word_counts) if sentence_word_counts else 0
    std_sen = statistics.stdev(sentence_word_counts) if len(sentence_word_counts) > 1 else 0

    wps = word_count / sentence_count if sentence_count else 0
    spw = total_syllables / word_count if word_count else 0

    buckets = {"1–5": 0, "6–10": 0, "11–15": 0, "16–20": 0,
               "21–25": 0, "26–30": 0, "31–40": 0, "41+": 0}
    for n in sentence_word_counts:
        if n <= 5: buckets["1–5"] += 1
        elif n <= 10: buckets["6–10"] += 1
        elif n <= 15: buckets["11–15"] += 1
        elif n <= 20: buckets["16–20"] += 1
        elif n <= 25: buckets["21–25"] += 1
        elif n <= 30: buckets["26–30"] += 1
        elif n <= 40: buckets["31–40"] += 1
        else: buckets["41+"] += 1

    pairs = sorted(zip(sentence_word_counts, sentences), key=lambda p: p[0])
    shortest = [[n, s] for n, s in pairs[:3]]
    longest = [[n, s] for n, s in pairs[-3:][::-1]]

    freqs = Counter(lowered)
    hapax = sum(1 for c in freqs.values() if c == 1)
    content_freqs = Counter(w for w in lowered if w not in STOP)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_words": len(set(lowered)),
        "total_syllables": total_syllables,
        "complex_words": complex_words,
        "avg_word_length": round(statistics.mean(word_lengths), 2) if word_lengths else 0,
        "stdev_word_length": round(statistics.stdev(word_lengths), 2) if len(word_lengths) > 1 else 0,
        "avg_sentence_length": round(avg_sen, 2),
        "stdev_sentence_length": round(std_sen, 2),
        "min_sentence_length": min(sentence_word_counts) if sentence_word_counts else 0,
        "max_sentence_length": max(sentence_word_counts) if sentence_word_counts else 0,
        "burstiness_cv": round(std_sen / avg_sen, 3) if avg_sen else 0,
        "ttr": round(len(set(lowered)) / word_count, 3) if word_count else 0,
        "mattr_100": round(mattr(lowered, 100), 3) if mattr(lowered, 100) is not None else None,
        "hapax_count": hapax,
        "hapax_ratio": round(hapax / word_count, 3) if word_count else 0,
        "flesch_ease": round(206.835 - 1.015 * wps - 84.6 * spw, 1),
        "flesch_kincaid_grade": round(0.39 * wps + 11.8 * spw - 15.59, 1),
        "gunning_fog": round(
            0.4 * (wps + 100 * complex_words / word_count), 1
        ) if word_count else 0,
        "smog": round(
            1.0430 * ((30 * complex_words / sentence_count) ** 0.5) + 3.1291, 1
        ) if sentence_count else 0,
        "sentence_buckets": buckets,
        "longest_sentences": longest,
        "shortest_sentences": shortest,
        "most_common_content_words": content_freqs.most_common(8),
    }


# ---------------------------------------------------------------------------
# metric: concreteness (Brysbaert norms)
# ---------------------------------------------------------------------------

_BRYSBAERT: Optional[dict[str, float]] = None


def _load_brysbaert() -> dict[str, float]:
    global _BRYSBAERT
    if _BRYSBAERT is not None:
        return _BRYSBAERT
    if not BRYSBAERT_PATH.exists():
        _BRYSBAERT = {}
        return _BRYSBAERT
    lex: dict[str, float] = {}
    for line in BRYSBAERT_PATH.read_text().splitlines():
        if "," not in line:
            continue
        word, score = line.rsplit(",", 1)
        try:
            lex[word.lower()] = float(score)
        except ValueError:
            continue
    _BRYSBAERT = lex
    return lex


def _normalize(word: str) -> str:
    """Lowercase + strip straight and typographic apostrophes."""
    return word.lower().replace("'", "").replace("'", "").replace("'", "")


def concreteness(text: str) -> dict:
    lex = _load_brysbaert()
    if not lex:
        return {"_error": "brysbaert lexicon not found at " + str(BRYSBAERT_PATH)}

    sentences = split_sentences(text)
    all_scores: list[float] = []
    covered_words = 0
    content_words_total = 0

    sentence_scores: list[dict] = []
    for sent in sentences:
        words = [w for w in tokenize(sent) if w.lower() not in STOP]
        content_words_total += len(words)
        s_scores: list[float] = []
        for w in words:
            norm = _normalize(w)
            if norm in lex:
                s_scores.append(lex[norm])
                covered_words += 1
                all_scores.append(lex[norm])
        sentence_scores.append({
            "sentence": sent,
            "word_count": len(words),
            "covered": len(s_scores),
            "avg": round(statistics.mean(s_scores), 3) if s_scores else None,
        })

    # sentences with at least 3 covered content words, sorted by avg
    scored = [s for s in sentence_scores if s["avg"] is not None and s["covered"] >= 3]
    scored.sort(key=lambda s: s["avg"])
    most_abstract = [
        [round(s["avg"], 2), s["sentence"]] for s in scored[:3]
    ]
    most_concrete = [
        [round(s["avg"], 2), s["sentence"]] for s in scored[-3:][::-1]
    ]

    return {
        "section_avg": round(statistics.mean(all_scores), 3) if all_scores else None,
        "section_stdev": round(statistics.stdev(all_scores), 3) if len(all_scores) > 1 else None,
        "section_min": round(min(all_scores), 2) if all_scores else None,
        "section_max": round(max(all_scores), 2) if all_scores else None,
        "coverage": round(covered_words / content_words_total, 3) if content_words_total else 0,
        "covered_words": covered_words,
        "content_words_total": content_words_total,
        "sentence_count_scored": len(scored),
        "most_abstract_sentences": most_abstract,
        "most_concrete_sentences": most_concrete,
    }


# ---------------------------------------------------------------------------
# metric: pangram (with content-hash cache)
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def pangram(text: str, key: str, no_api: bool = False) -> dict:
    """
    Returns the Pangram v3 response. Cached by sha1 of the exact text.
    If no_api is True, returns the cache entry or None — never hits the API.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = content_hash(text)
    cache_path = CACHE_DIR / f"{digest}.json"

    if cache_path.exists():
        entry = json.loads(cache_path.read_text())
        entry["_cache"] = "hit"
        return entry

    if no_api:
        return {"_cache": "miss", "_skipped": True}

    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        PANGRAM_ENDPOINT,
        data=body,
        headers={"x-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return {"_cache": "miss", "_error": f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        return {"_cache": "miss", "_error": str(e)}

    # persist raw response
    cache_path.write_text(json.dumps(result, indent=2))
    result["_cache"] = "miss"
    return result


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

STYLE = """
:root {
  --ink: #2a2520; --paper: #faf7f2; --rule: #e8e0d4; --rule-soft: #f0ebe0;
  --muted: #8a7f70; --mutedder: #b4ac9c;
}
* { box-sizing: border-box; }
body { font-family: "Inter", system-ui, sans-serif; background: var(--paper); color: var(--ink);
       margin: 0; font-size: 14px; line-height: 1.5; }
.serif { font-family: "EB Garamond", Georgia, serif; }
.mono  { font-family: "JetBrains Mono", ui-monospace, monospace; }
.muted { color: var(--muted); }
.mutedder { color: var(--mutedder); }
.page  { max-width: 1100px; margin: 0 auto; padding: 40px 32px 80px; }
header.top { display: flex; align-items: baseline; justify-content: space-between;
             border-bottom: 1px solid var(--rule); padding-bottom: 20px; margin-bottom: 32px; }
h1 { font-family: "EB Garamond", Georgia, serif; font-size: 30px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
h2 { font-family: "Inter", sans-serif; font-size: 11px; font-weight: 500; text-transform: uppercase;
     letter-spacing: 0.08em; color: var(--muted); margin: 0 0 14px 0; }
h3 { font-family: "EB Garamond", Georgia, serif; font-size: 20px; font-weight: 600; margin: 0 0 4px 0; }
table.cmp { width: 100%; border-collapse: collapse; font-size: 13px; }
table.cmp th, table.cmp td { padding: 6px 12px; text-align: right; border-bottom: 1px solid var(--rule-soft); }
table.cmp th:first-child, table.cmp td:first-child { text-align: left; font-weight: 400; color: var(--ink); padding-left: 0; }
table.cmp th:first-child { color: var(--muted); }
table.cmp thead th { font-weight: 500; font-size: 12px; color: var(--muted);
                     border-bottom: 1px solid var(--rule); padding-top: 2px; padding-bottom: 8px; }
table.cmp thead th.section-head { color: var(--ink); font-family: "EB Garamond", serif; font-size: 16px; font-weight: 600; }
table.cmp .num { font-family: "JetBrains Mono", monospace; font-size: 12px; }
table.cmp tr.group-start td { padding-top: 14px; }
table.cmp tr.group-start td:first-child { color: var(--muted); font-size: 11px; text-transform: uppercase;
                                          letter-spacing: 0.08em; padding-top: 18px; padding-bottom: 2px;
                                          border-bottom: none; font-weight: 500; }
table.cmp tr.group-start td:not(:first-child) { border-bottom: none; }
table.cmp tr:hover td { background: #f5ede0; }
table.cmp td.unit { color: var(--mutedder); font-family: "JetBrains Mono", monospace; font-size: 11px; padding-left: 4px; text-align: left; }
.tool-badge { display: inline-block; font-family: "JetBrains Mono", monospace; font-size: 10px;
              padding: 1px 6px; border-radius: 3px; margin-left: 6px; background: #ece4d4;
              color: var(--muted); vertical-align: middle; }
section.section { border-top: 1px solid var(--rule); padding-top: 24px; margin-top: 40px; }
.section-header { display: flex; align-items: baseline; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }
.section-header .status { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
                          padding: 2px 7px; border-radius: 3px; font-weight: 500; }
.status.polished { background: #e0ead9; color: #3d5132; }
.status.rough    { background: #efe7d2; color: #6a5c36; }
.status.unmarked { background: #ece4d4; color: #8a7f70; }
.section-header .meta { color: var(--muted); font-family: "JetBrains Mono", monospace; font-size: 12px; }
.deep-grid { display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 28px; }
.histogram { display: flex; flex-direction: column; gap: 2px; font-size: 11px; }
.histogram .row { display: grid; grid-template-columns: 50px 1fr 28px; align-items: center; gap: 8px; }
.histogram .label { font-family: "JetBrains Mono", monospace; color: var(--muted); text-align: right; font-size: 11px; }
.histogram .bar-wrap { height: 14px; display: flex; align-items: center; }
.histogram .bar { height: 10px; background: #c6b28f; border-radius: 1px; }
.histogram .count { font-family: "JetBrains Mono", monospace; color: var(--ink); font-size: 11px; text-align: right; }
ul.list { margin: 0; padding: 0; list-style: none; }
ul.list li { padding: 4px 0; border-bottom: 1px dotted var(--rule-soft); display: flex; align-items: baseline; gap: 10px; }
ul.list li:last-child { border-bottom: none; }
ul.list .k { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--muted); min-width: 28px; text-align: right; }
ul.list .v { font-family: "EB Garamond", Georgia, serif; font-size: 14px; font-style: italic; }
ul.list .v.plain { font-family: "Inter", sans-serif; font-style: normal; font-size: 13px; }
.stub { background: #f5ede0; border: 1px dashed #d8ceba; border-radius: 4px;
        padding: 14px 18px; font-size: 12px; color: var(--muted); margin-bottom: 20px; }
.stub strong { color: var(--ink); font-weight: 500; }
.group-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
               color: var(--muted); margin-bottom: 6px; font-weight: 500; }
.cache-hit  { color: #3d5132; }
.cache-miss { color: #8a4f27; }
.cache-err  { color: #8a3124; }
.pangram-ai td { background: #fdf0ec; color: #8a3124; }
footer.bottom { margin-top: 60px; padding-top: 16px; border-top: 1px solid var(--rule);
                color: var(--muted); font-size: 12px; line-height: 1.6; }
footer.bottom strong { color: var(--ink); font-weight: 500; }
"""

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>prose stats · {title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;450;500&display=swap" rel="stylesheet" />
<style>{style}</style>
</head>
<body><div class="page">
"""

FOOT = """
</div></body></html>
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def render_comparison_table(sections: list[dict], pangram_available: bool) -> str:
    """Top-level document-level comparison across all sections."""
    names = [s["name"] for s in sections]

    def row(label: str, values: list, unit: str = "") -> str:
        cells = "".join(f'<td class="num">{esc(v)}</td>' for v in values)
        return f'<tr><td>{esc(label)}</td>{cells}<td class="unit">{esc(unit)}</td></tr>'

    def group(title: str, badge: str) -> str:
        empty = "".join(f"<td></td>" for _ in range(len(names) + 1))
        return (
            f'<tr class="group-start"><td>{esc(title)} '
            f'<span class="tool-badge">{esc(badge)}</span></td>{empty}</tr>'
        )

    def stat(key, fallback="—"):
        return [
            s["stdlib"].get(key, fallback) if s["stdlib"].get(key) is not None else "n/a"
            for s in sections
        ]

    rows = []
    rows.append(group("counts", "stdlib"))
    rows.append(row("word_count", stat("word_count"), "words"))
    rows.append(row("sentence_count", stat("sentence_count"), "sentences"))
    rows.append(row("unique_words", stat("unique_words"), "types"))
    rows.append(row("total_syllables", stat("total_syllables"), "syllables"))
    rows.append(row("complex_words (≥3 syl)", stat("complex_words"), "words"))

    rows.append(group("length", "stdlib"))
    rows.append(row("avg_word_length", stat("avg_word_length"), "chars"))
    rows.append(row("stdev_word_length", stat("stdev_word_length"), "chars"))
    rows.append(row("avg_sentence_length", stat("avg_sentence_length"), "words"))
    rows.append(row("stdev_sentence_length", stat("stdev_sentence_length"), "words"))
    rows.append(row("min_sentence_length", stat("min_sentence_length"), "words"))
    rows.append(row("max_sentence_length", stat("max_sentence_length"), "words"))

    rows.append(group("burstiness", "stdlib"))
    rows.append(row("sentence_length_cv", stat("burstiness_cv"), "σ/μ"))

    rows.append(group("vocabulary", "stdlib"))
    rows.append(row("ttr (whole text)", stat("ttr"), "unique/total"))
    rows.append(row("mattr_100 (sliding window)", stat("mattr_100"), "avg 100-w win"))
    rows.append(row("hapax_count", stat("hapax_count"), "once-only"))
    rows.append(row("hapax_ratio", stat("hapax_ratio"), "of total"))

    rows.append(group("readability", "stdlib"))
    rows.append(row("flesch_ease", stat("flesch_ease"), "higher=easier"))
    rows.append(row("flesch_kincaid_grade", stat("flesch_kincaid_grade"), "US grade"))
    rows.append(row("gunning_fog", stat("gunning_fog"), "US grade"))
    rows.append(row("smog", stat("smog"), "US grade"))

    if pangram_available:
        def pg(fn, missing="—"):
            out = []
            for s in sections:
                p = s.get("pangram") or {}
                if p.get("_skipped") or p.get("_error") or not p:
                    out.append(missing)
                else:
                    out.append(fn(p))
            return out

        rows.append(group("pangram", "api"))
        rows.append(row("prediction", pg(lambda p: p.get("prediction_short", "—")), "overall label"))
        rows.append(row("fraction_ai", pg(lambda p: f"{p.get('fraction_ai', 0):.2f}"), "0–1"))
        rows.append(row("fraction_ai_assisted", pg(lambda p: f"{p.get('fraction_ai_assisted', 0):.2f}"), "0–1"))
        rows.append(row("fraction_human", pg(lambda p: f"{p.get('fraction_human', 0):.2f}"), "0–1"))
        rows.append(row("num_ai_segments", pg(lambda p: p.get("num_ai_segments", 0)), "segs"))
        rows.append(row("num_human_segments", pg(lambda p: p.get("num_human_segments", 0)), "segs"))
        rows.append(row(
            "max_window_score",
            pg(lambda p: f"{max((w.get('ai_assistance_score', 0) for w in p.get('windows', [])), default=0):.2f}"),
            "worst seg",
        ))
        rows.append(row(
            "confidence (max)",
            pg(lambda p: max(
                (w.get("confidence", "—") for w in p.get("windows", [])),
                key=lambda c: ["Low", "Medium", "High"].index(c) if c in ("Low", "Medium", "High") else -1,
                default="—",
            )),
            "",
        ))

    def cn(fn, missing="—"):
        out = []
        for s in sections:
            c = s.get("concreteness") or {}
            if c.get("_error") or not c:
                out.append(missing)
            else:
                out.append(fn(c))
        return out

    rows.append(group("concreteness", "brysbaert"))
    rows.append(row("section_avg", cn(lambda c: f"{c.get('section_avg', 0):.2f}" if c.get("section_avg") else "—"), "1–5"))
    rows.append(row("section_stdev", cn(lambda c: f"{c.get('section_stdev', 0):.2f}" if c.get("section_stdev") else "—"), ""))
    rows.append(row("section_min", cn(lambda c: f"{c.get('section_min', 0):.2f}" if c.get("section_min") else "—"), "word"))
    rows.append(row("section_max", cn(lambda c: f"{c.get('section_max', 0):.2f}" if c.get("section_max") else "—"), "word"))
    rows.append(row("coverage", cn(lambda c: f"{c.get('coverage', 0):.2f}"), "of content words"))

    head_cols = "".join(f'<th class="section-head">{esc(n)}</th>' for n in names)
    return f"""
<section>
  <h2>document-level metrics</h2>
  <table class="cmp">
    <thead><tr><th></th>{head_cols}<th></th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def render_histogram(buckets: dict) -> str:
    max_count = max(buckets.values()) if buckets.values() else 1
    rows = []
    for label, count in buckets.items():
        pct = (count / max_count * 100) if max_count else 0
        rows.append(
            f'<div class="row"><span class="label">{esc(label)}</span>'
            f'<div class="bar-wrap"><div class="bar" style="width: {pct:.0f}%;"></div></div>'
            f'<span class="count">{esc(count)}</span></div>'
        )
    return f'<div class="histogram">{"".join(rows)}</div>'


def render_sentence_list(pairs: list) -> str:
    items = "".join(
        f'<li><span class="k">{esc(n)}</span><span class="v">"{esc(s)}"</span></li>'
        for n, s in pairs
    )
    return f'<ul class="list">{items}</ul>'


def render_word_list(pairs: list) -> str:
    items = "".join(
        f'<li><span class="k">{esc(n)}</span><span class="v plain">{esc(w)}</span></li>'
        for w, n in pairs
    )
    return f'<ul class="list">{items}</ul>'


def render_concreteness_block(c: dict) -> str:
    if not c:
        return '<div class="stub"><strong>concreteness</strong> · not run</div>'
    if c.get("_error"):
        return f'<div class="stub cache-err"><strong>concreteness</strong> · error · {esc(c["_error"])}</div>'

    def sent_rows(pairs):
        return "".join(
            f'<li><span class="k">{esc(v)}</span><span class="v">"{esc(s)}"</span></li>'
            for v, s in pairs
        )

    avg = c.get("section_avg")
    return f"""
<div style="margin-top: 24px;">
  <div class="group-title">concreteness <span class="tool-badge">brysbaert</span></div>
  <div style="display: grid; grid-template-columns: 1fr 1.3fr 1.3fr; gap: 28px; margin-top: 8px;">
    <div>
      <div class="muted" style="font-size: 11px;">section avg (1 abstract → 5 concrete)</div>
      <div class="mono" style="font-size: 28px; margin-top: 4px;">{avg:.2f}</div>
      <div class="muted" style="font-size: 11px; margin-top: 6px;">
        stdev {c.get("section_stdev", 0):.2f} · range {c.get("section_min", 0):.2f}–{c.get("section_max", 0):.2f}<br/>
        coverage {c.get("coverage", 0):.0%} ({c.get("covered_words", 0)}/{c.get("content_words_total", 0)})
      </div>
    </div>
    <div>
      <div class="group-title">most abstract sentences</div>
      <ul class="list">{sent_rows(c.get("most_abstract_sentences", []))}</ul>
    </div>
    <div>
      <div class="group-title">most concrete sentences</div>
      <ul class="list">{sent_rows(c.get("most_concrete_sentences", []))}</ul>
    </div>
  </div>
</div>
"""


def render_pangram_windows(pg: dict) -> str:
    if not pg or pg.get("_skipped"):
        return '<div class="stub"><strong>pangram</strong> · skipped (--no-pangram)</div>'
    if pg.get("_error"):
        return f'<div class="stub cache-err"><strong>pangram</strong> · error · {esc(pg["_error"])}</div>'

    rows = []
    for w in pg.get("windows", []):
        is_ai = (w.get("label") or "").lower().startswith("ai")
        row_class = "pangram-ai" if is_ai else ""
        snippet = (w.get("text", "")[:90].replace("\n", " ")).strip()
        if len(w.get("text", "")) > 90:
            snippet += "…"
        rows.append(
            f'<tr class="{row_class}">'
            f'<td class="num">{w["start_index"]}–{w["end_index"]}</td>'
            f'<td class="num">{w.get("word_count", "—")}</td>'
            f'<td>{esc(w.get("label", "—"))}</td>'
            f'<td class="num">{w.get("ai_assistance_score", 0):.2f}</td>'
            f'<td>{esc(w.get("confidence", "—"))}</td>'
            f'<td style="text-align: left;" class="muted serif">'
            f'<span style="font-style: italic;">"{esc(snippet)}"</span></td>'
            f'</tr>'
        )

    cache_tag = (
        '<span class="cache-hit">cached</span>' if pg.get("_cache") == "hit"
        else '<span class="cache-miss">fresh</span>'
    )
    return f"""
<div style="margin-top: 24px;">
  <div class="group-title">pangram · per-window <span class="tool-badge">api</span>
    <span style="font-size: 10px; margin-left: 8px;">{cache_tag}</span>
  </div>
  <table class="cmp" style="font-size: 12px;">
    <thead><tr>
      <th>range (chars)</th><th>words</th><th>label</th><th>score</th><th>conf</th><th></th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
"""


def render_section_block(s: dict) -> str:
    stats = s["stdlib"]
    pg = s.get("pangram")
    slug = s["name"]

    return f"""
<section class="section" id="{esc(slug)}">
  <div class="section-header">
    <h3>{esc(slug)}</h3>
    <span class="status {esc(s['status'])}">{esc(s['status'])}</span>
    <span class="meta">{stats['word_count']} words · {stats['sentence_count']} sentences</span>
    <a href="../#{esc(slug)}" class="muted" style="font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-left: auto; text-decoration: none; border-bottom: 1px dotted var(--rule);">read ↗</a>
  </div>
  <div class="deep-grid">
    <div>
      <div class="group-title">sentence-length distribution <span class="tool-badge">stdlib</span></div>
      {render_histogram(stats['sentence_buckets'])}
    </div>
    <div>
      <div class="group-title">longest / shortest sentences <span class="tool-badge">stdlib</span></div>
      {render_sentence_list(stats['longest_sentences'] + stats['shortest_sentences'])}
    </div>
    <div>
      <div class="group-title">most-used content words <span class="tool-badge">stdlib</span></div>
      {render_word_list(stats['most_common_content_words'])}
    </div>
  </div>
  {render_pangram_windows(pg) if pg is not None else ''}
  {render_concreteness_block(s.get("concreteness") or {})}
</section>
"""


def render_page(sections: list[dict], run_meta: dict) -> str:
    pangram_available = any(s.get("pangram") for s in sections)

    n_sections = len(sections)
    n_cached = sum(1 for s in sections if (s.get("pangram") or {}).get("_cache") == "hit")
    n_fresh = sum(1 for s in sections if (s.get("pangram") or {}).get("_cache") == "miss" and not (s.get("pangram") or {}).get("_error"))
    n_err = sum(1 for s in sections if (s.get("pangram") or {}).get("_error"))

    pangram_note = ""
    if pangram_available:
        pangram_note = (
            f' · pangram: <span class="cache-hit">{n_cached} cached</span>, '
            f'<span class="cache-miss">{n_fresh} fresh</span>'
        )
        if n_err:
            pangram_note += f', <span class="cache-err">{n_err} error</span>'

    header = f"""
<header class="top">
  <div>
    <a href="../" class="muted" style="font-family: 'JetBrains Mono', monospace; font-size: 11px; text-decoration: none; border-bottom: 1px dotted var(--rule);">← reading</a>
    <h1 style="margin-top: 6px;">prose stats · raw</h1>
    <p class="muted" style="margin: 4px 0 0; font-size: 13px;">enough · moneysnake · every tool at its native granularity</p>
  </div>
  <div class="mono muted" style="text-align: right; font-size: 11px;">
    <div>run {esc(run_meta['timestamp'])}</div>
    <div style="margin-top: 4px;">{n_sections} sections{pangram_note}</div>
  </div>
</header>
"""

    section_blocks = "".join(render_section_block(s) for s in sections)

    footer = """
<footer class="bottom">
  <p><strong>What you're looking at.</strong> Every tool at its own native granularity. No percentiles, no normalization, no cross-metric severity blends.</p>
  <p style="margin-top: 10px;"><strong>Caching.</strong> Pangram responses are keyed by sha1(text). Edit a section → next run is fresh for that section only. Other sections stay cached. Cache lives in <code>research/prose_stats_cache/pangram/</code>.</p>
  <p style="margin-top: 10px;"><strong>Still to wire.</strong> LLM-judge (rubric scores + quoted offenders). Vale with a Phil-curated <code>slop.yml</code> ruleset.</p>
</footer>
"""

    return (
        HEAD.format(title="enough · raw", style=STYLE)
        + header
        + render_comparison_table(sections, pangram_available)
        + section_blocks
        + footer
        + FOOT
    )


# ---------------------------------------------------------------------------
# terminal summary
# ---------------------------------------------------------------------------

RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; BOLD = "\033[1m"
DIM = "\033[2m"; RESET = "\033[0m"


def print_summary(sections: list[dict]) -> None:
    print(f"\n{BOLD}{'section':<22} {'words':>6} {'burst':>6} {'mattr':>6} {'concr':>6} {'fog':>5} {'pangram':>18}{RESET}")
    print(DIM + "─" * 80 + RESET)
    for s in sections:
        st = s["stdlib"]
        pg = s.get("pangram") or {}
        c = s.get("concreteness") or {}
        if pg.get("_error"):
            pg_str = f"{RED}err{RESET}"
        elif pg.get("_skipped"):
            pg_str = f"{DIM}skipped{RESET}"
        elif pg:
            label = pg.get("prediction_short", "—")
            ai_frac = pg.get("fraction_ai", 0)
            cache = "c" if pg.get("_cache") == "hit" else "f"
            color = RED if ai_frac > 0.5 else GREEN if ai_frac < 0.2 else YELLOW
            pg_str = f"{color}{label} ai={ai_frac:.2f}{RESET} [{cache}]"
        else:
            pg_str = "—"
        mattr_val = st.get("mattr_100")
        concr_val = c.get("section_avg")
        concr_str = f"{concr_val:.2f}" if concr_val is not None else "—"
        mattr_str = f"{mattr_val:.3f}" if mattr_val is not None else "—"
        print(
            f"{s['name']:<22} {st['word_count']:>6} "
            f"{st['burstiness_cv']:>6.3f} "
            f"{mattr_str:>6} "
            f"{concr_str:>6} "
            f"{st['gunning_fog']:>5} {pg_str:>26}"
        )
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="prose stats · raw")
    parser.add_argument("--no-pangram", action="store_true", help="skip Pangram API calls")
    parser.add_argument("--open", action="store_true", help="open the HTML report on finish")
    args = parser.parse_args()

    drafts = get_drafts()
    if not drafts:
        print("no drafts found.")
        sys.exit(1)

    key = None if args.no_pangram else load_env_var("PANGRAM_API_KEY")
    if not args.no_pangram and not key:
        print(f"{YELLOW}warning: PANGRAM_API_KEY not set; skipping pangram{RESET}")

    sections = []
    for path in drafts:
        raw = path.read_text()
        text = strip_markdown(raw)
        if not text:
            continue
        print(f"→ {path.name}")
        entry = {
            "name": path.stem,
            "path": str(path.relative_to(REPO)),
            "status": draft_status(raw),
            "content_hash": content_hash(text),
            "stdlib": stdlib_stats(text),
            "concreteness": concreteness(text),
        }
        c = entry["concreteness"]
        if c.get("_error"):
            print(f"  concreteness: {RED}error{RESET} {c['_error'][:80]}")
        else:
            print(f"  concreteness: avg={c.get('section_avg', 0):.2f} coverage={c.get('coverage', 0):.0%}")
        if key or args.no_pangram:
            entry["pangram"] = pangram(text, key or "", no_api=args.no_pangram)
            cache = entry["pangram"].get("_cache", "—")
            if entry["pangram"].get("_error"):
                print(f"  pangram: {RED}error{RESET} {entry['pangram']['_error'][:80]}")
            elif entry["pangram"].get("_skipped"):
                print(f"  pangram: {DIM}skipped{RESET}")
            else:
                print(f"  pangram: {entry['pangram'].get('prediction_short', '?')} [{cache}]")
        sections.append(entry)

    # run dir
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp": ts,
        "n_sections": len(sections),
        "pangram_enabled": bool(key) and not args.no_pangram,
    }

    # JSON (omit pangram internals that are huge; keep key summary)
    run_json = {
        "meta": run_meta,
        "sections": [
            {
                **{k: v for k, v in s.items() if k != "pangram"},
                "pangram_summary": (
                    {
                        "prediction_short": (s.get("pangram") or {}).get("prediction_short"),
                        "fraction_ai": (s.get("pangram") or {}).get("fraction_ai"),
                        "fraction_ai_assisted": (s.get("pangram") or {}).get("fraction_ai_assisted"),
                        "fraction_human": (s.get("pangram") or {}).get("fraction_human"),
                        "cache": (s.get("pangram") or {}).get("_cache"),
                        "num_windows": len((s.get("pangram") or {}).get("windows", [])),
                    }
                    if s.get("pangram") else None
                ),
            }
            for s in sections
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2))

    # HTML — timestamped archive
    html_path = run_dir / "raw.html"
    rendered = render_page(sections, run_meta)
    html_path.write_text(rendered)

    # HTML — site copy (served by Astro at /enough/stats/)
    SITE_STATS_DIR.mkdir(parents=True, exist_ok=True)
    site_path = SITE_STATS_DIR / "index.html"
    site_path.write_text(rendered)

    print_summary(sections)
    print(f"{DIM}→ {html_path.relative_to(REPO)}{RESET}")
    print(f"{DIM}→ {site_path.relative_to(REPO)} (served at /enough/stats/){RESET}")

    if args.open:
        subprocess.run(["open", str(html_path)], check=False)


if __name__ == "__main__":
    main()
