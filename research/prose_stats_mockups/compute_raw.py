"""
Quick one-off: compute raw prose stats for each draft in writing/drafts/.
Stdlib only. Throwaway script to feed real numbers into the mockup.
"""

import re
import statistics
from pathlib import Path

DRAFTS = Path(__file__).parent.parent.parent / "writing" / "drafts"
SKIP_PREFIXES = ("00_", "three_axis")

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"\b[a-zA-Z'']+\b")
VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)


def strip_markdown(text: str) -> str:
    """Strip headings and status tags, keep the prose."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def split_sentences(text: str) -> list[str]:
    """Rough sentence split. Good enough for mockup numbers."""
    # collapse paragraphs into flowing text, but keep em-dash-free sentences whole
    flat = re.sub(r"\s+", " ", text)
    parts = SENTENCE_END.split(flat)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    return WORD.findall(text)


def count_syllables(word: str) -> int:
    """Heuristic syllable count. Good enough for Flesch/Kincaid estimates."""
    word = word.lower().strip("'")
    if not word:
        return 0
    # count vowel groups
    groups = VOWEL_GROUP.findall(word)
    count = len(groups)
    # silent trailing e
    if word.endswith("e") and count > 1 and not word.endswith("le"):
        count -= 1
    return max(1, count)


def mattr(words: list[str], window: int = 100) -> float | None:
    """Moving-Average Type-Token Ratio over a sliding window."""
    if len(words) < window:
        return None
    ratios = []
    for i in range(len(words) - window + 1):
        w = words[i : i + window]
        ratios.append(len(set(w)) / window)
    return statistics.mean(ratios)


def analyze(text: str) -> dict:
    sentences = split_sentences(text)
    words = tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    lowered = [w.lower() for w in words]

    word_lengths = [len(w) for w in words]
    sentence_word_counts = [len(tokenize(s)) for s in sentences]
    syllables_per_word = [count_syllables(w) for w in words]
    total_syllables = sum(syllables_per_word)
    complex_words = sum(1 for s in syllables_per_word if s >= 3)

    avg_sen_len = statistics.mean(sentence_word_counts) if sentence_word_counts else 0
    stdev_sen_len = statistics.stdev(sentence_word_counts) if len(sentence_word_counts) > 1 else 0
    burstiness_cv = stdev_sen_len / avg_sen_len if avg_sen_len else 0

    ttr = len(set(lowered)) / word_count if word_count else 0
    mattr_100 = mattr(lowered, 100)

    # readability formulas
    wps = word_count / sentence_count if sentence_count else 0
    spw = total_syllables / word_count if word_count else 0
    flesch_ease = 206.835 - 1.015 * wps - 84.6 * spw
    flesch_kincaid_grade = 0.39 * wps + 11.8 * spw - 15.59
    gunning_fog = 0.4 * (wps + 100 * complex_words / word_count) if word_count else 0
    smog = 1.0430 * ((30 * complex_words / sentence_count) ** 0.5) + 3.1291 if sentence_count else 0

    # sentence-length distribution buckets (for the mockup histogram)
    buckets = {
        "1-5": 0, "6-10": 0, "11-15": 0, "16-20": 0,
        "21-25": 0, "26-30": 0, "31-40": 0, "41+": 0,
    }
    for n in sentence_word_counts:
        if n <= 5: buckets["1-5"] += 1
        elif n <= 10: buckets["6-10"] += 1
        elif n <= 15: buckets["11-15"] += 1
        elif n <= 20: buckets["16-20"] += 1
        elif n <= 25: buckets["21-25"] += 1
        elif n <= 30: buckets["26-30"] += 1
        elif n <= 40: buckets["31-40"] += 1
        else: buckets["41+"] += 1

    # top sentence-length extremes
    longest = sorted(
        zip(sentence_word_counts, sentences), key=lambda p: -p[0]
    )[:3]
    shortest = sorted(
        zip(sentence_word_counts, sentences), key=lambda p: p[0]
    )[:3]

    # hapax legomena (words used exactly once)
    from collections import Counter
    freqs = Counter(lowered)
    hapax = sum(1 for c in freqs.values() if c == 1)

    # most-used words (excluding a small stopword list)
    stop = set("the a an and or but of to in it is was are be been being that this with for as at by on from he she they we you i his her its their our your my me us them not no so if then than also which who what when where why how".split())
    content_freqs = Counter(w for w in lowered if w not in stop)
    most_common = content_freqs.most_common(8)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_words": len(set(lowered)),
        "avg_word_length": statistics.mean(word_lengths) if word_lengths else 0,
        "stdev_word_length": statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0,
        "avg_sentence_length": avg_sen_len,
        "stdev_sentence_length": stdev_sen_len,
        "burstiness_cv": burstiness_cv,
        "min_sentence_length": min(sentence_word_counts) if sentence_word_counts else 0,
        "max_sentence_length": max(sentence_word_counts) if sentence_word_counts else 0,
        "ttr": ttr,
        "mattr_100": mattr_100,
        "hapax_count": hapax,
        "hapax_ratio": hapax / word_count if word_count else 0,
        "total_syllables": total_syllables,
        "complex_words": complex_words,
        "flesch_ease": flesch_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "gunning_fog": gunning_fog,
        "smog": smog,
        "sentence_buckets": buckets,
        "longest_sentences": longest,
        "shortest_sentences": shortest,
        "most_common_content_words": most_common,
    }


def main():
    files = sorted(DRAFTS.glob("*.md"))
    files = [f for f in files if not f.name.startswith(SKIP_PREFIXES)]

    for path in files:
        raw = path.read_text()
        text = strip_markdown(raw)
        if not text:
            continue
        stats = analyze(text)
        print(f"\n{'=' * 60}")
        print(f"{path.name}")
        print(f"{'=' * 60}")
        for k, v in stats.items():
            if k in ("sentence_buckets", "longest_sentences", "shortest_sentences", "most_common_content_words"):
                print(f"{k}:")
                if isinstance(v, dict):
                    for bk, bv in v.items():
                        print(f"  {bk:>6}: {bv}")
                else:
                    for item in v:
                        print(f"  {item}")
            elif isinstance(v, float):
                print(f"{k}: {v:.3f}")
            else:
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()
