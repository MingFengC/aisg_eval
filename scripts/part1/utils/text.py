"""Text normalization, tokenization, chunking, and hard-QA helpers."""

from __future__ import annotations

import hashlib
import json
import re


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
HEADING_RE = re.compile(r"^\s*(?:\*\*)?\s*(\d+(?:\.\d+)*)\b")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")

INDONESIAN_MARKERS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "dalam",
    "adalah",
    "sebagai",
    "pada",
    "ini",
    "itu",
    "akan",
    "tidak",
    "dapat",
    "juga",
    "karena",
    "atau",
    "oleh",
    "para",
    "lebih",
    "agar",
    "secara",
    "terhadap",
    "antara",
    "mereka",
    "kita",
    "siswa",
    "pembelajaran",
    "penelitian",
}

ENGLISH_STOPWORDS_FOR_COPY = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "with",
}


def normalize_source_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(str(text))


def word_count(text: str) -> int:
    return len(tokenize_words(text))


def stable_source_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def lower_tokens(text: str) -> list[str]:
    return [token.lower() for token in tokenize_words(text)]


def compute_copy_rate(source: str, reference: str) -> float:
    source_tokens = {
        token
        for token in lower_tokens(source)
        if len(token) >= 4 and token not in ENGLISH_STOPWORDS_FOR_COPY
    }
    if not source_tokens:
        return 0.0
    reference_tokens = set(lower_tokens(reference))
    return len(source_tokens & reference_tokens) / len(source_tokens)


def indonesian_marker_stats(reference: str) -> tuple[int, float]:
    tokens = lower_tokens(reference)
    if not tokens:
        return 0, 0.0
    marker_count = sum(1 for token in tokens if token in INDONESIAN_MARKERS)
    return marker_count, marker_count / len(tokens)


def heading_numbers(text: str) -> list[str]:
    numbers = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            numbers.append(match.group(1))
    return numbers


def list_marker_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if LIST_MARKER_RE.match(line))


def has_repeated_text(reference: str) -> bool:
    lines = [line.strip() for line in reference.splitlines() if line.strip()]
    if len(lines) >= 6:
        duplicate_lines = len(lines) - len(set(lines))
        if duplicate_lines / len(lines) > 0.35:
            return True
    sentences = SENTENCE_SPLIT_RE.split(reference)
    normalized = [sentence.strip().lower() for sentence in sentences if len(sentence.split()) >= 8]
    if len(normalized) >= 6:
        duplicate_sentences = len(normalized) - len(set(normalized))
        if duplicate_sentences / len(normalized) > 0.25:
            return True
    return False


def clean_model_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_from_model_output(response: str) -> dict:
    cleaned = clean_model_output(response)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)
