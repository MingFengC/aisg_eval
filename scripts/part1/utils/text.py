"""Text normalization, tokenization, chunking, and hard-QA helpers."""

from __future__ import annotations

import hashlib
import json
import re


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
HEADING_RE = re.compile(r"^\s*(?:\*\*)?\s*(\d+(?:\.\d+)*)\b")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
TERMINAL_END_CHARS = set(".!?)]}\"'’”>")
TRAILING_MARKDOWN_RE = re.compile(r"[\s*_`]+$")
WORD_END_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+|\d+")
LLM_ARTIFACT_PATTERNS = [
    re.compile(pattern, flags=re.I | re.S)
    for pattern in (
        r"^\s*(?:sure(?: thing)?|certainly|of course|absolutely)[!.]?\s+"
        r"(?:here(?:'s| is| are)|i(?:'d| would) be happy|let's|i can)\b",
        r"^\s*here(?:'s| is| are)\s+(?:a|an|the|sample|draft)\b",
        r"\bas an ai language model\b",
        r"\bi (?:am|'m) (?:an? )?(?:ai|language model)\b",
        r"\bi (?:cannot|can't) (?:browse|access|provide|assist)\b",
        r"\bi hope (?:this|the) (?:sub[- ]?unit|section|chapter|lesson|module|response|answer)"
        r".{0,120}\b(?:meets|satisfies|matches).{0,80}\b(?:your )?(?:requirements|needs|request)\b",
        r"\b(?:please )?let me know if you (?:have|need|would like|want)"
        r".{0,120}\b(?:feedback|suggestions|questions|changes|revisions?|improvements?)\b",
        r"\bi hope this helps\b",
        r"\bi hope you find this (?:helpful|useful)\b",
    )
]

DANGLING_FINAL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "based",
    "because",
    "between",
    "by",
    "come",
    "dari",
    "dan",
    "dengan",
    "di",
    "for",
    "from",
    "if",
    "in",
    "into",
    "ke",
    "ketika",
    "mencari",
    "of",
    "on",
    "or",
    "ratusan",
    "searching",
    "sebagai",
    "that",
    "the",
    "to",
    "untuk",
    "when",
    "where",
    "which",
    "while",
    "with",
    "yang",
}

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


def looks_incomplete_text(text: str) -> bool:
    """Conservative heuristic for source/reference rows that end mid-stream.

    The source dataset contains some rows that stop inside a sentence, citation,
    code block, markdown heading, or numeric expression. Those rows can produce
    faithful but unusable translations, so we filter or flag them before using
    the examples as references.
    """

    stripped = TRAILING_MARKDOWN_RE.sub("", str(text).strip())
    if not stripped:
        return True

    tail = stripped[-160:].strip()
    final_line = stripped.splitlines()[-1].strip()
    if final_line in {"#", "##", "###", "####"}:
        return True
    if stripped.endswith(("...", "…")):
        return True
    if stripped.count("```") % 2 == 1:
        return True
    if tail.endswith((",", ";", ":", "-", "–", "—", "/", "\\")):
        return True

    last_char = stripped[-1]
    if last_char in TERMINAL_END_CHARS:
        return False
    if last_char.isdigit():
        return True

    tokens = WORD_END_RE.findall(tail)
    last_token = tokens[-1].lower() if tokens else ""
    if last_token in DANGLING_FINAL_WORDS:
        return True
    if last_char not in TERMINAL_END_CHARS:
        return True
    return False


def has_llm_artifact_text(text: str) -> bool:
    """Detect explicit assistant/meta artifacts in otherwise educational text.

    Cosmopedia-style data is synthetic, so this intentionally does not try to
    classify whether a passage was machine-written. It only catches boilerplate
    that points outside the educational document itself, such as "let me know if
    you have feedback" or "I hope this sub-unit meets your requirements".
    """

    normalized = normalize_source_text(text).lower()
    return any(pattern.search(normalized) for pattern in LLM_ARTIFACT_PATTERNS)


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
