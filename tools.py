from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "a",
    "about",
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
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Return a short extractive summary using word-frequency scoring."""
    if not text or not text.strip():
        return "No text provided."

    max_sentences = max(1, min(int(max_sentences), 10))
    sentences = _sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    frequencies = Counter(
        word
        for word in _words(text)
        if word not in STOPWORDS and len(word) > 2
    )
    if not frequencies:
        return " ".join(sentences[:max_sentences])

    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        words = [
            word
            for word in _words(sentence)
            if word not in STOPWORDS and len(word) > 2
        ]
        score = sum(frequencies[word] for word in words) / max(len(words), 1)
        scored.append((score, index, sentence))

    top = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(sentence for _, _, sentence in sorted(top, key=lambda item: item[1]))


def extract_keywords(text: str, limit: int = 8) -> str:
    """Return the most frequent meaningful words."""
    if not text or not text.strip():
        return "No text provided."

    limit = max(1, min(int(limit), 20))
    words = [
        word
        for word in _words(text)
        if word not in STOPWORDS and len(word) > 2
    ]
    if not words:
        return "No keywords found."

    common = Counter(words).most_common(limit)
    return ", ".join(f"{word} ({count})" for word, count in common)


def generate_ideas(topic: str, count: int = 5) -> str:
    """Generate practical starter ideas for a topic."""
    if not topic or not topic.strip():
        return "No topic provided."

    count = max(1, min(int(count), 10))
    templates = [
        "Build a small {topic} prototype and test it with 5 users.",
        "Write a one-page beginner guide explaining {topic}.",
        "Compare 3 existing tools for {topic} and list their trade-offs.",
        "Create a checklist people can use before trying {topic}.",
        "Record a 2-minute demo showing one real use case of {topic}.",
        "Collect 10 common questions about {topic} and answer them.",
        "Make a simple cost and time estimate for a {topic} project.",
        "Design a tiny experiment to measure whether {topic} actually helps.",
    ]

    ideas = [
        templates[(index - 1) % len(templates)].format(topic=topic.strip())
        for index in range(1, count + 1)
    ]
    return "\n".join(f"{index}. {idea}" for index, idea in enumerate(ideas, start=1))


def word_count(text: str) -> str:
    """Return basic text statistics."""
    if not text or not text.strip():
        return "No text provided."

    words = len(_WORD_RE.findall(text))
    sentences = len(_sentences(text))
    return f"Words: {words}\nCharacters: {len(text)}\nSentences: {sentences}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize a longer piece of text into a few key sentences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize."},
                    "max_sentences": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                        "description": "Maximum number of sentences in the summary.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_keywords",
            "description": "Extract the most frequent meaningful keywords from text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 8,
                        "description": "Maximum number of keywords.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_ideas",
            "description": "Generate practical starter ideas for a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to brainstorm."},
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "Number of ideas to generate.",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Count words, characters, and sentences in a piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to measure."},
                },
                "required": ["text"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "summarize_text": summarize_text,
    "extract_keywords": extract_keywords,
    "generate_ideas": generate_ideas,
    "word_count": word_count,
}
