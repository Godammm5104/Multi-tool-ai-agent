from tools import (
    extract_keywords,
    generate_ideas,
    summarize_text,
    word_count,
)


def test_summarize_keeps_short_text_intact():
    assert summarize_text("One idea. Another idea.") == "One idea. Another idea."


def test_summarize_limits_longer_text():
    text = (
        "Distributed systems fail in partial ways. "
        "Timeouts and retries improve reliability. "
        "Idempotency prevents duplicate writes. "
        "Observability explains strange failures."
    )

    summary = summarize_text(text, max_sentences=2)

    assert summary.count(".") <= 2
    assert summary


def test_summarize_empty_text():
    assert summarize_text("   ") == "No text provided."


def test_extract_keywords_orders_by_frequency():
    result = extract_keywords("python python sql sql sql testing", limit=2)

    assert result == "sql (3), python (2)"


def test_extract_keywords_empty_text():
    assert extract_keywords("") == "No text provided."


def test_generate_ideas_respects_count():
    result = generate_ideas("urban gardening", count=3)
    lines = result.splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("1. ")
    assert "urban gardening" in lines[0]


def test_generate_ideas_empty_topic():
    assert generate_ideas("") == "No topic provided."


def test_word_count_returns_basic_stats():
    result = word_count("Hello world. This is fine.")

    assert "Words: 5" in result
    assert "Sentences: 2" in result
