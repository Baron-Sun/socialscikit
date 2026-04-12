"""Tests for socialscikit.qualikit.deidentifier."""

from __future__ import annotations

import pytest

from socialscikit.qualikit.deidentifier import (
    Deidentifier,
    DeidentResult,
    ReplacementRecord,
    ReplacementStrategy,
    SUPPORTED_ENTITIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def deident():
    return Deidentifier()


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_replacement_record(self):
        rec = ReplacementRecord(
            text_id=0, original_span="test@example.com",
            replacement="[EMAIL_1]", entity_type="EMAIL",
            confidence=0.95, position=(10, 26),
        )
        assert rec.entity_type == "EMAIL"
        assert rec.confidence == 0.95

    def test_deident_result(self):
        r = DeidentResult(
            deidentified_texts=["Hello [NAME_1]"],
            replacement_log=[],
            coverage_stats={"PERSON": 1},
        )
        assert len(r.deidentified_texts) == 1

    def test_replacement_strategies(self):
        assert ReplacementStrategy.PLACEHOLDER == "placeholder"
        assert ReplacementStrategy.CATEGORY == "category"
        assert ReplacementStrategy.REDACT == "redact"

    def test_supported_entities(self):
        assert "PERSON" in SUPPORTED_ENTITIES
        assert "EMAIL" in SUPPORTED_ENTITIES
        assert "PHONE" in SUPPORTED_ENTITIES


# ---------------------------------------------------------------------------
# Email detection
# ---------------------------------------------------------------------------


class TestEmailDetection:
    def test_simple_email(self, deident):
        result = deident.process(
            ["Contact me at john@example.com for info."],
            entities=["EMAIL"],
        )
        assert "john@example.com" not in result.deidentified_texts[0]
        assert "[EMAIL_1]" in result.deidentified_texts[0]
        assert result.coverage_stats.get("EMAIL", 0) == 1

    def test_multiple_emails(self, deident):
        result = deident.process(
            ["Email alice@test.org or bob@test.org."],
            entities=["EMAIL"],
        )
        assert "alice@test.org" not in result.deidentified_texts[0]
        assert "bob@test.org" not in result.deidentified_texts[0]
        assert result.coverage_stats["EMAIL"] == 2

    def test_no_email(self, deident):
        result = deident.process(
            ["This text has no personal info."],
            entities=["EMAIL"],
        )
        assert result.deidentified_texts[0] == "This text has no personal info."
        assert result.coverage_stats.get("EMAIL", 0) == 0


# ---------------------------------------------------------------------------
# Phone detection
# ---------------------------------------------------------------------------


class TestPhoneDetection:
    def test_us_phone(self, deident):
        result = deident.process(
            ["Call me at (555) 123-4567 today."],
            entities=["PHONE"],
        )
        assert "(555) 123-4567" not in result.deidentified_texts[0]
        assert "[PHONE_1]" in result.deidentified_texts[0]

    def test_phone_with_dashes(self, deident):
        result = deident.process(
            ["My number is 555-123-4567."],
            entities=["PHONE"],
        )
        assert "555-123-4567" not in result.deidentified_texts[0]

    def test_phone_with_country_code(self, deident):
        result = deident.process(
            ["Reach me at +1-555-123-4567."],
            entities=["PHONE"],
        )
        assert "555-123-4567" not in result.deidentified_texts[0]


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------


class TestURLDetection:
    def test_http_url(self, deident):
        result = deident.process(
            ["Visit https://example.com/page for details."],
            entities=["URL"],
        )
        assert "https://example.com" not in result.deidentified_texts[0]
        assert "[URL_1]" in result.deidentified_texts[0]

    def test_www_url(self, deident):
        result = deident.process(
            ["Go to www.example.org for more."],
            entities=["URL"],
        )
        assert "www.example.org" not in result.deidentified_texts[0]


# ---------------------------------------------------------------------------
# SSN detection
# ---------------------------------------------------------------------------


class TestSSNDetection:
    def test_ssn(self, deident):
        result = deident.process(
            ["SSN: 123-45-6789"],
            entities=["SSN"],
        )
        assert "123-45-6789" not in result.deidentified_texts[0]
        assert "[SSN_1]" in result.deidentified_texts[0]


# ---------------------------------------------------------------------------
# IP detection
# ---------------------------------------------------------------------------


class TestIPDetection:
    def test_ip_address(self, deident):
        result = deident.process(
            ["Server IP is 192.168.1.1 in the logs."],
            entities=["IP_ADDRESS"],
        )
        assert "192.168.1.1" not in result.deidentified_texts[0]
        assert "[IP_1]" in result.deidentified_texts[0]


# ---------------------------------------------------------------------------
# Replacement strategies
# ---------------------------------------------------------------------------


class TestReplacementStrategies:
    def test_placeholder_strategy(self, deident):
        result = deident.process(
            ["Email: alice@test.com"],
            entities=["EMAIL"],
            replacement_strategy="placeholder",
        )
        assert "[EMAIL_1]" in result.deidentified_texts[0]

    def test_category_strategy(self, deident):
        result = deident.process(
            ["Email: alice@test.com"],
            entities=["EMAIL"],
            replacement_strategy="category",
        )
        assert "[EMAIL]" in result.deidentified_texts[0]

    def test_redact_strategy(self, deident):
        result = deident.process(
            ["Email: alice@test.com"],
            entities=["EMAIL"],
            replacement_strategy="redact",
        )
        assert "[REDACTED]" in result.deidentified_texts[0]


# ---------------------------------------------------------------------------
# Multiple texts
# ---------------------------------------------------------------------------


class TestMultipleTexts:
    def test_batch_processing(self, deident):
        texts = [
            "Contact john@example.com for help.",
            "Call 555-123-4567 for support.",
            "No PII here.",
        ]
        result = deident.process(texts, entities=["EMAIL", "PHONE"])
        assert len(result.deidentified_texts) == 3
        assert "john@example.com" not in result.deidentified_texts[0]
        assert "555-123-4567" not in result.deidentified_texts[1]
        assert result.deidentified_texts[2] == "No PII here."

    def test_consistent_replacement(self, deident):
        """Same entity across texts gets same replacement."""
        texts = [
            "Email bob@test.com today.",
            "Also try bob@test.com later.",
        ]
        result = deident.process(texts, entities=["EMAIL"])
        r0 = result.deidentified_texts[0]
        r1 = result.deidentified_texts[1]
        # Both should use the same replacement for bob@test.com
        assert "[EMAIL_1]" in r0
        assert "[EMAIL_1]" in r1

    def test_empty_input(self, deident):
        result = deident.process([])
        assert result.deidentified_texts == []
        assert result.replacement_log == []


# ---------------------------------------------------------------------------
# Replacement log
# ---------------------------------------------------------------------------


class TestReplacementLog:
    def test_log_populated(self, deident):
        result = deident.process(
            ["Email: alice@test.com, phone: 555-123-4567."],
            entities=["EMAIL", "PHONE"],
        )
        assert len(result.replacement_log) >= 2
        types = {r.entity_type for r in result.replacement_log}
        assert "EMAIL" in types
        assert "PHONE" in types

    def test_log_text_id(self, deident):
        result = deident.process(
            ["Text 0: alice@test.com", "Text 1: bob@test.com"],
            entities=["EMAIL"],
        )
        text_ids = {r.text_id for r in result.replacement_log}
        assert 0 in text_ids
        assert 1 in text_ids

    def test_log_position(self, deident):
        text = "Email: alice@test.com here."
        result = deident.process([text], entities=["EMAIL"])
        rec = result.replacement_log[0]
        start, end = rec.position
        assert text[start:end] == "alice@test.com"


# ---------------------------------------------------------------------------
# Format helper
# ---------------------------------------------------------------------------


class TestFormatLog:
    def test_format_log_table(self, deident):
        result = deident.process(
            ["Email: alice@test.com"],
            entities=["EMAIL"],
        )
        rows = Deidentifier.format_log_table(result.replacement_log)
        assert len(rows) == 1
        assert "原文片段" in rows[0]
        assert "替换为" in rows[0]

    def test_format_empty(self):
        rows = Deidentifier.format_log_table([])
        assert rows == []


# ---------------------------------------------------------------------------
# Mixed entity types
# ---------------------------------------------------------------------------


class TestMixedEntities:
    def test_multiple_entity_types(self, deident):
        text = "Call 555-123-4567 or email help@example.com. SSN: 123-45-6789."
        result = deident.process(
            [text],
            entities=["PHONE", "EMAIL", "SSN"],
        )
        out = result.deidentified_texts[0]
        assert "555-123-4567" not in out
        assert "help@example.com" not in out
        assert "123-45-6789" not in out

    def test_selective_entities(self, deident):
        """Only detect specified entity types."""
        text = "Email: test@test.com, Phone: 555-123-4567"
        result = deident.process([text], entities=["EMAIL"])
        out = result.deidentified_texts[0]
        assert "test@test.com" not in out
        # Phone should NOT be replaced since we only asked for EMAIL
        assert "555-123-4567" in out
