"""Tests for label schema validation and PII redaction — biona_lab.labeling."""

from __future__ import annotations

import pytest
import jsonschema

from biona_lab.labeling.semantic_labeler import (
    INTENT_TAXONOMY,
    LABEL_SCHEMA,
    PIIRedactor,
    validate_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_label(**overrides) -> dict:
    """Return a minimal valid label dict, with optional field overrides."""
    base = {
        "intent":          "greeting",
        "topic":           ["introduction"],
        "entities":        [{"text": "Alice", "type": "PERSON"}],
        "domain":          "general",
        "confidence":      0.95,
        "labeler_model":   "gpt-4o",
        "labeler_version": "1.0",
        "schema_version":  "1.0",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema validation — happy path
# ---------------------------------------------------------------------------

def test_valid_label_passes() -> None:
    assert validate_label(_valid_label()) is True


def test_all_intents_accepted() -> None:
    for intent in INTENT_TAXONOMY:
        assert validate_label(_valid_label(intent=intent)) is True


def test_all_domains_accepted() -> None:
    for domain in ("medical", "legal", "finance", "general"):
        assert validate_label(_valid_label(domain=domain)) is True


def test_empty_topic_and_entities() -> None:
    assert validate_label(_valid_label(topic=[], entities=[])) is True


def test_multiple_entities() -> None:
    label = _valid_label(entities=[
        {"text": "Bob", "type": "PERSON"},
        {"text": "Acme Corp", "type": "ORG"},
        {"text": "New York", "type": "GPE"},
    ])
    assert validate_label(label) is True


def test_confidence_boundary_values() -> None:
    assert validate_label(_valid_label(confidence=0.0)) is True
    assert validate_label(_valid_label(confidence=1.0)) is True


# ---------------------------------------------------------------------------
# Schema validation — invalid inputs
# ---------------------------------------------------------------------------

def test_invalid_intent_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate_label(_valid_label(intent="unknown_intent"))


def test_invalid_domain_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate_label(_valid_label(domain="technology"))


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate_label(_valid_label(confidence=1.1))
    with pytest.raises(jsonschema.ValidationError):
        validate_label(_valid_label(confidence=-0.1))


def test_wrong_schema_version_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate_label(_valid_label(schema_version="2.0"))


def test_extra_field_rejected() -> None:
    label = _valid_label()
    label["unexpected_field"] = "value"
    with pytest.raises(jsonschema.ValidationError):
        validate_label(label)


def test_missing_required_field_rejected() -> None:
    label = _valid_label()
    del label["intent"]
    with pytest.raises(jsonschema.ValidationError):
        validate_label(label)


def test_entity_missing_type_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate_label(_valid_label(entities=[{"text": "Alice"}]))


def test_topic_not_list_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate_label(_valid_label(topic="not a list"))


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

def test_redacts_phone_number() -> None:
    r = PIIRedactor()
    assert "[REDACTED_PHONE]" in r.redact("Call me at 555-867-5309")
    assert "[REDACTED_PHONE]" in r.redact("Reach me at (800) 555-1234")
    assert "[REDACTED_PHONE]" in r.redact("+1 415 555 2671")


def test_redacts_email() -> None:
    r = PIIRedactor()
    result = r.redact("Send it to user.name+tag@example.co.uk please")
    assert "[REDACTED_EMAIL]" in result
    assert "user.name" not in result


def test_redacts_ssn() -> None:
    r = PIIRedactor()
    result = r.redact("My SSN is 123-45-6789")
    assert "[REDACTED_SSN]" in result
    assert "123-45-6789" not in result


def test_redacts_credit_card() -> None:
    r = PIIRedactor()
    result = r.redact("Card number 4111-1111-1111-1111 was declined")
    assert "[REDACTED_CC]" in result
    assert "4111" not in result


def test_no_false_positive_on_clean_text() -> None:
    r = PIIRedactor()
    clean = "Please schedule a meeting for Monday at nine am"
    assert r.redact(clean) == clean


def test_multiple_pii_types_in_one_string() -> None:
    r = PIIRedactor()
    text = "Email john@example.com or call 800-555-0199"
    result = r.redact(text)
    assert "[REDACTED_EMAIL]" in result
    assert "[REDACTED_PHONE]" in result
    assert "john@example.com" not in result
    assert "800-555-0199" not in result
