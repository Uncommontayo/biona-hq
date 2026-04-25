"""LLM-based semantic labeling pipeline for Biona Lab.

IMPORTANT: This pipeline runs at TRAINING TIME ONLY on servers.
It NEVER runs on-device. Cloud LLM inference is appropriate here.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import jsonschema
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

INTENT_TAXONOMY = [
    "schedule_meeting",
    "request_info",
    "complaint",
    "confirm",
    "cancel",
    "provide_info",
    "greeting",
    "farewell",
    "clarify",
    "other",
]

LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "intent",
        "topic",
        "entities",
        "domain",
        "confidence",
        "labeler_model",
        "labeler_version",
        "schema_version",
    ],
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": INTENT_TAXONOMY,
        },
        "topic": {
            "type": "array",
            "items": {"type": "string"},
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "type"],
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                },
            },
        },
        "domain": {
            "type": "string",
            "enum": ["medical", "legal", "finance", "general"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "labeler_model": {"type": "string"},
        "labeler_version": {"type": "string"},
        "schema_version": {
            "type": "string",
            "const": "1.0",
        },
    },
}

_LABELER_VERSION = "1.0"
_SCHEMA_VERSION  = "1.0"

_SYSTEM_PROMPT = (
    "You are a semantic labeling assistant for speech data. "
    "Given a transcript, extract: intent (from the fixed taxonomy), "
    "topics (free list), named entities with types, and domain. "
    "Respond ONLY with valid JSON matching the exact schema provided. "
    "Do not add any fields not in the schema. Do not explain your reasoning."
)

_USER_PROMPT_TEMPLATE = """\
Label the following transcript. Respond with a JSON object containing exactly these fields:
- intent: one of {intents}
- topic: list of strings
- entities: list of {{"text": string, "type": string}}
- domain: one of "medical", "legal", "finance", "general"
- confidence: float 0.0-1.0
- labeler_model: "{model}"
- labeler_version: "{version}"
- schema_version: "1.0"

Transcript: {transcript}"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_label(label: dict) -> bool:
    """Validate a label dict against LABEL_SCHEMA. Returns True or raises."""
    jsonschema.validate(instance=label, schema=LABEL_SCHEMA)
    return True


# ---------------------------------------------------------------------------
# PII Redaction
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    # US phone numbers
    (re.compile(
        r"\b(?:\+1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    ), "[REDACTED_PHONE]"),
    # Email addresses
    (re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ), "[REDACTED_EMAIL]"),
    # US SSN  xxx-xx-xxxx
    (re.compile(
        r"\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b"
    ), "[REDACTED_SSN]"),
    # Credit card  4 groups of 4 digits, various separators
    (re.compile(
        r"\b(?:\d{4}[\s\-]){3}\d{4}\b"
    ), "[REDACTED_CC]"),
]


class PIIRedactor:
    """Redacts common PII patterns before sending text to cloud LLM."""

    def redact(self, text: str) -> str:
        for pattern, replacement in _PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class LabelingError(Exception):
    """Raised when max_retries is exhausted without a valid label."""


# ---------------------------------------------------------------------------
# Semantic labeler
# ---------------------------------------------------------------------------

class SemanticLabeler:
    """
    Labels a single transcript using an OpenAI chat model.

    Mandatory PII redaction runs before any text is sent to the cloud.
    Validates every response against LABEL_SCHEMA; retries on failure.
    """

    def __init__(self, model: str = "gpt-4o", max_retries: int = 3) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI()
        self._redactor = PIIRedactor()

    def label(self, transcript: str) -> dict:
        """
        Label one transcript.

        1. Redact PII (mandatory).
        2. Call LLM with JSON mode.
        3. Validate against LABEL_SCHEMA.
        4. Retry up to max_retries on schema violation.
        5. Raise LabelingError after exhaustion.
        """
        redacted = self._redactor.redact(transcript)

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            intents=", ".join(f'"{i}"' for i in INTENT_TAXONOMY),
            model=self.model,
            version=_LABELER_VERSION,
            transcript=redacted,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw = response.choices[0].message.content or ""
                label = json.loads(raw)
                validate_label(label)
                return label

            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "Labeling attempt %d/%d failed schema validation: %s",
                    attempt, self.max_retries, exc,
                )
                time.sleep(0.5 * attempt)  # brief back-off

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Labeling attempt %d/%d raised unexpected error: %s",
                    attempt, self.max_retries, exc,
                )
                time.sleep(1.0 * attempt)

        raise LabelingError(
            f"Failed to label transcript after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Batch labeler
# ---------------------------------------------------------------------------

_REVIEW_SAMPLE_RATE = 0.05   # 5% flagged for human review


class BatchLabeler:
    """Labels a full shard directory and writes parallel .jsonl output files."""

    def __init__(self, model: str = "gpt-4o", max_retries: int = 3) -> None:
        self._labeler = SemanticLabeler(model=model, max_retries=max_retries)

    def label_dataset(
        self,
        shard_dir: Path,
        output_dir: Path,
        sample_rate: float = 1.0,
    ) -> None:
        """
        Label transcripts from every shard in shard_dir.

        Writes:
          <output_dir>/<shard_stem>.jsonl  — one label per line
          <output_dir>/review_queue.jsonl  — 5% randomly flagged for review

        Never aborts on a single failure — logs and continues.
        Prints a final summary report.
        """
        import io
        import webdataset as wds
        from tqdm import tqdm

        shard_dir  = Path(shard_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shard_paths = sorted(shard_dir.glob("shard-*.tar"))
        if not shard_paths:
            logger.warning("No shards found in %s", shard_dir)
            return

        total_labeled  = 0
        total_failed   = 0
        total_flagged  = 0
        review_records: list[dict] = []

        for shard_path in shard_paths:
            shard_stem = shard_path.stem
            out_path   = output_dir / f"{shard_stem}.jsonl"

            dataset = (
                wds.WebDataset(str(shard_path))
                .decode()
            )

            with out_path.open("w", encoding="utf-8") as out_fh:
                pbar = tqdm(
                    desc=f"Labeling {shard_stem}",
                    unit="samples",
                    dynamic_ncols=True,
                    leave=False,
                )
                t0 = time.monotonic()

                for sample in dataset:
                    # Sample-rate sub-sampling
                    if sample_rate < 1.0 and random.random() > sample_rate:
                        continue

                    key = sample.get("__key__", "")
                    transcript_raw = sample.get("text.txt", b"")
                    if isinstance(transcript_raw, bytes):
                        transcript = transcript_raw.decode("utf-8").strip()
                    else:
                        transcript = str(transcript_raw).strip()

                    if not transcript:
                        pbar.update(1)
                        continue

                    try:
                        label = self._labeler.label(transcript)
                        label["__key__"] = key
                        out_fh.write(json.dumps(label, ensure_ascii=False) + "\n")
                        total_labeled += 1

                        # Flag for human review
                        if random.random() < _REVIEW_SAMPLE_RATE:
                            review_records.append({"__key__": key, "transcript": transcript, **label})
                            total_flagged += 1

                    except LabelingError as exc:
                        logger.error("Failed to label sample '%s': %s", key, exc)
                        total_failed += 1

                    except Exception as exc:
                        logger.error("Unexpected error for sample '%s': %s", key, exc)
                        total_failed += 1

                    elapsed = time.monotonic() - t0
                    pbar.set_postfix(
                        rate=f"{(total_labeled + total_failed) / max(elapsed, 1):.1f}/s",
                        failed=total_failed,
                        refresh=False,
                    )
                    pbar.update(1)

                pbar.close()

        # Write review queue
        if review_records:
            review_path = output_dir / "review_queue.jsonl"
            with review_path.open("w", encoding="utf-8") as fh:
                for rec in review_records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("Review queue written: %s (%d samples)", review_path, total_flagged)

        # Final report
        logger.info(
            "Labeling complete — labeled: %d | failed: %d | flagged for review: %d",
            total_labeled, total_failed, total_flagged,
        )
