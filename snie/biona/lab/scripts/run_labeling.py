"""CLI: run the semantic labeling pipeline over a shard directory.

Usage:
    python scripts/run_labeling.py \\
        --shards /data/shards \\
        --output /data/labels

Optional:
    --model gpt-4o          (default: gpt-4o)
    --sample-rate 0.1       (label 10% of samples; default: 1.0 = all)
    --max-retries 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semantic labeling over Biona Lab WebDataset shards."
    )
    parser.add_argument(
        "--shards",
        type=Path,
        default=Path(os.environ.get("SHARD_OUTPUT_DIR", "/data/shards")),
        help="Directory containing shard-*.tar files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/labels"),
        help="Output directory for .jsonl label files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for labeling (default: gpt-4o)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Fraction of samples to label, 0.0-1.0 (default: 1.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max LLM retries per sample on schema violation (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not args.shards.exists():
        logger.error("Shard directory not found: %s", args.shards)
        sys.exit(1)

    if not (0.0 < args.sample_rate <= 1.0):
        logger.error("--sample-rate must be in (0.0, 1.0]")
        sys.exit(1)

    from biona_lab.labeling.semantic_labeler import BatchLabeler

    labeler = BatchLabeler(model=args.model, max_retries=args.max_retries)
    labeler.label_dataset(
        shard_dir=args.shards,
        output_dir=args.output,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
