#!/usr/bin/env bash
# Download and extract LibriSpeech splits from openslr.org
#
# Usage: bash download_librispeech.sh <split> [split ...] <output_dir>
# Example: bash download_librispeech.sh train-clean-100 /data/librispeech
#
# At least one split and the output directory must be provided.
# The output directory is always the last argument.

set -euo pipefail

OPENSLR_BASE="https://www.openslr.org/resources/12"

VALID_SPLITS=(
    "train-clean-100"
    "train-clean-360"
    "train-other-500"
    "dev-clean"
    "dev-other"
    "test-clean"
    "test-other"
)

usage() {
    echo "Usage: $0 <split> [split ...] <output_dir>"
    echo ""
    echo "Valid splits:"
    for s in "${VALID_SPLITS[@]}"; do
        echo "  $s"
    done
    exit 1
}

# Need at least 2 args (one split + output dir)
if [[ $# -lt 2 ]]; then
    usage
fi

# Last argument is the output directory; everything before it is splits
args=("$@")
OUTPUT_DIR="${args[-1]}"
SPLITS=("${args[@]:0:${#args[@]}-1}")

mkdir -p "$OUTPUT_DIR"

is_valid_split() {
    local split="$1"
    for s in "${VALID_SPLITS[@]}"; do
        [[ "$s" == "$split" ]] && return 0
    done
    return 1
}

for split in "${SPLITS[@]}"; do
    if ! is_valid_split "$split"; then
        echo "ERROR: Unknown split '$split'"
        usage
    fi

    # Map split name to tar.gz filename on OpenSLR
    filename="${split}.tar.gz"
    url="${OPENSLR_BASE}/${filename}"
    dest="${OUTPUT_DIR}/${filename}"

    echo "==> Downloading $split from $url"
    if [[ -f "$dest" ]]; then
        echo "    Archive already exists, skipping download: $dest"
    else
        curl -L --progress-bar -o "$dest" "$url"
    fi

    echo "==> Extracting $split to $OUTPUT_DIR"
    tar -xzf "$dest" -C "$OUTPUT_DIR"
    echo "    Done: $split"
done

echo ""
echo "All requested splits downloaded and extracted to: $OUTPUT_DIR"
