#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="$ROOT/../data/English/Interview with Jimmy Wales at Wikimania 2025 Nairobi.webm"
PLAY=(--play)

if [[ "${1:-}" == "--no-play" ]]; then
    PLAY=()
elif [[ -n "${1:-}" ]]; then
    echo "usage: $0 [--no-play]" >&2
    exit 2
fi

echo "Running the English interviewer/guest demo (SPEAKER_01 → SPEAKER_00)." >&2
echo "Model startup can take several seconds; the server is stopped automatically." >&2

cd "$ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run --no-capture-output -n multimodalsensorfusion \
    python tools/stream_video.py "$VIDEO" \
    --start-server --quiet-server --language en --start-at 90 --duration 35 \
    "${PLAY[@]}"
