#!/usr/bin/env bash
#
# publish_release.sh - cut the yeGPT v0.1.0 GitHub release and upload the
# distributable weights + model card.
#
# AUTHOR-RUN ONLY. This script pushes a public release to GitHub. The autonomous
# loop must NEVER execute it (it publishes, which the loop is forbidden to do).
# Run it by hand once the fp16 artifact has been built with:
#
#     .venv/Scripts/python scripts/export_checkpoint.py
#
# Requires an authenticated `gh` CLI (`gh auth status`).

set -euo pipefail

tag="v0.1.0"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

artifact="$repo_root/dist/yegpt-small-fp16.pt"
model_card="$repo_root/MODEL_CARD.md"

if [[ ! -f "$artifact" ]]; then
    echo "error: release artifact not found: $artifact" >&2
    echo "build it first: .venv/Scripts/python scripts/export_checkpoint.py" >&2
    exit 1
fi

if [[ ! -f "$model_card" ]]; then
    echo "error: model card not found: $model_card" >&2
    exit 1
fi

gh release create "$tag" \
    "$artifact" \
    "$model_card" \
    --title "yeGPT small $tag" \
    --notes "Character-level yeGPT (small, 1.87M params). See MODEL_CARD.md for architecture, training data, eval, sampling knobs, limitations, and the weights/parody notice. AI-generated parody / educational project; not affiliated with or endorsed by Kanye West."
