#!/usr/bin/env bash
# Phase 2: resume the finished study, restore Pareto trial <idx>, merge the abliteration into the weights and
# save a plain HF checkpoint (safetensors + tokenizer). Driven through heretic's own menus (see _pty_phase2.py).
# Usage: scripts/03_save_model.sh <hf_model_id> <run_dir> <pareto_index> <output_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; RUN="$2"; IDX="$3"; OUT="$(realpath -m "$4")"
mkdir -p "$OUT"
cd "$RUN"
../../.venv-heretic/bin/python ../../scripts/_pty_phase2.py "$IDX" "$OUT" -- \
  ../../.venv-heretic/bin/heretic --checkpoint-action continue --model "$MODEL"
../../.venv-heretic/bin/python ../../scripts/02_pareto.py . > "$OUT/heretic_pareto_front.txt"
echo "selected pareto index: $IDX" >> "$OUT/heretic_pareto_front.txt"
ls -la "$OUT"
