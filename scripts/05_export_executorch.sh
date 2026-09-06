#!/usr/bin/env bash
# HF checkpoint dir -> Meta-format .pt -> ExecuTorch .pte for CPU via the XNNPACK backend, quantised
# 8-bit dynamic activations / 4-bit grouped weights (8da4w). The .pte is portable: on Arm hosts
# (Apple Silicon, Android, Graviton) XNNPACK runs these matmuls on KleidiAI kernels; on x86 it uses AVX kernels.
# Usage: scripts/05_export_executorch.sh <hf_dir> <output_name> <params.json> <export_llm.yaml> [max_context]
# NB: the intermediate Meta-format checkpoint must not have "8da4w" or "int8" in its path: ExecuTorch's loader
# takes that as "pre-quantised checkpoint" and wraps the linears before quantising, which breaks export.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="$(realpath "$1")"; NAME="$2"; PARAMS="$(realpath "$3")"; CFG="$(realpath "$4")"; CTX="${5:-4096}"
PY="$PWD/.venv-et/bin/python"; OUT="$PWD/export"; mkdir -p "$OUT"

echo "== convert HF safetensors -> Meta format"
CKPT="$OUT/${NAME//8da4w/q4}_meta.pt"
$PY scripts/_convert_lfm2.py "$SRC" "$CKPT"

echo "== export_llm (XNNPACK, 8da4w, ctx=$CTX)"
cd "$OUT"
$PY -m executorch.extension.llm.export.export_llm \
  --config "$CFG" \
  +base.model_class="lfm2_5_1_2b" \
  +base.params="$PARAMS" \
  +base.checkpoint="$CKPT" \
  +export.max_seq_length="$CTX" \
  +export.max_context_length="$CTX" \
  +export.output_name="$NAME.pte" 2>&1 | grep -v -E "Warning|_check_is_size|register_constant"
rm -f "$CKPT"
ls -la "$OUT/$NAME.pte"
