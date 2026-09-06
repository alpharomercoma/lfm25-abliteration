#!/usr/bin/env bash
# Generate with the exported .pte on CPU using ExecuTorch's python runner (same runtime as the C++ llama_main).
# Usage: scripts/06_run_pte.sh <hf_dir_with_tokenizer> <pte> <params.json> [prompt]
set -euo pipefail
cd "$(dirname "$0")/.."
HF="$(realpath "$1")"; PTE="$(realpath "$2")"; PARAMS="$(realpath "$3")"
PROMPT="${4:-Give me one sentence about the city of Manila.}"
.venv-et/bin/python -m executorch.examples.models.llama.runner.native \
  --model lfm2_5_1_2b --pte "$PTE" \
  --tokenizer "$HF/tokenizer.json" --tokenizer_config "$HF/tokenizer_config.json" \
  --params "$PARAMS" \
  --prompt "<|startoftext|><|im_start|>user
$PROMPT<|im_end|>
<|im_start|>assistant
" --max_len 160 -kv --temperature 0
