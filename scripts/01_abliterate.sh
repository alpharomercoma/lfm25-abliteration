#!/usr/bin/env bash
# Phase 1: heretic optimisation (TPE over per-layer ablation weights), fully unattended.
# heretic checkpoints every trial to <run>/checkpoints and ends in an interactive trial menu, so we run it
# under a pty and stop it there; phase 2 (03_save_model.sh) resumes the study and saves the chosen trial.
# Usage: scripts/01_abliterate.sh <hf_model_id> <run_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; RUN="$2"
mkdir -p "$RUN"
cp -f configs/heretic.toml "$RUN/config.toml"      # heretic reads ./config.toml from its cwd
cd "$RUN"
exec ../../.venv-heretic/bin/python ../../scripts/_pty_driver.py \
  ../../.venv-heretic/bin/heretic --checkpoint-action restart --model "$MODEL"
