# LFM2.5-1.2B-Instruct → heretic abliteration → ExecuTorch CPU export

Reproducible pipeline that removes the refusal behaviour from `LiquidAI/LFM2.5-1.2B-Instruct` with
[heretic](https://github.com/p-e-w/heretic), exports the result to an ExecuTorch `.pte` for CPU inference
(XNNPACK backend, 8da4w quantisation, KleidiAI kernels on Arm), and publishes both to the Hugging Face Hub.

Built and run on an RTX PRO 6000 Blackwell (96 GB), Ubuntu 24.04, Python 3.12. System packages: `python3-dev build-essential`
(triton compiles a small driver stub at first use); `uv` for the venvs.

## Results

| size | published model | selected trial | refusals | KL | ExecuTorch .pte |
|---|---|---|---|---|---|
| 1.2B | [alpharomercoma/LFM2.5-1.2B-Instruct-heretic](https://huggingface.co/alpharomercoma/LFM2.5-1.2B-Instruct-heretic) | Pareto 0 (trial 133) | 99 → 6 /100 | 0.053 | 1.15 GB, `executorch/` in the repo |
| 2.6B | [alpharomercoma/LFM2.5-2.6B-heretic](https://huggingface.co/alpharomercoma/LFM2.5-2.6B-heretic) | Pareto 4 (trial 8) | 99 → 7 /100 | 0.034 | 2.45 GB, `executorch/` in the repo |

Both fronts are in `runs/<run>/heretic_pareto_front.txt`. Tool calling was verified intact on both
(`make sanity`). `requirements-*.lock.txt` are written by `make setup` on the target machine.

## Why these choices

- **Instruct, not Base.** Abliteration removes a direction that alignment training installed; the base model has
  no refusal direction to remove and no chat template / tool-call tokens, which the downstream chat app needs.
- **heretic as shipped, pinned to a commit.** The unattended CLI flags are newer than the 1.4.0 PyPI release, so
  `requirements-heretic.txt` pins the git commit. Default `config.default.toml` (200 TPE trials, 60 random start-up trials, refusal rate
  and KL divergence as the two objectives, `row_normalization = "full"`). Nothing about the method is modified;
  the only automation is a pty driver so the run is unattended and a non-interactive phase 2 that saves the chosen
  Pareto trial as merged weights.
- **8da4w on XNNPACK.** The quantisation scheme ExecuTorch's LFM2 example targets; it is what XNNPACK routes to
  KleidiAI on Arm CPUs and what keeps the model around 1.1 GB.

## Gotchas worth knowing

- heretic resumes a checkpointed study with the *stored* settings, so `--trial-index`, `--model-action` and
  `--save-directory` are ignored on resume. Phase 2 therefore drives heretic's own menus through a pty
  (`scripts/_pty_phase2.py`) instead of forking heretic.
- heretic rewrites a trailing positional argument into `--model`; always pass `--model <id>` explicitly.
- ExecuTorch's llama loader treats any checkpoint path containing `8da4w` or `int8` as a pre-quantised checkpoint.
  The intermediate `.pt` is therefore named without that suffix.
- The `executorch` wheel ships the LFM2 model code but not its params JSON; `configs/lfm2_5_1_2b_params.json`
  is the copy from the v1.4.1 tag.
- triton (pulled in by torch) compiles a driver stub at first use: `python3-dev build-essential` are required.

## Layout

```
Makefile                      one target per stage (make help)
configs/heretic.toml          heretic configuration used for the run (copied into the run dir)
configs/lfm2_5_<size>_params.json   Meta-format model args for the ExecuTorch LFM2 example
configs/lfm2_5_<size>_xnnpack_q8da4w.yaml  export_llm config (XNNPACK, 8da4w, kv-cache, sdpa, bos/eos ids)
scripts/00_setup.sh           two venvs: .venv-heretic (CUDA torch) and .venv-et (executorch, CPU)
scripts/01_abliterate.sh      phase 1: heretic optimisation, checkpointed in runs/<run>/checkpoints
scripts/02_pareto.py          print the Pareto front, sorted like heretic's menu
scripts/03_save_model.sh      phase 2: restore trial <idx>, merge, save HF checkpoint to models/
scripts/04_sanity_check.py    original vs abliterated: benign, tool call, refusal probe
scripts/_convert_lfm2.py      shard-aware HF -> Meta-format conversion (ExecuTorch's reads one file only)
scripts/05_export_executorch.sh  HF dir -> .pte
scripts/06_run_pte.sh         generate with the .pte on CPU
scripts/07_publish_hf.py      upload model + .pte + model card to the Hub
requirements-*.txt / *.lock.txt  inputs and the exact resolved versions
```

## Run

Every target takes `SIZE=1_2b` (default, `LiquidAI/LFM2.5-1.2B-Instruct`) or `SIZE=2_6b` (`LiquidAI/LFM2.5-2.6B`,
which is the instruct model of that size; `-Base` is the pretrained one). Run and output paths derive from it.

```sh
make setup
make abliterate                 # ~10 min on the RTX 6000 (200 trials)
make pareto                     # inspect the front
make save TRIAL=<idx>           # merged HF checkpoint in models/LFM2.5-1.2B-Instruct-heretic
make sanity                     # compare original vs abliterated on 3 prompts
make export                     # export/lfm2_5_1_2b_heretic_8da4w.pte
make run-pte                    # run it on CPU
cp .env.example .env            # HF_TOKEN + HF_REPO_ID
make publish                    # HF_REPO_ID=user/name make publish SIZE=2_6b to override the .env repo
```

Choosing the trial: the front is sorted by refusal rate then KL. Index 0 is the least refusing; move down the list
for lower KL (closer to the original model). For a chat app with tools, prefer the lowest index whose KL is still
small (roughly < 0.1) and confirm with `make sanity` that tool calls are intact.
