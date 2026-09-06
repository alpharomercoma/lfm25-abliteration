"""Publish the abliterated checkpoint to the Hugging Face Hub: safetensors + tokenizer, the ExecuTorch .pte
under executorch/, the heretic Pareto front, and a generated model card.
Usage: HF_TOKEN=... HF_REPO_ID=user/name python scripts/07_publish_hf.py <model_dir> [--base <hf_id>] [--pte <file>] [--run <run_dir>] [--private]"""
import argparse, os, pathlib, subprocess, sys
from huggingface_hub import HfApi

def model_card(BASE, repo_id, front, pte_name):
    return f"""---
base_model: {BASE}
license: other
license_name: lfm1.0
license_link: https://huggingface.co/{BASE}/blob/main/LICENSE
library_name: transformers
tags: [lfm2, abliterated, heretic, executorch, xnnpack]
---

# {repo_id}

Abliterated ("uncensored") version of [{BASE}](https://huggingface.co/{BASE}), produced with
[heretic](https://github.com/p-e-w/heretic): directional ablation of the refusal direction in the residual
stream, with a TPE search over per-layer ablation weights that minimises both the refusal rate and the
KL divergence from the original model. Weights are merged, so this loads as a plain `Lfm2ForCausalLM`.
Chat template, tool-calling tokens and everything else are unchanged from the base model.

## heretic Pareto front for this run

```
{front}
```

## ExecuTorch (CPU)

`executorch/{pte_name}` is an ExecuTorch program exported for the XNNPACK backend with 8-bit dynamic
activations / 4-bit grouped weights (`8da4w`); on Arm hosts XNNPACK dispatches these to KleidiAI kernels.
Run it with the `llama_main` runner from ExecuTorch (`examples/models/llama`) using this repo's `tokenizer.json`
and the prompt format `<|startoftext|><|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n`.

## Pipeline

Reproducible end to end from https://github.com/ (see the repository this was built with):
`make setup abliterate pareto`, `make save TRIAL=<idx>`, `make export`, `make publish`.

Use responsibly; the safety training of the base model has been removed on purpose.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--pte")
    ap.add_argument("--run", default="runs/lfm2_5_1_2b")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--base", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID"))
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token or not a.repo_id:
        sys.exit("set HF_TOKEN and HF_REPO_ID (see .env.example)")
    model_dir = pathlib.Path(a.model_dir)
    front_file = model_dir / "heretic_pareto_front.txt"
    front = front_file.read_text() if front_file.exists() else "(not recorded)"
    pte_name = pathlib.Path(a.pte).name if a.pte else "(none)"

    api = HfApi(token=token)
    api.create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    (model_dir / "README.md").write_text(model_card(a.base, a.repo_id, front, pte_name))
    print(f"uploading {model_dir} -> {a.repo_id}")
    api.upload_folder(repo_id=a.repo_id, folder_path=str(model_dir), commit_message="heretic abliteration of " + a.base)
    if a.pte:
        print(f"uploading {a.pte} -> executorch/{pte_name}")
        api.upload_file(repo_id=a.repo_id, path_or_fileobj=a.pte, path_in_repo=f"executorch/{pte_name}",
                        commit_message="ExecuTorch XNNPACK 8da4w export")
    print(f"done: https://huggingface.co/{a.repo_id}")

if __name__ == "__main__":
    main()
