#!/usr/bin/env bash
# Create the two isolated environments. heretic needs a CUDA torch; the executorch wheel pins its own
# (CPU) torch, so they cannot share a venv.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-$HOME/.local/bin/uv}

# triton (pulled in by torch) JIT-compiles a small C driver at first use and needs Python headers + a compiler
if ! [ -f /usr/include/python3.12/Python.h ]; then sudo apt-get install -y -q python3-dev build-essential; fi

echo "== .venv-heretic (heretic + CUDA torch)"
$UV venv --python 3.12 .venv-heretic -q
$UV pip install --python .venv-heretic/bin/python -r requirements-heretic.txt
.venv-heretic/bin/python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list()[-3:])
import heretic, importlib.metadata as m; print("heretic", m.version("heretic-llm"))
PY

echo "== .venv-et (executorch, CPU)"
$UV venv --python 3.12 .venv-et -q
$UV pip install --python .venv-et/bin/python -r requirements-executorch.txt
.venv-et/bin/python -c "import importlib.metadata as m, executorch.examples.models.lfm2; print('executorch', m.version('executorch'), 'lfm2 example ok')"

# exact resolved versions, for the record
$UV pip freeze --python .venv-heretic/bin/python > requirements-heretic.lock.txt
$UV pip freeze --python .venv-et/bin/python > requirements-executorch.lock.txt
echo "setup done"
