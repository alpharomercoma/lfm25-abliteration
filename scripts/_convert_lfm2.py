"""HF LFM2 checkpoint dir (single or sharded safetensors) -> Meta-format .pt for ExecuTorch's LFM2 example.
ExecuTorch's own convert_weights only reads a single model.safetensors; the 2.6B checkpoint is sharded."""
import glob, sys, torch
from safetensors.torch import load_file
from executorch.examples.models.lfm2.convert_weights import lfm_2_to_meta
src, dst = sys.argv[1], sys.argv[2]
sd = {}
for f in sorted(glob.glob(f"{src}/*.safetensors")):
    sd.update(load_file(f))
print(f"loaded {len(sd)} tensors from {src}")
torch.save(lfm_2_to_meta(sd), dst)
print(f"saved {dst}")
