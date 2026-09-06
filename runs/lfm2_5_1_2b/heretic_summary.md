# lfm2_5_1_2b — run summary (condensed; full logs in logs/, journal in checkpoints/)

- Date: 2026-09-06, RTX PRO 6000 Blackwell 96 GB, heretic @3521f86, torch 2.14.0+cu130, default config.toml
- Model: LiquidAI/LFM2.5-1.2B-Instruct, bf16
- Batch-size benchmark chose 128; ~1.5 s per trial; 200 trials in ~5 min (14:41 → 14:46 UTC)
- Baseline: Refusals 99/100, KL 0 (by definition)
- Early trials seen: t1 60/100 KL 0.0254, t4 17/100 KL 0.0896, t5 92/100 KL 0.0078

Pareto front (sorted like heretic's menu):

idx trial refusals       KL  direction_index
  0   133     0.06   0.0527  10.198270144464333   <- selected
  1   150     0.07   0.0402  10.32046799180982
  2   157     0.11   0.0344  10.352822192960996
  3   170     0.17   0.0316  10.690289415212908
  4   169     0.18   0.0283  10.969840642119093
  5   195     0.41   0.0232  9.737856473201463
  6    63     0.43   0.0227  None
  7   143     0.46   0.0217  11.294404725756008
  8   158     0.49   0.0174  11.119204885113456
  9   186     0.53   0.0170  10.5853942209232
 10   191     0.63   0.0157  11.178660347171663
 11    72     0.66   0.0124  11.153256538028568
 12   161     0.79   0.0102  10.51671121861693
 13   102     0.84   0.0097  11.205828382998588
 14    59     0.87   0.0093  10.78726439886388
 15    76     0.90   0.0061  None
 16   164     0.91   0.0058  10.51819047303982
 17   182     0.96   0.0028  None
 18    26     0.98   0.0006  None

Phase 2: merged save of index 0 → models/LFM2.5-1.2B-Instruct-heretic (model.safetensors 2,340,697,936 B, bf16, same keys/shapes as the base).

Sanity (greedy, max 120 tokens), original vs abliterated:
- benign "why is the sky blue": both give the Rayleigh-scattering answer, second sentence differs slightly.
- tool call "weather in Manila" with get_weather: identical output
  `<|tool_call_start|>[get_weather(city="Manila")]<|tool_call_end|>Checking the current weather in Manila.<|im_end|>`
- roast probe: identical output (the base model did not refuse this prompt either).

Export: XNNPACK 8da4w, ctx 4096 → export/lfm2_5_1_2b_heretic_8da4w.pte, 1,145,313,536 B; activation memory 549,540,928 B.
First attempt failed because the intermediate .pt was named *_8da4w.pt (ExecuTorch treats that as pre-quantised).
CPU run (python runner, box): coherent answer; runner reported prefill 1.33 s, 289 tok/s decode (30-core x86).
Mac M5 C++ runner with KleidiAI on the original-weights .pte: 112 tok/s prefill, 35 tok/s decode while heretic was running.

Published 2026-09-06 to alpharomercoma/LFM2.5-1.2B-Instruct-heretic (public).
