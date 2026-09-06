# lfm2_5_2_6b — run summary (reconstructed from session capture; raw heretic.log lost with the VM)

- Date: 2026-09-06, RTX PRO 6000 Blackwell 96 GB, heretic @3521f86, torch 2.14.0+cu130, default config.toml
- Model: LiquidAI/LFM2.5-2.6B (instruct; thinking model, emits a reasoning block then `</think>`), bf16, 2 shards
- ~2 s per trial; 200 trials in ~8 min (16:20 → 16:28 UTC)
- Baseline: Refusals 99/100
- Early trials seen: t4 97/100 KL 0.0473, t5 44/100 KL 0.0308

Pareto front:

idx trial refusals       KL  direction_index
  0   171     0.03   0.1522  None
  1   192     0.04   0.0806  None
  2   191     0.05   0.0801  None
  3   197     0.06   0.0648  None
  4     8     0.07   0.0343  None                 <- selected (knee of the front)
  5   133     0.08   0.0238  17.379784537320543
  6    92     0.11   0.0225  None
  7    77     0.23   0.0210  None
  8   158     0.24   0.0103  17.028759195631267
  9    89     0.45   0.0051  17.485544563435877
 10    45     0.79   0.0046  17.737500883050494
 11   113     0.89   0.0035  16.83671987276088
 12   194     0.96   0.0020  17.497441205715695
 13   101     0.97   0.0016  18.074113894983057
 14    97     0.98   0.0009  17.811708539059808
 15   187     0.99   0.0008  11.863942700812128

Phase 2: merged save of index 4 → models/LFM2.5-2.6B-heretic (model-00001-of-00002.safetensors 4,998,013,464 B + model-00002-of-00002.safetensors 396,413,952 B).

Sanity (greedy, max 120 tokens): abliterated model still reasons then answers; tool call intact:
`...Let me call this function to get the current weather.</think><|tool_call_start|>[get_weather(city='Manila')]<|tool_call_end|><|im_end|>`

Export: control export of the original weights passed first (validates configs/lfm2_5_2_6b_params.json, bos 124894 / eos 124900).
Abliterated: export/lfm2_5_2_6b_heretic_8da4w.pte, 2,445,033,088 B (about 1 GB of it is the fp32 128k-entry embedding table).
CPU run (python runner, box): coherent reasoning + answer; prefill 2.86 s, 81 tok/s decode.

Published 2026-09-06 to alpharomercoma/LFM2.5-2.6B-heretic (public).
