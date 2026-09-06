# Run records

Both runs were executed on 2026-09-06 on an RTX PRO 6000 Blackwell (heretic pinned to commit 3521f86,
default `config.toml`, 200 trials). Each run directory holds the optuna journal (`checkpoints/*.jsonl`, every trial with parameters and scores;
`make pareto` reads it), the heretic config used, the Pareto front with the selected index, a short
`heretic_summary.md`, and `logs/` with the cleaned heretic, save, export and publish logs. `setup.log` is the
environment build. `mac-partial-1_2b/` is the aborted first attempt on the MacBook (MPS).

| run | base model | published | selected | refusals | KL |
|---|---|---|---|---|---|
| `lfm2_5_1_2b` | LiquidAI/LFM2.5-1.2B-Instruct | [alpharomercoma/LFM2.5-1.2B-Instruct-heretic](https://huggingface.co/alpharomercoma/LFM2.5-1.2B-Instruct-heretic) | index 0, trial 133 | 6/100 (from 99) | 0.053 |
| `lfm2_5_2_6b` | LiquidAI/LFM2.5-2.6B | [alpharomercoma/LFM2.5-2.6B-heretic](https://huggingface.co/alpharomercoma/LFM2.5-2.6B-heretic) | index 4, trial 8 | 7/100 (from 99) | 0.034 |
