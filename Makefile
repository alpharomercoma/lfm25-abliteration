# LFM2.5-1.2B-Instruct: heretic abliteration -> ExecuTorch CPU export -> Hugging Face publish
# Model selection: SIZE=1_2b (default) or SIZE=2_6b; every path below derives from it.
SIZE       ?= 1_2b
MODEL_1_2b  = LiquidAI/LFM2.5-1.2B-Instruct
MODEL_2_6b  = LiquidAI/LFM2.5-2.6B
MODEL      ?= $(MODEL_$(SIZE))
RUN        ?= runs/lfm2_5_$(SIZE)
OUT_MODEL  ?= models/$(notdir $(MODEL))-heretic
PTE_NAME   ?= lfm2_5_$(SIZE)_heretic_8da4w
PARAMS     ?= configs/lfm2_5_$(SIZE)_params.json
ET_CONFIG  ?= configs/lfm2_5_$(SIZE)_xnnpack_q8da4w.yaml
CTX        ?= 4096
TRIAL      ?=

.PHONY: setup abliterate pareto save sanity export run-pte publish all clean

setup:            ## create both venvs (CUDA torch + heretic, CPU executorch)
	scripts/00_setup.sh

abliterate:       ## heretic optimisation, checkpointed under $(RUN)
	scripts/01_abliterate.sh $(MODEL) $(RUN)

pareto:           ## print the Pareto front of the finished study
	.venv-heretic/bin/python scripts/02_pareto.py $(RUN)

save:             ## restore trial TRIAL=<idx> from the front, merge, save HF checkpoint
	@test -n "$(TRIAL)" || (echo "usage: make save TRIAL=<pareto index>"; exit 1)
	scripts/03_save_model.sh $(MODEL) $(RUN) $(TRIAL) $(OUT_MODEL)

sanity:           ## eager generation on original vs abliterated (benign / tool call / refusal probe)
	.venv-heretic/bin/python scripts/04_sanity_check.py $(MODEL) $(OUT_MODEL)

export:           ## HF checkpoint -> ExecuTorch XNNPACK 8da4w .pte
	scripts/05_export_executorch.sh $(OUT_MODEL) $(PTE_NAME) $(PARAMS) $(ET_CONFIG) $(CTX)

run-pte:          ## generate with the exported .pte on CPU (ExecuTorch python runner)
	scripts/06_run_pte.sh $(OUT_MODEL) export/$(PTE_NAME).pte $(PARAMS)

publish:          ## upload model dir + .pte + model card to Hugging Face (needs .env; HF_REPO_ID=user/name overrides)
	set -a; . ./.env; set +a; .venv-heretic/bin/python scripts/07_publish_hf.py $(OUT_MODEL) --base $(MODEL) --pte export/$(PTE_NAME).pte --run $(RUN) $(if $(HF_REPO_ID),--repo-id $(HF_REPO_ID),)

clean:
	rm -rf export/*.pt

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'
