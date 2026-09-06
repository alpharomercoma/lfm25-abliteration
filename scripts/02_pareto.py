"""Print the Pareto front of a heretic study, sorted exactly like heretic's trial menu
(ascending refusal rate, then KL divergence), so an index can be passed to `make save TRIAL=<idx>`."""
import glob, sys, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

optuna.logging.set_verbosity(optuna.logging.WARNING)
run = sys.argv[1] if len(sys.argv) > 1 else "runs/lfm2_5_1_2b"
(path,) = glob.glob(f"{run}/checkpoints/*.jsonl")
storage = JournalStorage(JournalFileBackend(path))
study = optuna.load_study(study_name=storage.get_all_studies()[0].study_name, storage=storage)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"study: {path}\ncompleted trials: {len(done)} / {len(study.trials)}\n")

def scores(t):
    s = {x["name"]: x["score"]["value"] for x in t.user_attrs["scores"]}
    return s.get("Refusals"), s.get("KL divergence")

front = sorted(study.best_trials, key=scores)
print(f"{'idx':>3} {'trial':>5} {'refusals':>8} {'KL':>8}  direction_index")
for i, t in enumerate(front):
    r, kl = scores(t)
    print(f"{i:>3} {t.number:>5} {r:>8.2f} {kl:>8.4f}  {t.user_attrs.get('direction_index')}")
