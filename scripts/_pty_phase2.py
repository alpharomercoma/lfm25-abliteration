"""Phase 2 under a pty: resume the finished study and drive heretic's menus to save one Pareto trial as merged
weights. heretic replaces its settings with the ones stored in the checkpoint on resume, so --trial-index /
--model-action / --save-directory are ignored there; navigating the menus is the upstream-faithful way.
Usage: _pty_phase2.py <pareto_index> <abs_output_dir> -- <heretic cmd...>"""
import sys, time, pexpect
idx = int(sys.argv[1]); out = sys.argv[2]; cmd = sys.argv[sys.argv.index("--") + 1:]
child = pexpect.spawn(cmd[0], cmd[1:], encoding="utf-8", codec_errors="replace", timeout=None, dimensions=(50, 200))
log = open("save.log", "a", buffering=1); child.logfile_read = log

def menu(prompt, downs=0, timeout=None):
    child.expect(prompt, timeout=timeout)
    time.sleep(1)
    for _ in range(downs):
        child.send("\x1b[B"); time.sleep(0.15)      # arrow down
    child.send("\r")

menu("Which trial do you want to use\\?", downs=idx)          # trials are listed in Pareto order
menu("What do you want to do with the decensored model\\?")   # first choice: save to a local folder
child.expect("Path to the folder:"); time.sleep(1)
child.send(out + "\r")
menu("How do you want to export the model\\?")                # first choice: merge the LoRA, full model
i = child.expect(["Model saved to", "Error", "Traceback", pexpect.EOF], timeout=1800)
if i != 0:
    log.write("\n[driver] save failed\n"); sys.exit(1)
log.write(f"\n[driver] saved pareto index {idx} to {out}\n")
child.expect("What do you want to do with the decensored model\\?", timeout=300)
child.sendcontrol("c")
try: child.expect(pexpect.EOF, timeout=60)
except Exception: child.terminate(force=True)
