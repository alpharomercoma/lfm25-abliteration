"""Run heretic under a pseudo-terminal, mirror its output to heretic.log, and stop at the trial menu.
heretic's optuna study is journaled per trial, so nothing is lost; phase 2 resumes it non-interactively."""
import sys, time, pexpect
cmd = sys.argv[1:]
child = pexpect.spawn(cmd[0], cmd[1:], encoding="utf-8", codec_errors="replace", timeout=None, dimensions=(50, 200))
log = open("heretic.log", "a", buffering=1)
child.logfile_read = log
i = child.expect(["Which trial do you want to use\\?", pexpect.EOF])
if i == 0:
    log.write("\n[driver] optimisation finished, trial menu reached; stopping (study is checkpointed)\n")
    time.sleep(2)
    child.sendcontrol("c")
    try:
        child.expect(pexpect.EOF, timeout=60)
    except Exception:
        child.terminate(force=True)
    sys.exit(0)
log.write("\n[driver] heretic exited before the trial menu\n")
sys.exit(child.exitstatus or 1)
