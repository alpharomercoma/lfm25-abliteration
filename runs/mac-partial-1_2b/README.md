Logs from the first, aborted attempt on the MacBook M5 (MPS), kept because they are the only raw heretic logs that survive.
`phase1.log.txt`: full run until killed at trial ~24 (baseline 99/100, batch 64, ~1 min/trial). `probe.log.txt`: 2-trial
probe after patching torch.svd_lowrank to run on CPU (it hangs on MPS). `export_original_mac.log`: ExecuTorch export of the
unmodified 1.2B on the Mac, and the KleidiAI C++ runner was built from ExecuTorch v1.4.1 there.
