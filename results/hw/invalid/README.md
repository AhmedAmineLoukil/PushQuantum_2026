# Invalid runs — kept for the record, excluded from every analysis

`20261122-001109_emerald_{Z,X}_ddon_ro-fidelity_rep{0,1}.npz` (jobs listed in
`../pending_20261122-001109_emerald_ZX.json`, entries with `"final_readout": "fidelity"`).

These jobs replaced the 9 final data measurements with IQM's high-fidelity
shelved readout (`measure_fidelity` / `shelved_constant`) via the circuit API.
The server executed them but **returned results only for the 8 `measure`
instructions per round, not for `measure_fidelity`**: the measurement keys of
the final readout are missing from the raw job result, and `qiskit-iqm`
silently filled the missing classical bits with 0. Every final data bit is
therefore 0 in every shot (a real surface-code state gives ~50 % per data
qubit), so the logical observable and the final detectors are meaningless.

Consequences:
* the shelved readout is not usable through the circuit API — it needs
  pulse-level access (PulLA, organisation token);
* `qec.hardware.job_to_arrays` now checks that every measurement key comes back
  and raises otherwise, so this cannot pass silently again.

The interleaved `qnd` jobs of the same session
(`../20261122-001109_emerald_{Z,X}_ddon_rep{0,1}.npz`) are unaffected and valid.
