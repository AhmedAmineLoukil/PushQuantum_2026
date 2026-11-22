# QEC-Emerald — surface-code memory on IQM Emerald

Distance-3 rotated surface code (17 qubits) run as a memory experiment on IQM Emerald,
decoded offline, with the logical error per round fitted from raw shots.

```
Stim circuit (native H/PRX + CZ, no resets) → Qiskit, qubits pinned, 0 SWAPs → Emerald (+ DD)
  → raw shots → no-reset detectors → BP+OSD / MWPM → P_L(n) → fit ε_L
```

## Results (Emerald, calibration set `ecec1d0a…`)

| | ε_L per round |
|---|---|
| Logical Z-memory, DD on (BP+OSD) | **2.61 %** [2.56, 2.67] |
| Logical X-memory, DD on (BP+OSD) | **2.65 %** [2.59, 2.70] |
| Bare physical qubit, same qubits and timing, DD on | 1.63 % [1.60, 1.65] |

No break-even yet: the logical qubit loses 1.6× more per round than one physical qubit.
Every number is regenerated from raw shots by `experiments/reproduce_all.py` →
[`results/reproduced/summary.md`](results/reproduced/summary.md). Details: [RESULTS.md](RESULTS.md).

## What is in here

| Feature | Where |
|---|---|
| Surface code in Emerald's native gates; no-reset detectors (`m_r ⊕ m_{r-2}`), Z and X memory | `qec/code.py`, `qec/circuit.py` |
| Zero-SWAP placement (VF2 + calibration score); every CZ checked against a coupler | `qec/layout.py`, `experiments/layout.py` |
| Hardware runs: one batch per sweep, server-side DD, interleaved repeats, exact payloads saved | `experiments/run_hardware.py` |
| Decoders: MWPM, correlated MWPM, Union-Find, belief-matching, BP+OSD | `qec/decode.py` |
| Decoder priors: SI1000, calibration, measured readout, calibrated per-qubit readout | `qec/noise.py` |
| Fit: binomial MLE, profile-likelihood CI, goodness of fit, held-out decoder choice | `qec/analysis.py` |
| Physical-qubit baseline (fair "no QEC" reference) | `qec/physical.py`, `experiments/run_physical_baseline.py` |
| Pauli twirling of CZ layers (helps against coherent errors in simulation; **hurts on Emerald**) | `qec/circuit.py` (`twirl_seed`) |
| Pulse level (PulLA) runner — needs an organisation token | `experiments/run_pulla.py` |

## What we learned on hardware

* **DD matters most**: X-memory 5.8 % → 2.6 % per round.
* BP+OSD beats correlated MWPM by ~0.2 pp; a measured-readout prior adds ~0.1 pp.
* Picking qubits from calibration alone is worse than avoiding qubits measured bad.
* Twirling: 2.69 % → 2.95 % (worse); Emerald's CZ errors are not mainly coherent.
* High-fidelity (shelved) readout cannot be used via circuit jobs — results are not returned
  ([details](results/hw/invalid/README.md)). At **pulse level** it works: 2.88 % vs 2.92 % with
  QND final readout (same session, within error).
* **Soft information (pulse level, IQ readout)**: on identical shots, per-shot readout confidence
  fixes 127 failures and causes 69 (sign test p = 4·10⁻⁵): ε_L 3.31 % → 3.23 % (MWPM).
  Small because Emerald's readout clouds are 5–9 σ apart. `qec/soft.py`, `experiments/soft_decoding.py`.

## Run it

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # iqm-client 37
.venv/Scripts/python -m pytest -q                         # 40 tests, no token
.venv/Scripts/python experiments/reproduce_all.py         # all numbers from raw shots, no token

# hardware: put IQM_TOKEN=... in .env.  Without --confirm these are dry runs.
.venv/Scripts/python experiments/run_hardware.py --basis Z X --dd on --rounds 1 2 3 4 6 8 --shots 2000 --confirm
.venv/Scripts/python experiments/run_physical_baseline.py --dd on --confirm
```

⚠️ `--confirm` submits real jobs and spends credits. Resonance accepts at most 100 circuits per job.

## Open items for the merge

* Pulse-level access (organisation token) → readout-gate overlap, shelved final readout, soft information.
* Distance 5 has no SWAP-free placement on Emerald, so no Λ (distance-scaling) result.
* Bring in the other implementation's pieces we keep; run `pytest` and `reproduce_all.py` before merging.

Acknowledgement: IQM's challenge scaffold (`PushQuantumQEC`) defined the interfaces.
