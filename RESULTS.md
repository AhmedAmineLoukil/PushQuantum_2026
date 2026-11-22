# Results — d = 3 rotated surface-code memory on IQM Emerald

**Setup (every run):** distance-3 rotated surface code, 9 data + 8 ancilla qubits, **no
mid-circuit reset** (detectors `m_r ⊕ m_{r-2}`), native H/PRX + CZ, **0 SWAPs** (layout pinned,
routing disabled, every CZ checked against a coupler). Main patch: QB13, 14, 19, 20, 21, 22, 27,
28, 29, 30, 31, 36, 37, 38, 39, 44, 45. All Emerald runs used **one calibration set,
`ecec1d0a-1529-4c90-9b02-d1b30ddaa2e3`**, so differences between runs are not calibration changes.

**Definitions.** *P_L(n)* = fraction of shots whose decoded logical value differs from the prepared
one after *n* rounds. *ε_L* = logical error per round from a binomial maximum-likelihood fit of
`P_L(n) = ½[1 − (1 − 2ε_L)^n]` over all round counts (the challenge formula, A = 1), with 68 %
profile-likelihood intervals and a G² goodness-of-fit test. A fit rejected at p < 0.01 is flagged
and its ε is not quoted as a per-round rate. Unless stated otherwise: decoder BP+OSD, prior SI1000
(p = 1 %), DD = IQM's standard server-side dynamical decoupling.

**Reproducibility.** Every number here is regenerated from the raw shots in `results/hw/` by
`experiments/reproduce_all.py` → [`results/reproduced/summary.md`](results/reproduced/summary.md)
(per-point exact intervals, both fit forms, held-out decoder selection). The no-reset detectors
are re-derived independently in numpy and match Stim on every shot of every run.
`experiments/verify_circuits.py` rebuilds every executed circuit against the live backend and
checks it (all pass).

## Headline

| Emerald, DD on (interleaved session, 4 000 shots per point, n = 1, 2, 3, 4, 6, 8) | ε_L per round | fit p |
|---|---|---|
| **Logical Z-memory** | **2.61 %** [2.56, 2.67] | 0.44 |
| **Logical X-memory** | **2.65 %** [2.59, 2.70] | 0.48 |
| Physical qubit, same qubits, same timing and readout windows, no encoding | 1.63 % [1.60, 1.65] | 0.22 |

* **No break-even yet**: the logical qubit loses 1.6× more per round than one bare physical qubit
  sitting through the same rounds (S2). The old comparison "8.6 % without correction" used the raw
  parity of the encoded circuit, which is not a physical-qubit reference and whose exponential fit
  fails (p ≈ 10⁻⁹); it only shows how much the decoder adds.
* **Reproducible**: Z-memory with DD on gives 2.34–2.68 % in 5 runs over 3 sessions; X-memory
  2.52–2.67 % in 4 runs. P_L(1) ≈ 3.0–3.5 %.
* **Best decoder without selection bias**: choosing the prior/decoder on half of the shots and
  scoring it on the other half gives 2.33–2.70 % (Z) and 2.38–2.56 % (X); the in-sample best is
  within 0.05 pp of this, so picking the best combination does not inflate the result.
* d = 5 has no SWAP-free placement on Emerald (VF2 finds none), so no distance scaling (Λ) is shown.

## What changes ε_L — controlled experiments

One factor changed at a time, configurations interleaved in the same session (ABBA order) and,
for decoder questions, decoded on **identical shots**.

| Factor | Off / A | On / B | Result |
|---|---|---|---|
| **Dynamical decoupling** (S1) | Z 2.96 %, X 5.78 % | Z 2.61 %, X 2.65 % | **DD is the largest single gain**; essential for X-memory (2.2×) |
| Decoder (same shots) | correlated MWPM ≈ 2.8 % | BP+OSD ≈ 2.6 % | BP+OSD best in every Emerald run, ~0.2 pp |
| Prior: measured readout (E1 characterization + calibrated per-qubit assignment error) vs SI1000 | – | – | small, consistent gain (~0.05–0.15 pp), survives held-out selection |
| Placement (S1, same window) | main patch 2.55–2.68 % | calibration-ranked best patch 2.77 % | calibration-only ranking picks a **worse** patch (it includes qubits measured bad, QB26) |
| **Pauli twirling** of every CZ layer (S3) | 2.69 % [2.64, 2.75] | 2.95 % [2.89, 3.01] | **worse on hardware**; see S3 |
| Final data readout, pulse level (S4) | QND 2.92 % [2.86, 2.98] | shelved high-fidelity 2.88 % [2.82, 2.94] | works, no significant gain |
| **Soft information** (IQ readout), same shots (S4) | hard MWPM 3.31 % | soft MWPM 3.23 % | significant: 127 vs 69 shots fixed/broken, sign test p = 4·10⁻⁵ |

## S1. Interleaved repeat of the main experiment (20261121-2337)

Z and X memory × DD off/on, rounds 1, 2, 3, 4, 6, 8, 2 × 2 000 shots per point in ABBA order
(8 jobs). Reproduces the earlier headline (2.65 % / 2.60 %, 4 000 shots, n = 1…5) within
errors. X-memory without DD: 5.78 %, i.e. idle dephasing dominates X-type logical errors.

## S2. Physical-qubit baseline (`qec/physical.py`, 20261121-2339)

The fair "no QEC" reference: the 9 data qubits of the same patch store |0⟩, |1⟩, |+⟩ or |−⟩
through *n* "rounds" of (one round's gate time as a delay + the same mid-circuit readout of the 8
ancillas), then are measured. Same qubits, timing, readout windows and server-side DD; no entangling
gates. 1 000 shots × 9 qubits per point, n = 0, 1, 2, 3, 4, 6, 8. Fit with free amplitude A
(state preparation and measurement error).

| | DD on | DD off |
|---|---|---|
| \|0⟩, \|1⟩ | 1.80 %, 1.73 % | 0.0 %, 3.71 % |
| \|+⟩, \|−⟩ | 1.52 %, 1.47 % | 9.60 %, 9.85 % |
| **average of the four states** | **1.63 %** [1.60, 1.65] | 5.22 % (fit rejected) |

DD removes dephasing of |±⟩ (9.7 → 1.5 % per round) but costs |0⟩ about 1.8 % per round (pulse
errors on a state that would otherwise not decay). Best single qubit: QB20, 0.98 % per round.

## S3. Pauli twirling (`qec/circuit.py`, `twirl_seed`)

Random Paulis before every CZ layer and their exact CZ-conjugate after it, fenced by barriers so the
transpiler cannot cancel them; the ideal circuit, detectors and decoder are unchanged (tested in
Stim, Aer and after transpilation). 16 random twirls per round count, 125 shots each.

* **Simulation** (Aer state vector, coherent CZ phase error RZZ(θ) after every CZ, 4 000 shots):
  twirling lowers the logical failures, e.g. θ = 0.3: ε 3.64 → 3.12 %; θ = 0.15: 40 → 20 failures
  at 3 rounds. (Coherent errors do not follow the exponential model, so these fits are rejected and
  serve only as a comparison.)
* **Emerald** (interleaved, 2 × 2 000 shots per point): 2.69 → 2.95 %, detection rate
  0.116 → 0.123. The twirl adds ~60 % single-qubit gates and one layer per round; Emerald's CZ
  errors are evidently not dominated by coherent errors, so the extra gates cost more than the
  twirl gains. A clean negative result.

## S4. Pulse level (PulLA, organisation token; `experiments/run_pulla.py`, `soft_decoding.py`)

Z-memory, standard DD, rounds 1, 2, 3, 4, 6, 8, 2 × 2 000 shots per configuration, order
IQ / shelved / shelved / IQ.

* **QND vs high-fidelity final readout.** Mid-circuit ancilla readouts must stay QND (no reset: the
  post-measurement state carries over). The final data readout destroys the state anyway, so the
  shelved readout (`measure_fidelity`, calibrated assignment error 0.81 % vs 1.19 %) was used there.
  Result 2.88 % vs 2.92 %: no significant gain at this shot count. Through *circuit* jobs the shelved
  results are not returned at all (bits silently zero-filled; `results/hw/invalid/README.md`);
  `job_to_arrays` now refuses such jobs.
* **Soft information.** With `acquisition_type = "complex"` every readout returns its integrated IQ
  value. Each qubit's |0⟩/|1⟩ clouds are learned from all its readouts in the run (two-Gaussian
  mixture; clouds 4.9–8.9 σ apart), and every readout gets P(bit is wrong | IQ). The decoder sets
  each misread edge per shot from it (`qec/soft.py`); the control uses the same graph with average
  probabilities. On identical shots soft decoding fixes 127 failures and causes 69 (p = 4·10⁻⁵),
  ε_L 3.31 → 3.23 % (MWPM). The gain is small because Emerald's readout is already well separated:
  only ~0.5 % of readouts are ambiguous. In a synthetic test with 1.9 % misreads the same code
  lowers failures by 35 %. BP+OSD on hard bits (2.92 %) is still the best decoder on these shots.
* **Readout-gate overlap / shorter readout** (the largest remaining lever: readout is ~80 % of a
  1.3 µs round) needs re-calibrated readout and scheduling changes; not attempted. The circuit-level
  `fast_constant` readout is only ~0.09 µs shorter and has no calibration data.

## Earlier experiments (verified from saved data)

* **E1 Repeated-measurement characterization** (Emerald and Garnet, every qubit, 5 readouts
  without reset): |1⟩ decays 4–8 % per readout on every qubit; the per-ancilla flip rate predicts
  the time-like error learned from QEC data (Pearson r = 0.84; calibration readout r = 0.37).
* **E2 Spatial variation**: the same code on three other Emerald regions gives 3.0–3.9 % (main
  patch 2.65 %); each region's hottest ancilla is one of the worst E1 qubits.
* **E3 DD strategy**: dense DD (174 pulses on the centre data qubit vs 94) gives no further gain.
* **E4 Pulse level**: PulLA with the standard compiler reproduces the circuit-level result.
* **E5 Readout-informed prior**: SI1000 for gates + E1 readout is the best prior in 11/12 runs
  (MWPM), small gains (~0.1 pp).
* **Garnet** (20-qubit device): ε_L ≈ 11–12 % with or without decoding — above threshold for this
  circuit; most fits there fail the goodness-of-fit test.
* **Ancilla errors and placement**: the time-like error learned per stabilizer exposed ancillas the
  calibration underestimates by up to 30× (Emerald QB33, QB26; Garnet QB11); excluding QB33/QB26
  made the main patch.

## Comparison with the first independent pipeline

Uniform weights, MWPM, no DD, other qubits, 500 shots per point: ε_L ≈ 11.5 % with an anomalous
P_L(1) = 0.8 % that breaks the fit. Same calibration set as all runs here; the 4× difference
comes from placement, DD, native-gate compilation and the decoder prior. The jump from round 1
to round 2 suggests (unverified) a no-reset detector problem in that converter; its raw shots can
be decoded with `qec.decode.measurements_to_detections` to check.

## Limitations

* No break-even: logical 2.6 % vs physical 1.6 % per round. d = 3 only; no Λ.
* Statistical uncertainty only; drift is controlled by interleaving and one calibration set, not
  modelled.
* SI1000 / characterization priors are still generic for gates; learned (p_ij) weights win only on
  noisy data (Garnet).
* Soft decoding is implemented for matching only, not BP+OSD.
* Simulations (twirling, sim-vs-hardware) are indicative, not predictions of Emerald.
