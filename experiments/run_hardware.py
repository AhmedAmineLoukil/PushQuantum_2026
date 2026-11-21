"""Run the no-reset memory experiment on IQM Emerald for several round counts.

All round counts go into ONE batch job (same calibration for every point of
the P_L(n) curve). Raw per-shot measurements are saved, so every decoding /
noise-model choice can be revisited later without spending QPU time.

    .venv/Scripts/python experiments/run_hardware.py --rounds 1 2 3 4 5 6 --shots 4000            # dry run
    .venv/Scripts/python experiments/run_hardware.py --rounds 1 2 3 4 5 6 --shots 4000 --confirm  # spends credits
    ... --dd both      # DD off and on, back to back, same layout
    ... --fake         # full dry run on IQM's fake 54-qubit backend (noisy Aer sim)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import build_memory_circuit, describe  # noqa: E402
from qec.hardware import (DATA_DIR, coupling_edges, get_backend, job_to_arrays,  # noqa: E402
                          save_run, stim_to_qiskit, submit, swap_report,
                          timestamp, transpile_pinned)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--basis", default="Z", choices=["Z", "X"])
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--dd", default="off", choices=["off", "on", "both"])
    ap.add_argument("--quantum-computer", default="emerald")
    ap.add_argument("--layout", default=str(RESULTS / "layout.json"))
    ap.add_argument("--confirm", action="store_true", help="actually submit (spends credits)")
    ap.add_argument("--fake", action="store_true", help="run on IQMFakeAphrodite instead")
    args = ap.parse_args()

    lay_info = json.loads(Path(args.layout).read_text())
    layout = {int(k): v for k, v in lay_info["layout"].items()}
    print("layout:", lay_info["physical"])

    if args.fake:
        from iqm.qiskit_iqm.fake_backends import IQMFakeAphrodite
        backend = IQMFakeAphrodite()
        edges = coupling_edges(None)
    else:
        backend = get_backend(args.quantum_computer)
        edges = coupling_edges(backend)

    circuits, tqcs = [], []
    for r in args.rounds:
        c = build_memory_circuit(3, r, args.basis, True, None)
        tqc = transpile_pinned(stim_to_qiskit(c), backend, layout)
        rep = swap_report(tqc, edges)
        assert rep["swaps"] == 0 and rep["non_native_pairs"] == 0, rep
        print(f"  rounds={r}: {describe(c)['gate_counts']}  -> transpiled depth {rep['depth']}, "
              f"{rep['two_qubit_gates']} CZ, 0 SWAP")
        circuits.append(c)
        tqcs.append(tqc)

    total = len(tqcs) * args.shots * (2 if args.dd == "both" else 1)
    print(f"{len(tqcs)} circuits x {args.shots} shots"
          f"{' x 2 (DD off/on)' if args.dd == 'both' else ''} = {total} shots total")
    if not (args.confirm or args.fake):
        print("dry run only - add --confirm to submit")
        return

    dd_settings = {"off": [False], "on": [True], "both": [False, True]}[args.dd]
    jobs = []
    for dd in dd_settings:
        if args.fake:
            job = backend.run(tqcs, shots=args.shots)
        else:
            job = submit(backend, tqcs, args.shots, dd=dd)
        print(f"submitted DD={'on' if dd else 'off'}: job {job.job_id()}")
        jobs.append((dd, job))

    cal_file = DATA_DIR / f"calibration_{args.quantum_computer}_latest.json"
    if args.fake:
        # Record the fake backend's own error profile as the run's calibration.
        from qec.hardware import calibration_from_fake
        cal_file = DATA_DIR / "calibration_fake_aphrodite.json"
        cal_file.write_text(json.dumps({"fetched_at": "fake", "calibration": calibration_from_fake(backend)}))
    for dd, job in jobs:
        arrays = job_to_arrays(job, [c.num_measurements for c in circuits])
        tag = f"{timestamp()}_{'fake' if args.fake else args.quantum_computer}_{args.basis}_dd{'on' if dd else 'off'}"
        path = RESULTS / "hw" / f"{tag}.npz"
        save_run(path, rounds=args.rounds, basis=args.basis, shots=args.shots, dd=dd,
                 job_id=str(job.job_id()), quantum_computer=args.quantum_computer,
                 fake=args.fake, layout=lay_info["layout"], physical=lay_info["physical"],
                 calibration_file=str(cal_file) if cal_file.exists() else None,
                 depths=[t.depth() for t in tqcs],
                 **{f"raw_r{r}": a for r, a in zip(args.rounds, arrays)})
        print(f"saved {path}")


if __name__ == "__main__":
    main()
