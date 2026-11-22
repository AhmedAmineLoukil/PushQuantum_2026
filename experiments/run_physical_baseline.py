"""Physical-qubit memory baseline on the code's own 17 qubits (qec/physical.py).

Every circuit stores |0>, |1>, |+> or |-> on all 9 data qubits and runs n
"rounds" of (gate-time delay + mid-circuit readout of the 8 ancillas), the
same timing and readout environment as the n-round memory experiment, without
encoding. One batch job per DD setting; raw shots saved like run_hardware.py.

    .venv/Scripts/python experiments/run_physical_baseline.py                       # dry run
    .venv/Scripts/python experiments/run_physical_baseline.py --dd both --confirm   # spends credits
    .venv/Scripts/python experiments/run_physical_baseline.py --fake                # noisy Aer dry run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec.hardware import (DATA_DIR, coupling_edges, fetch_calibration, get_backend,  # noqa: E402
                          job_to_arrays, repo_relative, save_run, submit, swap_report,
                          timestamp, transpile_pinned)
from qec.physical import STATE_KEY, STATES, build_physical_baseline, round_gate_time_ns  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8])
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--dd", default="on", choices=["off", "on", "both"])
    ap.add_argument("--quantum-computer", default="emerald")
    ap.add_argument("--layout", help="default: results/layout_<quantum-computer>.json")
    ap.add_argument("--round-ns", type=float, help="gate time per round (default: from the code circuit)")
    ap.add_argument("--confirm", action="store_true", help="actually submit (spends credits)")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--wait", type=float, default=6 * 3600)
    args = ap.parse_args()

    lay_info = json.loads(Path(args.layout or RESULTS / f"layout_{args.quantum_computer}.json").read_text())
    layout = {int(k): v for k, v in lay_info["layout"].items()}
    round_ns = args.round_ns if args.round_ns is not None else round_gate_time_ns()
    print(f"layout: {lay_info['physical']}\ngate time per round: {round_ns:.0f} ns (+ ancilla readout)")

    if args.fake:
        from iqm.qiskit_iqm.fake_backends import IQMFakeAphrodite
        backend, edges = IQMFakeAphrodite(), coupling_edges(None)
    else:
        backend = get_backend(args.quantum_computer)
        edges = coupling_edges(backend)

    keys, tqcs = [], []
    for s in STATES:
        for r in args.rounds:
            tqc = transpile_pinned(build_physical_baseline(r, s, round_ns), backend, layout)
            rep = swap_report(tqc, edges)
            assert rep["two_qubit_gates"] == 0 and "reset" not in rep["ops"], rep
            keys.append((s, r))
            tqcs.append(tqc)
    dd_settings = {"off": [False], "on": [True], "both": [False, True]}[args.dd]
    total = len(tqcs) * args.shots * len(dd_settings)
    print(f"{len(tqcs)} circuits ({len(STATES)} states x rounds {args.rounds}) x {args.shots} shots "
          f"x {len(dd_settings)} DD setting(s) = {total} shots")
    if not (args.confirm or args.fake):
        print("dry run only - add --confirm to submit")
        return

    stamp = timestamp()
    if args.fake:
        from qec.hardware import calibration_from_fake
        cal = calibration_from_fake(backend)
    else:
        cal = fetch_calibration(backend)
    cal_file = DATA_DIR / f"calibration_{'fake' if args.fake else args.quantum_computer}_run{stamp}.json"
    cal_file.write_text(json.dumps({"fetched_at": stamp, "calibration": cal}))

    jobs = []
    for dd in dd_settings:
        job = backend.run(tqcs, shots=args.shots) if args.fake else submit(backend, tqcs, args.shots, dd=dd)
        print(f"submitted DD={'on' if dd else 'off'}: job {job.job_id()}", flush=True)
        jobs.append((dd, job))
    pending = dict(stamp=stamp, rounds=args.rounds, states=list(STATES), shots=args.shots,
                   round_ns=round_ns, quantum_computer=args.quantum_computer, fake=args.fake,
                   layout=lay_info["layout"], physical=lay_info["physical"],
                   calibration_file=repo_relative(cal_file),
                   jobs=[{"dd": dd, "job_id": str(j.job_id())} for dd, j in jobs])
    pp = RESULTS / "hw" / f"pending_phys_{stamp}_{'fake' if args.fake else args.quantum_computer}.json"
    pp.write_text(json.dumps(pending, indent=1))

    for dd, job in jobs:
        nm = [8 * r + 9 for _, r in keys]
        arrays = job_to_arrays(job, nm, timeout=args.wait)
        tag = f"phys_{stamp}_{'fake' if args.fake else args.quantum_computer}_dd{'on' if dd else 'off'}"
        meta = {k: v for k, v in pending.items() if k != "jobs"}
        save_run(RESULTS / "hw" / f"{tag}.npz", dd=dd, job_id=str(job.job_id()), **meta,
                 **{f"raw_{STATE_KEY[s]}_r{r}": a for (s, r), a in zip(keys, arrays)})
        print(f"saved results/hw/{tag}.npz")


if __name__ == "__main__":
    main()
