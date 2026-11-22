"""Run the no-reset memory experiment on IQM Emerald for several round counts.

All round counts of one configuration go into ONE batch job (same calibration
for every point of the P_L(n) curve). Raw per-shot measurements are saved, so
every decoding / noise-model choice can be revisited later without QPU time.

    .venv/Scripts/python experiments/run_hardware.py --rounds 1 2 3 4 5 6 --shots 4000            # dry run
    .venv/Scripts/python experiments/run_hardware.py --rounds 1 2 3 4 5 6 --shots 4000 --confirm  # spends credits
    ... --dd both            # DD off and on, back to back, same layout
    ... --basis Z X          # both memory bases (one job per basis and DD setting)
    ... --repeats 2          # repeat the whole block in mirrored (ABBA) order, so a
                             # linear drift during the session cancels between configs
    ... --fake               # full dry run on IQM's fake 54-qubit backend (noisy Aer sim)

At submission the live calibration is fetched and frozen next to the run, and
the exact payload IQM receives (backend.serialize_circuit) is written to
results/hw/payloads/<run>_r<n>.submitted.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import build_memory_circuit  # noqa: E402
from qec.hardware import (DATA_DIR, coupling_edges, fetch_calibration, get_backend,  # noqa: E402
                          job_definition, job_to_arrays, repo_relative, save_run, stim_to_qiskit,
                          submit, swap_report, timestamp, transpile_pinned)

RESULTS = Path(__file__).resolve().parent.parent / "results"
MAX_CIRCUITS_PER_JOB = 100        # IQM Resonance declines larger batches


def run_tag(pending: dict, entry: dict) -> str:
    """File stem of one job's results (backwards compatible with old runs)."""
    basis = entry.get("basis", pending["basis"])
    dd = entry["dd"]
    strat = pending.get("dd_strategy", "standard")
    rep = entry.get("rep")
    ro = entry.get("final_readout", pending.get("final_readout", "qnd"))
    return (f"{pending['stamp']}_{'fake' if pending['fake'] else pending['quantum_computer']}"
            f"_{basis}_dd{'on' if dd else 'off'}"
            f"{'-' + strat if dd and strat != 'standard' else ''}"
            f"{'_ro-' + ro if ro != 'qnd' else ''}"
            f"{'_tw' if entry.get('twirl') else ''}"
            f"{f'_rep{rep}' if rep is not None else ''}")


def twirl_seeds(rounds, twirl: bool, instances: int, base: int) -> list[tuple[int, int | None]]:
    """(rounds, seed) of every circuit in a job, round-major. Untwirled: one
    circuit per round count; twirled: `instances` random twirls per round count."""
    if not twirl:
        return [(r, None) for r in rounds]
    return [(r, base + 1000 * r + i) for r in rounds for i in range(instances)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--basis", nargs="+", default=["Z"], choices=["Z", "X"])
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--dd", default="off", choices=["off", "on", "both"])
    ap.add_argument("--dd-strategy", default="standard", choices=["standard", "dense"],
                    help="DD sequence used when DD is on (see qec.hardware.dd_strategy)")
    ap.add_argument("--final-readout", nargs="+", default=["qnd"], choices=["qnd"],
                    help="readout of the FINAL data measurement. Only 'qnd' (measure/constant) "
                         "works in circuit jobs: the shelved high-fidelity readout "
                         "(measure_fidelity) executes but its results are not returned "
                         "(results/hw/invalid/README.md); it needs pulse-level access.")
    ap.add_argument("--twirl", nargs="+", default=["off"], choices=["off", "on"],
                    help="random Pauli twirl of every CZ layer (qec.circuit twirl_seed); "
                         "'off on' -> one job each, interleaved")
    ap.add_argument("--twirl-instances", type=int, default=20,
                    help="random twirls per round count; each runs shots/instances shots")
    ap.add_argument("--twirl-base-seed", type=int, default=20261122)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the block of configurations in mirrored order (ABBA...)")
    ap.add_argument("--quantum-computer", default="emerald")
    ap.add_argument("--layout", help="default: results/layout_<quantum-computer>.json")
    ap.add_argument("--confirm", action="store_true", help="actually submit (spends credits)")
    ap.add_argument("--fake", action="store_true", help="run on IQMFakeAphrodite instead")
    ap.add_argument("--wait", type=float, default=6 * 3600,
                    help="seconds to wait for results before leaving the job pending")
    ap.add_argument("--retrieve", help="collect a previously submitted run from its pending_*.json")
    args = ap.parse_args()

    if args.retrieve:
        collect(json.loads(Path(args.retrieve).read_text()), None, args.wait)
        return

    layout_path = Path(args.layout or RESULTS / f"layout_{args.quantum_computer}.json")
    lay_info = json.loads(layout_path.read_text())
    if lay_info.get("quantum_computer", args.quantum_computer) != args.quantum_computer and not args.fake:
        raise SystemExit(f"{layout_path} was made for {lay_info['quantum_computer']}, "
                         f"not {args.quantum_computer}: run experiments/layout.py first")
    layout = {int(k): v for k, v in lay_info["layout"].items()}
    print("layout:", lay_info["physical"])

    if args.fake:
        from iqm.qiskit_iqm.fake_backends import IQMFakeAphrodite
        backend = IQMFakeAphrodite()
        edges = coupling_edges(None)
    else:
        backend = get_backend(args.quantum_computer)
        edges = coupling_edges(backend)

    twirls = [t == "on" for t in args.twirl]
    if any(twirls) and args.shots % args.twirl_instances:
        raise SystemExit("--shots must be a multiple of --twirl-instances")
    n_circ = len(args.rounds) * (args.twirl_instances if any(twirls) else 1)
    if n_circ > MAX_CIRCUITS_PER_JOB:
        raise SystemExit(f"{n_circ} circuits per job, but IQM Resonance declines jobs with more than "
                         f"{MAX_CIRCUITS_PER_JOB}: lower --twirl-instances or the number of round counts")
    tqcs, specs = {}, {}
    for basis in args.basis:
        for tw in twirls:
            spec = twirl_seeds(args.rounds, tw, args.twirl_instances, args.twirl_base_seed)
            specs[(basis, tw)] = spec
            tqcs[(basis, tw)] = []
            for r, seed in spec:
                c = build_memory_circuit(3, r, basis, True, None, twirl_seed=seed)
                tqc = transpile_pinned(stim_to_qiskit(c), backend, layout)
                rep = swap_report(tqc, edges)
                assert rep["swaps"] == 0 and rep["non_native_pairs"] == 0, rep
                assert "reset" not in rep["ops"], rep["ops"]
                tqcs[(basis, tw)].append(tqc)
                if seed is None or seed % 1000 == args.twirl_base_seed % 1000:   # print one per round
                    print(f"  {basis} rounds={r} twirl={'on' if tw else 'off'}: transpiled depth "
                          f"{rep['depth']}, {rep['two_qubit_gates']} CZ, 0 SWAP, ops {rep['ops']}")

    dd_settings = {"off": [False], "on": [True], "both": [False, True]}[args.dd]
    block = [(b, dd, ro, tw) for b in args.basis for dd in dd_settings
             for ro in args.final_readout for tw in twirls]
    order = []
    for k in range(args.repeats):
        order += [(b, dd, ro, tw, k if args.repeats > 1 else None)
                  for b, dd, ro, tw in (block if k % 2 == 0 else block[::-1])]
    total = len(order) * len(args.rounds) * args.shots
    print(f"{len(order)} jobs x {len(args.rounds)} round counts x {args.shots} shots = {total} shots total")
    print("submission order:", [f"{b}/DD{'on' if dd else 'off'}/twirl-{'on' if tw else 'off'}"
                                + (f"/rep{k}" if k is not None else "") for b, dd, ro, tw, k in order])
    if not (args.confirm or args.fake):
        print("dry run only - add --confirm to submit")
        return
    stamp = timestamp()
    if args.fake:
        from qec.hardware import calibration_from_fake
        cal = calibration_from_fake(backend)
    else:
        cal = fetch_calibration(backend)        # live calibration at submission time
    cal_file = DATA_DIR / f"calibration_{'fake' if args.fake else args.quantum_computer}_run{stamp}.json"
    cal_file.write_text(json.dumps({"fetched_at": stamp, "calibration": cal}))

    pending = dict(stamp=stamp, rounds=args.rounds, basis=args.basis[0], bases=args.basis,
                   shots=args.shots, dd_strategy=args.dd_strategy,
                   quantum_computer=args.quantum_computer, fake=args.fake, jobs=[],
                   layout=lay_info["layout"], physical=lay_info["physical"],
                   calibration_file=repo_relative(cal_file),
                   depths={b: [t.depth() for t in tqcs[(b, False)]] for b in args.basis
                           if (b, False) in tqcs})
    payload_dir = RESULTS / "hw" / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    live = {}
    for basis, dd, ro, tw, k in order:
        spec = specs[(basis, tw)]
        circuits = tqcs[(basis, tw)]
        shots = args.shots // (args.twirl_instances if tw else 1)     # per circuit
        entry = {"basis": basis, "dd": dd, "final_readout": ro, "twirl": tw, "rep": k,
                 "circuits": [[r, s] for r, s in spec], "shots_per_circuit": shots}
        if not args.fake:
            # Exactly the job IQM receives (DD itself is inserted server-side).
            jd = job_definition(backend, circuits, shots, dd, args.dd_strategy)
            (payload_dir / f"{run_tag(pending, entry)}.submitted.json").write_text(json.dumps(
                {"dd": dd, "dd_strategy": args.dd_strategy, "twirl": tw, "shots_per_circuit": shots,
                 "calibration_set_id": str(jd.calibration_set_id),
                 "circuits": [{"rounds": r, "twirl_seed": s, "name": c.name,
                               "instructions": [{"name": i.name, "locus": list(i.locus), "args": i.args,
                                                 "implementation": i.implementation}
                                                for i in c.instructions]}
                              for (r, s), c in zip(spec, jd.circuits)]}))
        job = (backend.run(circuits, shots=shots) if args.fake else
               submit(backend, circuits, shots, dd=dd, dd_strategy_name=args.dd_strategy))
        print(f"submitted {basis} DD={'on' if dd else 'off'} twirl={'on' if tw else 'off'} rep={k}: "
              f"job {job.job_id()} ({len(circuits)} circuits x {shots} shots)", flush=True)
        entry["job_id"] = str(job.job_id())
        pending["jobs"].append(entry)
        live[entry["job_id"]] = job

    pending_path = (RESULTS / "hw" /
                    f"pending_{stamp}_{'fake' if args.fake else args.quantum_computer}"
                    f"_{''.join(args.basis)}.json")
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(json.dumps(pending, indent=1))
    print(f"job info saved to {pending_path}", flush=True)
    collect(pending, live, args.wait, pending_path)


def collect(pending: dict, live_jobs: dict | None, wait: float, pending_path: Path | None = None):
    """Wait for (or re-attach to) the jobs of a run and save their raw shots."""
    backend = None
    rounds = pending["rounds"]
    for entry in pending["jobs"]:
        jid, dd = entry["job_id"], entry["dd"]
        basis = entry.get("basis", pending["basis"])
        spec = [tuple(x) for x in entry.get("circuits", [[r, None] for r in rounds])]
        num_meas = [build_memory_circuit(3, r, basis, True, None).num_measurements for r, _ in spec]
        path = RESULTS / "hw" / f"{run_tag(pending, entry)}.npz"
        if path.exists():
            print(f"already saved: {path}")
            continue
        job = (live_jobs or {}).get(jid)
        if job is None:
            backend = backend or get_backend(pending["quantum_computer"])
            job = backend.retrieve_job(jid)
        if "ERROR" in str(job.status()).upper():
            print(f"job {jid} FAILED on the server: {job.error_message()} - nothing to collect")
            continue
        try:
            arrays = job_to_arrays(job, num_meas, timeout=wait)
        except Exception as exc:
            print(f"job {jid} not finished ({type(exc).__name__}: {exc}). "
                  f"collect later with: experiments/run_hardware.py --retrieve "
                  f"{pending_path or '<pending json>'}")
            continue
        depths = pending.get("depths")
        if isinstance(depths, dict):
            depths = depths[basis]
        meta = {k: v for k, v in pending.items() if k not in ("jobs", "stamp", "bases", "depths")}
        meta["basis"] = basis
        meta["final_readout"] = entry.get("final_readout", meta.get("final_readout", "qnd"))
        # Pool the twirl instances of each round count (shots stay in circuit order).
        raw = {r: np.concatenate([a for (rr, _), a in zip(spec, arrays) if rr == r]) for r in rounds}
        save_run(path, dd=dd, job_id=jid, rep=entry.get("rep"), depths=depths, **meta,
                 twirl=bool(entry.get("twirl", False)),
                 twirl_seeds={str(r): [s for rr, s in spec if rr == r and s is not None] for r in rounds},
                 **{f"raw_r{r}": a for r, a in raw.items()})
        print(f"saved {path}")


if __name__ == "__main__":
    main()
