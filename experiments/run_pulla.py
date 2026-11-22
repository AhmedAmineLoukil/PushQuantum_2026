"""Pulse-level execution of the memory experiment through IQM PulLA.

The same transpiled circuits as run_hardware.py are converted to IQM pulse
circuits, compiled with PulLA's standard compiler (using the current
calibration set), and submitted as pulse playlists. Pulse-level access lets us
set compiler stages that the circuit API does not expose, here a custom
dynamical-decoupling strategy:

  --dd standard : IQM's STANDARD_DD_STRATEGY (same as the server-side DD)
  --dd dense    : up to 3 back-to-back EDD sequences (24 pulses), centre-
                  aligned, in the ~1 us mid-circuit readout window in which data
                  qubits idle (standard DD places a single 8-pulse EDD there)
  --dd off      : no DD

Results are saved in the run_hardware.py format, so analyze.py and
decoder_comparison.py work unchanged.

    .venv/Scripts/python experiments/run_pulla.py --dd dense --rounds 1 2 3 4 5 --shots 1000            # compile only
    .venv/Scripts/python experiments/run_pulla.py --dd dense --rounds 1 2 3 4 5 --shots 1000 --confirm
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import build_memory_circuit  # noqa: E402
from qec.soft import measured_qubits  # noqa: E402
from qec.hardware import (DATA_DIR, SERVER_URL, coupling_edges, fetch_calibration,  # noqa: E402
                          get_backend, memory_to_array, repo_relative, save_run,
                          stim_to_qiskit, swap_report, timestamp, transpile_pinned,
                          with_final_readout)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def dd_strategy(kind: str):
    from iqm.station_control.interface.models.circuit import DDStrategy
    if kind == "standard":
        from iqm.cpc.compiler.dd import STANDARD_DD_STRATEGY
        return STANDARD_DD_STRATEGY
    if kind == "dense":
        edd = "XYXYYXYX"
        return DDStrategy(gate_sequences=[(25, edd * 3, "center"),
                                          (17, edd * 2, "center"),
                                          (9, edd, "center"),
                                          (5, "YXYX", "center"),
                                          (2, "XX", "center")])
    return None


class PulseAPI:
    """Pulse-level entry points across IQM SDK versions.

    iqm-client >= 37 folds pulse access into the client itself
    (IQMClient.submit_playlist, iqm.qiskit_iqm.pulse_level_utils); older setups
    use the separate iqm-pulla client (iqm.pulla.Pulla).
    """

    def __init__(self, backend, quantum_computer: str):
        self.backend = backend
        try:
            from iqm.qiskit_iqm.pulse_level_utils import qiskit_to_iqm, sweep_job_to_qiskit
        except ImportError:                                     # iqm-client < 37 + iqm-pulla
            from iqm.pulla.pulla import Pulla
            from iqm.pulla.utils_qiskit import qiskit_to_pulla, sweep_job_to_qiskit
            pulla = Pulla(SERVER_URL, quantum_computer=quantum_computer)
            self.convert = lambda tqcs: qiskit_to_pulla(pulla, backend, tqcs)
            self.submit = pulla.submit_playlist
        else:
            self.convert = lambda tqcs: qiskit_to_iqm(backend, tqcs)
            self.submit = backend.client.submit_playlist
        self.to_qiskit = sweep_job_to_qiskit


def compile_for_pulla(api: PulseAPI, tqcs, shots: int, dd: str, final_readout: str = "qnd",
                      first_final_clbits: list[int] | None = None, acquisition: str = "threshold",
                      qubits: list[str] | None = None):
    """Compile to a pulse playlist.

    final_readout : "qnd" (measure/constant) or "fidelity" (measure_fidelity/
                    shelved_constant) for the final data readout; the shelved
                    results ARE returned at pulse level (not via circuit jobs).
    acquisition   : "threshold" (on-chip 0/1) or "complex" (integrated IQ value
                    per readout = soft information) for `measure` on `qubits`.
    """
    circuits, compiler = api.convert(tqcs)
    if final_readout != "qnd":
        circuits = [with_final_readout(c, f, final_readout) if f is not None else c
                    for c, f in zip(circuits, first_final_clbits)]
    settings = compiler.get_settings(circuits=circuits)
    settings.set_shots(shots)
    settings.stages.dynamical_decoupling.apply_dd_strategy.dd_is_disabled = dd == "off"
    thresholds = {}
    for q in qubits or []:
        node = getattr(settings.gates.measure.constant, q)
        thresholds[q] = float(node.integration_threshold.value)
        if acquisition == "complex":
            node.acquisition_type = "complex"
    context = {"DDStrategy": dd_strategy(dd)} if dd != "off" else {}
    run_def, context = compiler.compile(circuits, settings=settings, context=context)
    return run_def, context, thresholds


def raw_results(job, num_meas: list[int], shots: int, complex_values: bool) -> list[np.ndarray]:
    """Per circuit a (shots, num_measurements) array in clbit order, straight
    from the job's measurement keys. Raises if any measurement is missing."""
    res = job.result()
    out = []
    for i, (cm, nm) in enumerate(zip(res.circuit_measurement_results, num_meas)):
        if len(cm) != nm:
            raise RuntimeError(f"circuit {i}: {len(cm)} measurement keys returned, expected {nm}")
        cols = sorted(cm, key=lambda k: int(k.rsplit("_", 1)[1]))
        arr = np.stack([np.asarray(cm[k]).reshape(-1) for k in cols], axis=1)
        if arr.shape[0] != shots:
            raise RuntimeError(f"circuit {i}: {arr.shape[0]} shots returned, expected {shots}")
        out.append(arr.astype(complex if complex_values else bool))
    return out


def drive_pulse_count(run_def, qubit: str, segment: int = -1) -> int:
    """Number of physical drive pulses on one qubit in one circuit (shows DD)."""
    pl = run_def.sweep_definition.playlist
    ch = f"{qubit}__drive.awg"
    cd = pl.channel_descriptions[ch]
    idx = pl.segments[segment].instructions[ch]
    return sum(type(cd.instruction_table[i].operation).__name__ == "IQPulse" for i in idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--basis", default="Z", choices=["Z", "X"])
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--dd", default="standard", choices=["off", "standard", "dense"])
    ap.add_argument("--final-readout", default="qnd", choices=["qnd", "fidelity"],
                    help="final data readout: QND measure, or the shelved high-fidelity readout")
    ap.add_argument("--acquisition", default="threshold", choices=["threshold", "complex"],
                    help="complex = return the integrated IQ value of every readout (soft "
                         "information); two |0>/|1> reference circuits are added to the job")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. rep0)")
    ap.add_argument("--quantum-computer", default="emerald")
    ap.add_argument("--layout")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--wait", type=float, default=6 * 3600)
    args = ap.parse_args()
    if args.acquisition == "complex" and args.final_readout != "qnd":
        raise SystemExit("complex acquisition is implemented for the QND measure only")

    layout_path = Path(args.layout or RESULTS / f"layout_{args.quantum_computer}.json")
    lay_info = json.loads(layout_path.read_text())
    layout = {int(k): v for k, v in lay_info["layout"].items()}
    backend = get_backend(args.quantum_computer)
    api = PulseAPI(backend, args.quantum_computer)

    circuits, tqcs = [], []
    for r in args.rounds:
        c = build_memory_circuit(3, r, args.basis, True, None)
        tqc = transpile_pinned(stim_to_qiskit(c), backend, layout)
        rep = swap_report(tqc, coupling_edges(backend))
        assert rep["swaps"] == 0 and rep["non_native_pairs"] == 0
        circuits.append(c)
        tqcs.append(tqc)

    n_code = len(tqcs)
    first_final = [8 * r for r in args.rounds]
    # No reference circuits: in a test run (20261122-014220) the |0>/|1> reference
    # circuits came back thresholded even with complex acquisition, so the IQ
    # clouds are learned from the run's own readouts instead (qec.soft.soft_inputs).
    phys = [lay_info["physical"][str(q)] for q in range(len(layout))]   # dense qubit -> name
    run_def, context, thresholds = compile_for_pulla(
        api, tqcs, args.shots, args.dd, args.final_readout, first_final, args.acquisition, phys)
    data_q = lay_info["physical"]["4"]          # centre data qubit
    print(f"compiled {len(tqcs)} circuits (DD={args.dd}, final readout={args.final_readout}, "
          f"acquisition={args.acquisition}): centre data qubit {data_q} gets "
          f"{drive_pulse_count(run_def, data_q, n_code - 1)} drive pulses in the "
          f"{args.rounds[-1]}-round circuit")
    if not args.confirm:
        print("compile only - add --confirm to submit")
        return

    live_cal = fetch_calibration(backend, save=False)   # calibration at submission time
    job = api.submit(run_def, context=context)
    stamp = timestamp()
    print(f"submitted pulse job {job.job_id}", flush=True)
    variant = (f"pulla-{args.dd}{'_ro-fidelity' if args.final_readout != 'qnd' else ''}"
               f"{'_iq' if args.acquisition == 'complex' else ''}{'_' + args.tag if args.tag else ''}")
    pending = RESULTS / "hw" / f"pending_pulla_{stamp}_{args.quantum_computer}_{args.basis}_{variant}.json"
    pending.write_text(json.dumps({"job_id": str(job.job_id), "rounds": args.rounds, "dd": args.dd,
                                   "shots": args.shots, "final_readout": args.final_readout,
                                   "acquisition": args.acquisition}))

    t0 = time.time()
    while True:
        job.update()
        if str(job.status).lower().endswith("completed"):
            break
        if any(s in str(job.status).lower() for s in ("failed", "cancel", "abort")):
            raise SystemExit(f"job ended with status {job.status}")
        if time.time() - t0 > args.wait:
            raise SystemExit(f"still {job.status} after {args.wait}s; job id in {pending}")
        time.sleep(15)

    num_meas = [c.num_measurements for c in circuits]
    extra = {}
    if args.acquisition == "complex":
        iq = raw_results(job, num_meas, args.shots, complex_values=True)
        th = np.array([thresholds[q] for q in phys])             # by dense qubit
        arrays = []
        for r, z in zip(args.rounds, iq):
            mq = measured_qubits(build_memory_circuit(3, r, args.basis, True, None))
            arrays.append(z.real > th[mq])                        # the on-chip rule
            extra[f"iq_r{r}"] = z
        extra["iq_threshold"] = th
    else:
        arrays = [a.astype(bool) for a in raw_results(job, num_meas, args.shots, complex_values=False)]
    cal_file = DATA_DIR / f"calibration_{args.quantum_computer}_run{stamp}.json"
    cal_file.write_text(json.dumps({"fetched_at": stamp, "calibration": live_cal}))
    path = RESULTS / "hw" / f"{stamp}_{args.quantum_computer}_{args.basis}_{variant}.npz"
    save_run(path, rounds=args.rounds, basis=args.basis, shots=args.shots,
             dd=args.dd != "off", pulse_level=True, dd_strategy=args.dd,
             final_readout=args.final_readout, acquisition=args.acquisition,
             job_id=str(job.job_id), quantum_computer=args.quantum_computer, fake=False,
             layout=lay_info["layout"], physical=lay_info["physical"],
             calibration_file=repo_relative(cal_file), depths=[t.depth() for t in tqcs[:n_code]],
             **{f"raw_r{r}": a for r, a in zip(args.rounds, arrays)}, **extra)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
