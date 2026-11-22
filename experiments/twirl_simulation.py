"""Does Pauli twirling help the memory experiment against COHERENT errors?

Stim only simulates Pauli noise, so this uses Qiskit Aer's state-vector
simulator on the 17-qubit circuit. After every CZ a coherent conditional-phase
error RZZ(theta) is applied (a mis-calibrated CZ phase, a typical coherent
error on transmons), plus optional coherent single-qubit over-rotation RX(phi)
after every H. The same circuits are run untwirled and with random Pauli
twirls (qec.circuit twirl_seed; the error sits inside the twirl, right after
the CZ). Both are decoded with the same SI1000 prior.

    .venv/Scripts/python experiments/twirl_simulation.py --theta 0.2 --shots 2000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import NoiseModel, build_memory_circuit  # noqa: E402
from qec.analysis import fit_ler  # noqa: E402
from qec.decode import decode, measurements_to_detections  # noqa: E402
from qec.hardware import memory_to_array, stim_to_qiskit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def with_coherent_errors(qc: QuantumCircuit, theta: float, phi: float) -> QuantumCircuit:
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        out.append(inst)
        q = inst.qubits
        if inst.operation.name == "cz" and theta:
            out.rzz(theta, q[0], q[1])
        elif inst.operation.name == "h" and phi:
            out.rx(phi, q[0])
    return out


def run(rounds, basis, theta, phi, shots, instances, seed):
    sim = AerSimulator(method="statevector")
    prior = NoiseModel.si1000(0.01)
    rows = []
    for r in rounds:
        ref = build_memory_circuit(3, r, basis, True, None)
        noisy = build_memory_circuit(3, r, basis, True, prior)
        res = {}
        for mode in ("plain", "twirled"):
            seeds = [None] if mode == "plain" else [seed * 1000 + 17 * r + i for i in range(instances)]
            per = shots // len(seeds)
            mems = []
            for s in seeds:
                qc = with_coherent_errors(stim_to_qiskit(build_memory_circuit(3, r, basis, True, None,
                                                                              twirl_seed=s)), theta, phi)
                mems += sim.run(qc, shots=per, memory=True,
                                seed_simulator=(s or seed) + r).result().get_memory()
            det, obs = measurements_to_detections(memory_to_array(mems, ref.num_measurements), ref)
            d = decode(det, obs, noisy, "correlated")
            res[mode] = (d.errors, d.shots, float(det.mean()))
        rows.append({"rounds": r, **{m: {"errors": e, "shots": n, "detection_rate": dr}
                                     for m, (e, n, dr) in res.items()}})
        print(f"  r={r}: plain {res['plain'][0]}/{res['plain'][1]} (det {res['plain'][2]:.3f})   "
              f"twirled {res['twirled'][0]}/{res['twirled'][1]} (det {res['twirled'][2]:.3f})", flush=True)
    fits = {}
    for mode in ("plain", "twirled"):
        f = fit_ler([x["rounds"] for x in rows], [x[mode]["errors"] for x in rows],
                    [x[mode]["shots"] for x in rows])
        fits[mode] = {"eps": f.eps, "ci68": f.ci68, "p_gof": f.p_gof}
        print(f"  {mode:>8}: {f}")
    return rows, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--theta", type=float, nargs="+", default=[0.2], help="RZZ angle after every CZ")
    ap.add_argument("--phi", type=float, default=0.0, help="RX over-rotation after every H")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = []
    for th in args.theta:
        print(f"theta = {th}, phi = {args.phi}")
        rows, fits = run(args.rounds, args.basis, th, args.phi, args.shots, args.instances, args.seed)
        out.append({"theta": th, "phi": args.phi, "basis": args.basis, "rows": rows, "fits": fits})
    path = ROOT / "results" / "figures" / "twirl_simulation.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
