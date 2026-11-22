"""Pulse-level readout experiments: soft-information decoding and readout choice.

1. Soft information (runs *_iq_*.npz, acquisition_type="complex"): on IDENTICAL
   shots,
     hard-MWPM : matching graph with every readout's AVERAGE misread probability
     soft-MWPM : same graph, misread probability of each readout in each shot
                 from its IQ value (qec.soft)
     BP+OSD    : the standard hard decoder (SI1000 prior), for reference.
   The prior is SI1000 for gates/idling + measured readout state changes
   (characterization E1) without classical misreads, which the IQ data supply.
2. Readout choice: shelved high-fidelity final readout (*_ro-fidelity_*) vs the
   QND readout of the IQ runs (their bits follow the on-chip threshold rule, so
   they are equivalent to threshold acquisition). Interleaved in one session.

    .venv/Scripts/python experiments/soft_decoding.py [--session 20261122-0142]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import NoiseModel, build_memory_circuit  # noqa: E402
from qec.analysis import fit_ler  # noqa: E402
from qec.decode import decode_failures, measurements_to_detections  # noqa: E402
from qec.hardware import load_run  # noqa: E402
from qec.soft import SoftMatcher, soft_inputs, without_misreads  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HW = ROOT / "results" / "hw"
CHAR = HW / "meas_20261121-170050_emerald.json"


def fit_row(rounds, fails):
    k = [int(fails[r].sum()) for r in rounds]
    n = [len(fails[r]) for r in rounds]
    f = fit_ler(rounds, k, n)
    return {"failures": k, "shots": n, "eps": f.eps, "ci68": f.ci68, "p_gof": f.p_gof}


def pooled(rows):
    """Sum failures/shots of several repeats and refit."""
    rounds = sorted({r for row in rows for r in row["rounds"]})
    k = [sum(row["failures"][row["rounds"].index(r)] for row in rows) for r in rounds]
    n = [sum(row["shots"][row["rounds"].index(r)] for row in rows) for r in rounds]
    f = fit_ler(rounds, k, n)
    return {"rounds": rounds, "failures": k, "shots": n, "eps": f.eps, "ci68": f.ci68, "p_gof": f.p_gof}


def analyze_iq(path: Path):
    run = load_run(path)
    rounds, basis = [int(r) for r in run["rounds"]], run["basis"]
    names = {int(k): v for k, v in run["physical"].items()}
    base = NoiseModel.si1000(0.01).with_characterization(json.loads(CHAR.read_text()), names)
    base_nm = without_misreads(base)
    si = soft_inputs(run)
    fails = {"bposd (SI1000)": {}, "hard-MWPM": {}, "soft-MWPM": {}}
    for r in rounds:
        bits, p, _ = si[r]
        clean = build_memory_circuit(3, r, basis, True, None)
        det, obs = measurements_to_detections(bits, clean)
        fails["bposd (SI1000)"][r] = decode_failures(
            det, obs, build_memory_circuit(3, r, basis, True, NoiseModel.si1000(0.01)), "bposd")
        sm = SoftMatcher(build_memory_circuit(3, r, basis, True, base_nm), clean)
        fails["hard-MWPM"][r] = (sm.decode(det, p.mean(axis=0)) != obs).any(axis=1)
        fails["soft-MWPM"][r] = (sm.decode(det, p) != obs).any(axis=1)
    out = {"file": path.name, "rounds": rounds, "decoders": {}}
    for name, f in fails.items():
        out["decoders"][name] = {"rounds": rounds, **fit_row(rounds, f)}
    # paired per-shot comparison of hard vs soft MWPM (same shots, same graph)
    h = np.concatenate([fails["hard-MWPM"][r] for r in rounds])
    s = np.concatenate([fails["soft-MWPM"][r] for r in rounds])
    out["paired"] = {"hard_only": int((h & ~s).sum()), "soft_only": int((s & ~h).sum())}
    out["readout"] = {
        "mean_p_misread": float(np.mean([si[r][1].mean() for r in rounds])),
        "ambiguous_fraction_p_gt_0.1": float(np.mean([(si[r][1] > 0.1).mean() for r in rounds])),
        "leakage_like_fraction": float(np.mean([si[r][2].mean() for r in rounds])),
        "separation_sigma": {names[q]: float(abs(m.mu1[0] - m.mu0[0]) / m.sigma[0])
                             for q, m in si["models"].items()}}
    return out


def analyze_threshold(path: Path):
    run = load_run(path)
    rounds, basis = [int(r) for r in run["rounds"]], run["basis"]
    f = {}
    for r in rounds:
        det, obs = measurements_to_detections(run[f"raw_r{r}"], build_memory_circuit(3, r, basis, True, None))
        f[r] = decode_failures(det, obs, build_memory_circuit(3, r, basis, True, NoiseModel.si1000(0.01)),
                               "bposd")
    return {"file": path.name, "final_readout": run.get("final_readout", "qnd"),
            "bposd (SI1000)": {"rounds": rounds, **fit_row(rounds, f)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20261122-014", help="file-name prefix of the session")
    args = ap.parse_args()
    iq_files = sorted(glob.glob(str(HW / f"{args.session}*_pulla-*_iq*.npz")))
    ro_files = sorted(glob.glob(str(HW / f"{args.session}*_pulla-*_ro-fidelity*.npz")))
    iq = [analyze_iq(Path(f)) for f in iq_files]
    ro = [analyze_threshold(Path(f)) for f in ro_files]
    P = lambda x: f"{100 * x:.2f}"  # noqa: E731
    print("== Soft information (identical shots) ==")
    for r in iq:
        print(f"{r['file']}: mean p_misread {r['readout']['mean_p_misread']:.4f}, "
              f"ambiguous readouts {r['readout']['ambiguous_fraction_p_gt_0.1']:.3%}, "
              f"paired failures hard-only {r['paired']['hard_only']} / soft-only {r['paired']['soft_only']}")
        for name, d in r["decoders"].items():
            print(f"   {name:<16} eps = {P(d['eps'])} % [{P(d['ci68'][0])}, {P(d['ci68'][1])}]  "
                  f"k = {d['failures']}")
    summary = {"iq_runs": iq, "fidelity_runs": ro, "pooled": {}}
    if iq:
        for name in iq[0]["decoders"]:
            summary["pooled"][f"QND/{name}"] = pooled([r["decoders"][name] for r in iq])
        hard = sum(r["paired"]["hard_only"] for r in iq)
        soft = sum(r["paired"]["soft_only"] for r in iq)
        from scipy.stats import binomtest
        summary["pooled"]["paired_sign_test"] = {
            "hard_only": hard, "soft_only": soft,
            "p_value": binomtest(hard, hard + soft).pvalue if hard + soft else None}
    if ro:
        summary["pooled"]["shelved/bposd (SI1000)"] = pooled([r["bposd (SI1000)"] for r in ro])
    print("== Pooled over repeats ==")
    for name, d in summary["pooled"].items():
        if "eps" in d:
            print(f"   {name:<28} eps = {P(d['eps'])} % [{P(d['ci68'][0])}, {P(d['ci68'][1])}]  "
                  f"p_gof = {d['p_gof']:.2g}  k = {d['failures']}")
        else:
            print(f"   {name:<28} {d}")
    out = ROOT / "results" / "figures" / "pulse_readout.json"
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
