"""Regenerate every headline number from the saved raw shots (no token).

For every hardware memory run in results/hw/:
  * detection events from the raw measurement records (Stim m2d, no-reset
    detectors), checked against an independent numpy re-derivation;
  * per round count: failures / shots / P_L with exact (Clopper-Pearson)
    68 % and 95 % intervals, for decoders none / MWPM / correlated MWPM /
    BP+OSD with the generic SI1000 (p = 1 %) prior;
  * binomial-MLE fit of P_L(n) = (1 - A (1-2 eps)^n)/2 with A = 1 (challenge
    form) and A free: profile-likelihood intervals, G^2 goodness of fit,
    Pearson residuals. A fit rejected at the 1 % level is flagged and its
    eps is NOT a meaningful per-round rate;
  * held-out decoder selection: the prior/decoder combination is chosen on
    half of the shots (even indices) and evaluated on the other half (odd),
    and vice versa. This removes the "best of N combinations" optimism.
For the physical-qubit baseline (results/hw/phys_*.npz): per-state and
pooled physical error per round on the same qubits and timing.

    .venv/Scripts/python experiments/reproduce_all.py            # everything
    .venv/Scripts/python experiments/reproduce_all.py --quick    # skip BP+OSD and held-out selection

Writes results/reproduced/summary.json and summary.md.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import NoiseModel, RotatedSurfaceCode, build_memory_circuit  # noqa: E402
from qec.analysis import clopper_pearson, fit_ler  # noqa: E402
from qec.decode import decode_failures, measurements_to_detections  # noqa: E402
from qec.hardware import load_run, run_calibration  # noqa: E402
from qec.physical import STATE_KEY, STATES, data_errors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HW = ROOT / "results" / "hw"
OUT = ROOT / "results" / "reproduced"
CHAR = {"emerald": HW / "meas_20261121-170050_emerald.json",
        "garnet": HW / "meas_20261121-170050_garnet.json"}
CODE = RotatedSurfaceCode(3)


# ------------------------------------------------------------ detectors
def numpy_detectors(raw: np.ndarray, rounds: int, basis: str):
    """Independent re-derivation of the no-reset detectors (not via Stim).

    Ancilla outcome without reset: m_r = s_r XOR m_{r-1}, so the syndrome is
    s_r = m_r XOR m_{r-1} (m_0 = 0) and the detector is s_r XOR s_{r-1}; the
    final detector compares the data parity with s_R.
    """
    anc = CODE.ancillas
    m = [np.zeros((raw.shape[0], len(anc)), bool)]
    for r in range(1, rounds + 1):
        m.append(raw[:, 8 * (r - 1): 8 * r])
    s = [m[0]] + [m[r] ^ m[r - 1] for r in range(1, rounds + 1)]
    data = raw[:, 8 * rounds:]
    cols = []
    for r in range(1, rounds + 1):
        for i, st in enumerate(CODE.stabilizers):
            if r == 1 and st.basis != basis:
                continue
            cols.append(s[r][:, i] ^ s[r - 1][:, i])
    for i, st in enumerate(CODE.stabilizers):
        if st.basis == basis:
            cols.append(np.bitwise_xor.reduce(data[:, list(st.data)], axis=1) ^ s[rounds][:, i])
    obs = np.bitwise_xor.reduce(data[:, list(CODE.logical[basis])], axis=1)
    return np.stack(cols, 1), obs


# ----------------------------------------------------------------- fits
def fit_summary(rounds, k, n):
    out = {}
    for name, free in (("A=1", False), ("A free", True)):
        f = fit_ler(rounds, k, n, free_amplitude=free)
        out[name] = dict(eps=f.eps, eps_err=f.eps_err, ci68=f.ci68, ci95=f.ci95,
                         A=f.amp, A_err=f.amp_err, g2=f.g2, dof=f.dof, p_gof=f.p_gof,
                         model_ok=f.model_ok, residuals=[float(x) for x in f.residuals])
    return out


def points(rounds, k, n):
    return [dict(rounds=int(r), failures=int(ki), shots=int(ni), p_l=ki / ni,
                 cp68=clopper_pearson(int(ki), int(ni)), cp95=clopper_pearson(int(ki), int(ni), 0.95))
            for r, ki, ni in zip(rounds, k, n)]


def priors_for(run):
    names = {int(k): v for k, v in run["physical"].items()}
    si = NoiseModel.si1000(0.01)
    out = {"si1000": si}
    cal = run_calibration(run)
    if cal is not None:
        out["calibrated"] = NoiseModel.from_calibration(cal, names)
    ch = CHAR.get(run["quantum_computer"])
    if ch is not None and ch.exists():
        out["si1000+char"] = si.with_characterization(json.loads(ch.read_text()), names)
    # Calibration-aware final readout: each data qubit's readout error is the
    # calibrated assignment error of the readout implementation this run used.
    ro = readout_metrics(run["quantum_computer"])
    if ro is not None:
        impl = {"qnd": "measure.constant",
                "fidelity": "measure_fidelity.shelved_constant"}[run.get("final_readout", "qnd")]
        err = {q: v["assignment_error"] for q, v in ro["metrics"][impl].items()}
        out["si1000+ro-cal"] = si.with_readout_errors(err, names, CODE.data_qubits)
        if "si1000+char" in out:
            out["si1000+char+ro-cal"] = out["si1000+char"].with_readout_errors(err, names, CODE.data_qubits)
    return out


def readout_metrics(quantum_computer: str):
    """Per-implementation readout metrics (qec.hardware.fetch_readout_metrics).
    All runs in this repository used one calibration set, checked here."""
    files = sorted((ROOT / "data").glob(f"readout_metrics_{quantum_computer}_*.json"))
    if not files:
        return None
    m = json.loads(files[-1].read_text())
    return m if m["calibration_set_id"] == CALSET.get(quantum_computer) else None


CALSET = {"emerald": "ecec1d0a-1529-4c90-9b02-d1b30ddaa2e3"}


# ------------------------------------------------------------- memory runs
def analyze_memory(path: Path, quick: bool):
    run = load_run(path)
    rounds, basis = [int(r) for r in run["rounds"]], run["basis"]
    data, indep_ok = {}, True
    for r in rounds:
        raw = run[f"raw_r{r}"]
        assert raw.shape[1] == 9 + 8 * r, (path.name, r, raw.shape)
        clean = build_memory_circuit(3, r, basis, True, None)
        det, obs = measurements_to_detections(raw, clean)
        assert det.shape[1] == 8 * r
        d2, o2 = numpy_detectors(raw, r, basis)
        indep_ok &= bool(np.array_equal(det, d2) and np.array_equal(obs[:, 0], o2))
        data[r] = (det, obs)

    priors = priors_for(run)
    decoders = ["none", "mwpm", "correlated"] + ([] if quick else ["bposd"])
    combos = [("si1000", d) for d in decoders]
    if not quick:
        combos += [(p, d) for p in priors if p != "si1000" for d in ("mwpm", "correlated", "bposd")]
    fails = {}
    for pname, dec in combos:
        fails[(pname, dec)] = {
            r: decode_failures(*data[r], build_memory_circuit(3, r, basis, True, priors[pname]), dec)
            for r in rounds}

    res = {"file": path.name, "basis": basis, "dd": bool(run.get("dd")),
           "dd_strategy": run.get("dd_strategy", "standard"),
           "pulse_level": bool(run.get("pulse_level", False)), "rep": run.get("rep"),
           "final_readout": run.get("final_readout", "qnd"),
           "twirl": bool(run.get("twirl", False)),
           "quantum_computer": run["quantum_computer"], "job_id": run.get("job_id"),
           "qubits": sorted(run["physical"].values(), key=lambda s: int(s[2:])),
           "calibration_found": run_calibration(run) is not None,
           "independent_detectors_match": indep_ok, "decoders": {}}
    for (pname, dec), f in fails.items():
        k = [int(f[r].sum()) for r in rounds]
        n = [len(f[r]) for r in rounds]
        res["decoders"][f"{pname}/{dec}"] = {"points": points(rounds, k, n),
                                             "fit": fit_summary(rounds, k, n)}

    if not quick:
        cand = [c for c in fails if c[1] != "none"]
        sel = {}
        for fold, (train, test) in enumerate(((0, 1), (1, 0))):
            def eps_on(c, part):
                k = [int(fails[c][r][part::2].sum()) for r in rounds]
                n = [len(fails[c][r][part::2]) for r in rounds]
                return fit_ler(rounds, k, n).eps
            best = min(cand, key=lambda c: eps_on(c, train))
            sel[f"fold{fold}"] = {"chosen": "/".join(best), "eps_train": eps_on(best, train),
                                  "eps_test": eps_on(best, test)}
        full_best = min(cand, key=lambda c: res["decoders"]["/".join(c)]["fit"]["A=1"]["eps"])
        res["held_out_selection"] = {
            **sel, "eps_test_mean": float(np.mean([v["eps_test"] for v in sel.values()])),
            "in_sample_best": "/".join(full_best),
            "in_sample_best_eps": res["decoders"]["/".join(full_best)]["fit"]["A=1"]["eps"],
            "n_candidates": len(cand)}
    return res


# --------------------------------------------------------- physical runs
def analyze_physical(path: Path):
    run = load_run(path)
    rounds = [int(r) for r in run["rounds"]]
    out = {"file": path.name, "dd": bool(run.get("dd")), "round_ns": run.get("round_ns"),
           "job_id": run.get("job_id"), "states": {}, "pooled": {}, "per_qubit": {}}
    err = {}
    for s in STATES:
        for r in rounds:
            err[(s, r)] = data_errors(run[f"raw_{STATE_KEY[s]}_r{r}"], r, s)
    fit_rounds = [r for r in rounds if r >= 1]
    for s in STATES:
        k = [int(err[(s, r)].sum()) for r in rounds]
        n = [err[(s, r)].size for r in rounds]
        out["states"][s] = {"points": points(rounds, k, n),
                            "fit": fit_summary(fit_rounds, k[-len(fit_rounds):], n[-len(fit_rounds):])}
    for name, group in (("Z (|0>,|1>)", "01"), ("X (|+>,|->)", "+-"), ("all four", "01+-")):
        k = [sum(int(err[(s, r)].sum()) for s in group) for r in rounds]
        n = [sum(err[(s, r)].size for s in group) for r in rounds]
        out["pooled"][name] = {"points": points(rounds, k, n),
                               "fit": fit_summary(fit_rounds, k[-len(fit_rounds):], n[-len(fit_rounds):])}
    phys = run["physical"]
    for i in range(9):
        k = [sum(int(err[(s, r)][:, i].sum()) for s in STATES) for r in fit_rounds]
        n = [sum(err[(s, r)].shape[0] for s in STATES) for r in fit_rounds]
        f = fit_ler(fit_rounds, k, n, free_amplitude=True)
        out["per_qubit"][phys[str(i)]] = {"eps": f.eps, "eps_err": f.eps_err, "A": f.amp}
    return out


# ------------------------------------------------------------------ report
def pct(x):
    return f"{100 * x:.2f}"


def write_markdown(summary):
    L = ["# Reproduced results (generated by experiments/reproduce_all.py — do not edit)", ""]
    L += ["Every number below is recomputed from the raw shots in `results/hw/`. "
          "Intervals: exact Clopper-Pearson (points), profile likelihood (fits). "
          "`G²/dof, p` is the deviance goodness-of-fit test of the exponential model; "
          "**rejected** = p < 0.01, in which case the fitted rate is not a meaningful per-round rate.", ""]
    L += ["## Memory runs (SI1000 p = 1 % prior)", "",
          "| run | basis | DD | qubits ok | decoder | failures / shots per n | ε_L (A=1) [68 % PL] | G²/dof, p | ε_L (A free), A |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in summary["memory"]:
        for key in ("si1000/none", "si1000/mwpm", "si1000/correlated", "si1000/bposd"):
            if key not in r["decoders"]:
                continue
            d = r["decoders"][key]
            f1, fa = d["fit"]["A=1"], d["fit"]["A free"]
            pts = " ".join(f"{p['failures']}/{p['shots']}" for p in d["points"])
            gof = f"{f1['g2']:.1f}/{f1['dof']}, {f1['p_gof']:.2g}" + ("" if f1["model_ok"] else " **rejected**")
            L.append(f"| {r['file'].replace('.npz', '')} | {r['basis']} | "
                     f"{'on' if r['dd'] else 'off'}{'' if r['dd_strategy'] == 'standard' else '-' + r['dd_strategy']}"
                     f"{' (pulse)' if r['pulse_level'] else ''}"
                     f"{', final readout: ' + r['final_readout'] if r['final_readout'] != 'qnd' else ''}"
                     f"{', Pauli-twirled' if r.get('twirl') else ''} | {'✓' if r['independent_detectors_match'] else '✗'} | "
                     f"{key.split('/')[1]} | {pts} | {pct(f1['eps'])} [{pct(f1['ci68'][0])}, {pct(f1['ci68'][1])}] | "
                     f"{gof} | {pct(fa['eps'])}, {fa['A']:.3f} |")
    hs = [r for r in summary["memory"] if "held_out_selection" in r]
    if hs:
        L += ["", "## Held-out decoder/prior selection", "",
              "Chosen on half of the shots, evaluated on the other half (2-fold); "
              "compare with the in-sample best of the same candidates.", "",
              "| run | candidates | chosen (fold 0 / fold 1) | ε_L held-out (mean) | in-sample best | ε_L in-sample |",
              "|---|---|---|---|---|---|"]
        for r in hs:
            h = r["held_out_selection"]
            L.append(f"| {r['file'].replace('.npz', '')} | {h['n_candidates']} | {h['fold0']['chosen']} / "
                     f"{h['fold1']['chosen']} | {pct(h['eps_test_mean'])} | {h['in_sample_best']} | "
                     f"{pct(h['in_sample_best_eps'])} |")
    if summary["physical"]:
        L += ["", "## Physical-qubit baseline (same 9 data qubits, same readout windows, no encoding)", "",
              "| run | DD | group | errors / samples per n | ε_phys per round (A free) [68 % PL] | A | G²/dof, p |",
              "|---|---|---|---|---|---|---|"]
        for p in summary["physical"]:
            for g, d in list(p["pooled"].items()) + [(f"|{s}>", v) for s, v in p["states"].items()]:
                fa = d["fit"]["A free"]
                pts = " ".join(f"{x['failures']}/{x['shots']}" for x in d["points"])
                gof = f"{fa['g2']:.1f}/{fa['dof']}, {fa['p_gof']:.2g}" + ("" if fa["model_ok"] else " **rejected**")
                L.append(f"| {p['file'].replace('.npz', '')} | {'on' if p['dd'] else 'off'} | {g} | {pts} | "
                         f"{pct(fa['eps'])} [{pct(fa['ci68'][0])}, {pct(fa['ci68'][1])}] | {fa['A']:.3f} | {gof} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="MWPM only, no held-out selection")
    ap.add_argument("--only", nargs="*", help="substrings of run files to include")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    def wanted(f):
        return not args.only or any(s in f for s in args.only)

    summary = {"memory": [], "physical": []}
    for f in sorted(glob.glob(str(HW / "2026*.npz"))):
        if "fake" in Path(f).name or not wanted(f):
            continue
        r = analyze_memory(Path(f), args.quick)
        summary["memory"].append(r)
        b = r["decoders"].get("si1000/bposd") or r["decoders"]["si1000/correlated"]
        print(f"{r['file']:<45} detectors match: {r['independent_detectors_match']}  "
              f"eps = {pct(b['fit']['A=1']['eps'])} %  p_gof = {b['fit']['A=1']['p_gof']:.2g}"
              + (f"  held-out {pct(r['held_out_selection']['eps_test_mean'])} %"
                 if "held_out_selection" in r else ""), flush=True)
        if not r["independent_detectors_match"]:
            raise SystemExit(f"detector mismatch in {f}")
    for f in sorted(glob.glob(str(HW / "phys_*.npz"))):
        if "fake" in Path(f).name or not wanted(f):
            continue
        p = analyze_physical(Path(f))
        summary["physical"].append(p)
        for g, d in p["pooled"].items():
            fa = d["fit"]["A free"]
            print(f"{p['file']:<45} {g:<12} eps_phys = {pct(fa['eps'])} %  A = {fa['A']:.3f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    write_markdown(summary)
    print(f"saved {OUT / 'summary.json'} and summary.md")


if __name__ == "__main__":
    main()
