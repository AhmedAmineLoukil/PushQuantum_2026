"""Logical vs physical memory on the same qubits, from results/reproduced/summary.json.

Left: P_L(n) of the interleaved session (Z and X memory, DD on, both repeats
pooled, BP+OSD SI1000) against the pooled physical-qubit error of the same 9
data qubits (DD on). Right: per-round rates with 68 % profile intervals.

    .venv/Scripts/python experiments/plot_breakeven.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec.analysis import fit_ler, p_logical_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SESSION = "20261121-233734"


def pooled(runs, basis, dd, key):
    sel = [r for r in runs if r["file"].startswith(SESSION) and r["basis"] == basis and r["dd"] == dd]
    pts = [r["decoders"][key]["points"] for r in sel]
    rounds = [p["rounds"] for p in pts[0]]
    k = [sum(p[i]["failures"] for p in pts) for i in range(len(rounds))]
    n = [sum(p[i]["shots"] for p in pts) for i in range(len(rounds))]
    return np.array(rounds), np.array(k), np.array(n)


def main():
    s = json.loads((ROOT / "results" / "reproduced" / "summary.json").read_text())
    key = "si1000/bposd" if "si1000/bposd" in s["memory"][0]["decoders"] else "si1000/correlated"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = []
    for i, (b, dd) in enumerate([("Z", True), ("X", True), ("Z", False), ("X", False)]):
        r, k, n = pooled(s["memory"], b, dd, key)
        f = fit_ler(r, k, n)
        p = k / n
        if dd:
            ax[0].errorbar(r, p, yerr=np.sqrt(p * (1 - p) / n), fmt="o", color=f"C{i}",
                           label=f"logical {b}-memory, DD on ({key.split('/')[1]})")
            x = np.linspace(0, r.max(), 100)
            ax[0].plot(x, p_logical_model(x, f.eps), color=f"C{i}", alpha=0.7)
        bars.append((f"logical {b}, DD {'on' if dd else 'off'}", f, "C0" if dd else "C7"))
    for p in s["physical"]:
        g = p["pooled"]["all four"]
        r = np.array([x["rounds"] for x in g["points"]])
        k = np.array([x["failures"] for x in g["points"]])
        n = np.array([x["shots"] for x in g["points"]])
        f = fit_ler(r[r >= 1], k[r >= 1], n[r >= 1], free_amplitude=True)
        if p["dd"]:
            q = k / n
            ax[0].errorbar(r, q, yerr=np.sqrt(q * (1 - q) / n), fmt="s", color="C3",
                           label="physical qubit (avg |0>,|1>,|+>,|->), DD on")
            x = np.linspace(0, r.max(), 100)
            ax[0].plot(x, p_logical_model(x, f.eps, f.amp), color="C3", alpha=0.7)
        bars.append((f"physical, DD {'on' if p['dd'] else 'off'}", f, "C3" if p["dd"] else "C1"))
    ax[0].set_xlabel("rounds n")
    ax[0].set_ylabel("error probability")
    ax[0].set_title("IQM Emerald, same 17 qubits, same session")
    ax[0].legend(fontsize=7)
    ax[1].barh(range(len(bars)), [100 * f.eps for _, f, _ in bars],
               xerr=[[100 * (f.eps - f.ci68[0]) for _, f, _ in bars],
                     [100 * (f.ci68[1] - f.eps) for _, f, _ in bars]],
               color=[c for *_, c in bars])
    ax[1].set_yticks(range(len(bars)), [lab + ("" if f.model_ok else "  (fit rejected)")
                                        for lab, f, _ in bars], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("error per round (%)")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "logical_vs_physical.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for lab, f, _ in bars:
        print(f"  {lab:<22} {f}")


if __name__ == "__main__":
    main()
