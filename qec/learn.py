"""Data-driven decoder weights: learn the error model from the detection events.

Calibration numbers describe gates in isolation; they miss whatever happens
only inside this circuit (measurement-induced errors on an un-reset ancilla,
leakage, crosstalk, TLS defects ...). The detection events themselves contain
that information: an error mechanism that flips detectors i and j makes them
fire *together* more often than chance. Inverting that correlation gives the
probability of every edge of the matching graph (Google Quantum AI, Nature
2021, supplementary "p_ij" method):

    p_ij = 1/2 - 1/2 sqrt( 1 - 4 (<x_i x_j> - <x_i><x_j>) / (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>) )

Boundary edges get whatever firing rate of detector i the pair edges do not
explain:  1 - 2<x_i> = (1 - 2 p_i,boundary) * prod_j (1 - 2 p_ij).

The graph *structure* (which detector pairs can be flipped together, and
which edges flip the logical observable) comes from the circuit's DEM; only
the probabilities are learned. To avoid scoring a decoder on the same shots
it was fit to, `cross_validated_decode` learns on one fold and decodes the
other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymatching
import stim

from .decode import DecodeResult

P_MIN, P_MAX = 1e-5, 0.499


@dataclass
class Edge:
    dets: tuple[int, ...]       # one detector (boundary edge) or two
    obs: frozenset[int]
    p_prior: float


def graph_edges(noisy_circuit: stim.Circuit) -> list[Edge]:
    """Graphlike edges of the circuit's decomposed DEM (structure + prior p)."""
    dem = noisy_circuit.detector_error_model(decompose_errors=True)
    edges: dict[tuple, Edge] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comp: list[list] = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                comp.append([])
            else:
                comp[-1].append(t)
        for c in comp:
            dets = tuple(sorted(t.val for t in c if t.is_relative_detector_id()))
            obs = frozenset(t.val for t in c if t.is_logical_observable_id())
            if not dets or len(dets) > 2:
                continue
            if dets in edges:
                e = edges[dets]
                # independent mechanisms on the same edge combine as XOR
                e.p_prior = e.p_prior * (1 - p) + p * (1 - e.p_prior)
            else:
                edges[dets] = Edge(dets, obs, p)
    return list(edges.values())


def estimate_edge_probabilities(det_events: np.ndarray, edges: list[Edge]) -> dict[tuple, float]:
    x = np.asarray(det_events, dtype=np.float64)
    n = len(x)
    mean = x.mean(axis=0)
    xx = (x.T @ x) / n
    p: dict[tuple, float] = {}
    for e in edges:
        if len(e.dets) != 2:
            continue
        i, j = e.dets
        num = 4 * (xx[i, j] - mean[i] * mean[j])
        den = 1 - 2 * mean[i] - 2 * mean[j] + 4 * xx[i, j]
        if num <= 0 or den <= 0:
            p[e.dets] = P_MIN
            continue
        val = 1 - num / den
        p[e.dets] = float(np.clip(0.5 - 0.5 * np.sqrt(max(val, 0.0)), P_MIN, P_MAX))

    pairs = dict(p)
    for e in edges:
        if len(e.dets) != 1:
            continue
        i = e.dets[0]
        prod = 1.0
        for (a, b), pij in pairs.items():
            if i in (a, b):
                prod *= 1 - 2 * pij
        ratio = (1 - 2 * mean[i]) / prod if prod > 0 else 0.0
        p[e.dets] = float(np.clip(0.5 - 0.5 * ratio, P_MIN, P_MAX))
    return p


def build_matching(edges: list[Edge], probs: dict[tuple, float], num_detectors: int
                   ) -> pymatching.Matching:
    m = pymatching.Matching()
    for e in edges:
        p = probs.get(e.dets, e.p_prior)
        w = float(np.log((1 - p) / p))
        if len(e.dets) == 2:
            m.add_edge(*e.dets, fault_ids=set(e.obs), weight=w, error_probability=p,
                       merge_strategy="independent")
        else:
            m.add_boundary_edge(e.dets[0], fault_ids=set(e.obs), weight=w,
                                error_probability=p, merge_strategy="independent")
    # make sure every detector exists as a node, even if it has no edges
    if m.num_detectors < num_detectors:
        m.add_boundary_edge(num_detectors - 1, weight=float(np.log((1 - P_MIN) / P_MIN)),
                            error_probability=P_MIN, merge_strategy="independent")
    return m


def cross_validated_decode(det_events, obs_flips, noisy_circuit: stim.Circuit,
                           folds: int = 2, seed: int = 0) -> tuple[DecodeResult, dict]:
    """Learn edge probabilities on k-1 folds, decode the held-out fold.

    Returns the pooled DecodeResult ("learned") and the edge probabilities
    learned on the full data set (for diagnostics only, never for scoring).
    """
    det = np.asarray(det_events, dtype=bool)
    obs = np.asarray(obs_flips, dtype=bool).reshape(len(det), -1)
    edges = graph_edges(noisy_circuit)
    nd = noisy_circuit.num_detectors
    idx = np.random.default_rng(seed).permutation(len(det))
    errors = 0
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        probs = estimate_edge_probabilities(det[train], edges)
        pred = build_matching(edges, probs, nd).decode_batch(det[test])
        errors += int(np.any(pred != obs[test], axis=1).sum())
    full = estimate_edge_probabilities(det, edges)
    return DecodeResult("learned", len(det), errors), {"edges": edges, "p": full}


def summarize_learned(noisy_circuit: stim.Circuit, learned: dict, code, physical: dict
                      ) -> list[dict]:
    """Per-stabilizer learned error rates.

    * time-like edges connect the same stabilizer in consecutive rounds; they
      are dominated by ancilla measurement / ancilla idling errors.
    * space-like edges connect different stabilizers in the same round (data
      qubit errors, CZ errors).
    """
    coords = noisy_circuit.get_detector_coordinates()
    anc_at = {code.coords[s.ancilla]: s for s in code.stabilizers}
    rows = {}
    for e in learned["edges"]:
        if len(e.dets) != 2:
            continue
        (xa, ya, ta), (xb, yb, tb) = coords[e.dets[0]][:3], coords[e.dets[1]][:3]
        if (xa, ya) == (xb, yb) and abs(ta - tb) == 1:
            s = anc_at[(int(xa), int(ya))]
            r = rows.setdefault(s.ancilla, {"timelike": [], "prior": []})
            r["timelike"].append(learned["p"][e.dets])
            r["prior"].append(e.p_prior)
    out = []
    for anc, r in sorted(rows.items()):
        s = next(st for st in code.stabilizers if st.ancilla == anc)
        out.append({"stabilizer": f"{s.basis}@{physical[str(anc)]}",
                    "timelike_learned": float(np.mean(r["timelike"])),
                    "timelike_prior": float(np.mean(r["prior"]))})
    return out
