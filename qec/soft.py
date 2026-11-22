"""Soft-information decoding from IQ readout (pulse-level `acquisition_type="complex"`).

Each readout returns an integrated IQ value z instead of a bit. With reference
shots of every qubit prepared in |0> and |1> we know where the two clouds sit.
For every readout of every shot we compute

    p_misread = P(the recorded bit is wrong | z),

from a two-Gaussian model along the axis joining the cloud centres. A readout
deep in its cloud gets p ~ 0, one between the clouds gets p ~ 0.5.

Decoding: a classical misread of measurement record j flips a fixed set of
detectors (and possibly the observable) - one edge of the matching graph. The
soft decoder sets that edge's probability shot by shot,
p_edge = p_rest XOR p_misread(shot, j), where p_rest is the same edge's
probability from every other error mechanism of the prior. The hard-decision
control uses the same graph with each record's AVERAGE misread probability, so
the comparison isolates the value of per-shot readout confidence.
(Pattison, Beverland, da Silva, Delfosse, "Improved quantum error correction
using soft information", 2021.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymatching
import stim

P_MIN, P_MAX = 1e-6, 0.4999


# ------------------------------------------------------------- IQ model
@dataclass
class IQModel:
    """Per readout column: cloud centres mu0, mu1 (complex) and the noise
    width sigma along the axis joining them."""
    mu0: np.ndarray
    mu1: np.ndarray
    sigma: np.ndarray

    def axis_coordinate(self, z: np.ndarray) -> np.ndarray:
        """Position along mu0 -> mu1, in units where mu0 = 0 and mu1 = 1."""
        d = self.mu1 - self.mu0
        return ((z - self.mu0) * np.conj(d)).real / np.abs(d) ** 2

    def llr(self, z: np.ndarray) -> np.ndarray:
        """log P(z | 1) / P(z | 0) for equal-width Gaussians along the axis."""
        x = self.axis_coordinate(z)
        s = self.sigma / np.abs(self.mu1 - self.mu0)          # width in axis units
        return ((x - 0.0) ** 2 - (x - 1.0) ** 2) / (2 * s ** 2)

    def misread_probability(self, z: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """P(recorded bit is wrong | z), clipped to [P_MIN, P_MAX]."""
        llr = np.clip(self.llr(z), -50, 50)
        p_one = 1.0 / (1.0 + np.exp(-llr))
        p_wrong = np.where(bits, 1.0 - p_one, p_one)
        return np.clip(p_wrong, P_MIN, P_MAX)


def fit_iq_model(cal0: np.ndarray, cal1: np.ndarray) -> IQModel:
    """Cloud centres from the medians of the |0> and |1> reference shots (robust
    to the |1> -> |0> decays and to excitations), width from the median absolute
    deviation of each cloud along the axis (pooled)."""
    mu0 = np.median(cal0.real, axis=0) + 1j * np.median(cal0.imag, axis=0)
    mu1 = np.median(cal1.real, axis=0) + 1j * np.median(cal1.imag, axis=0)
    d = mu1 - mu0
    u = d / np.abs(d)
    x0 = ((cal0 - mu0) * np.conj(u)).real
    x1 = ((cal1 - mu1) * np.conj(u)).real
    mad = 0.5 * (np.median(np.abs(x0), axis=0) + np.median(np.abs(x1), axis=0))
    return IQModel(mu0=mu0, mu1=mu1, sigma=1.4826 * mad)


def fit_iq_mixture(z: np.ndarray, threshold: float, iters: int = 100) -> IQModel:
    """Learn one qubit's |0> and |1> clouds from ALL its readouts in a run
    (no reference circuits needed): a two-component isotropic 2D Gaussian
    mixture fitted by EM, started from the on-chip rule Re(z) > threshold
    = |1>. Returns an IQModel for a single column (arrays of length 1)."""
    z = np.asarray(z).ravel()
    one = z.real > threshold
    mu0, mu1 = z[~one].mean(), z[one].mean()
    var = 0.5 * (np.var(z[~one]) + np.var(z[one]))        # complex variance = 2 sigma^2
    w1 = one.mean()
    for _ in range(iters):
        l0 = (1 - w1) * np.exp(-np.abs(z - mu0) ** 2 / var)
        l1 = w1 * np.exp(-np.abs(z - mu1) ** 2 / var)
        g = l1 / np.maximum(l0 + l1, 1e-300)
        w1 = g.mean()
        mu0 = np.sum((1 - g) * z) / np.sum(1 - g)
        mu1 = np.sum(g * z) / np.sum(g)
        var = np.sum((1 - g) * np.abs(z - mu0) ** 2 + g * np.abs(z - mu1) ** 2) / len(z)
    return IQModel(mu0=np.array([mu0]), mu1=np.array([mu1]), sigma=np.array([np.sqrt(var / 2)]))


def measured_qubits(clean: stim.Circuit) -> list[int]:
    """Dense qubit measured by each record, in record (= clbit) order."""
    return [t.value for inst in clean.flattened() if inst.name == "M" for t in inst.targets_copy()]


def soft_inputs(run: dict, basis: str | None = None) -> dict:
    """From a saved IQ run: per round count, (bits, p_misread, leakage_flags).

    bits follow the on-chip rule Re(z) > integration_threshold (so they equal
    what threshold acquisition would have returned); p_misread comes from a
    per-qubit mixture model fitted to every readout of that qubit in the run.
    """
    from .circuit import build_memory_circuit

    basis = basis or run["basis"]
    th = np.asarray(run["iq_threshold"])
    rounds = [int(r) for r in run["rounds"]]
    cols = {r: measured_qubits(build_memory_circuit(3, r, basis, True, None)) for r in rounds}
    per_q: dict[int, list] = {}
    for r in rounds:
        z = run[f"iq_r{r}"]
        for j, q in enumerate(cols[r]):
            per_q.setdefault(q, []).append(z[:, j])
    models = {q: fit_iq_mixture(np.concatenate(v), th[q]) for q, v in per_q.items()}
    out = {"models": models}
    for r in rounds:
        z = run[f"iq_r{r}"]
        bits = z.real > th[cols[r]]
        p = np.empty(z.shape)
        leak = np.zeros(z.shape, dtype=bool)
        for j, q in enumerate(cols[r]):
            m = models[q]
            p[:, j] = m.misread_probability(z[:, j], bits[:, j])
            leak[:, j] = leakage_like(m, z[:, j])
        out[r] = (bits, p, leak)
    return out


def leakage_like(model: IQModel, z: np.ndarray, n_sigma: float = 5.0) -> np.ndarray:
    """Readouts far outside both clouds (e.g. the |2> state): diagnostic only."""
    d0 = np.abs(z - model.mu0) / model.sigma
    d1 = np.abs(z - model.mu1) / model.sigma
    return (d0 > n_sigma) & (d1 > n_sigma)


# --------------------------------------------------- record -> graph edge
def record_effects(clean: stim.Circuit) -> list[tuple[tuple[int, ...], bool]]:
    """For every measurement record j: the detectors and observable flip that a
    classical misread of record j causes (from the circuit's own detector
    definitions, so the no-reset structure is handled automatically)."""
    n = clean.num_measurements
    conv = clean.compile_m2d_converter()
    ref = np.zeros((1, n), dtype=bool)
    det0, obs0 = conv.convert(measurements=ref, separate_observables=True)
    eye = np.eye(n, dtype=bool)
    det, obs = conv.convert(measurements=eye, separate_observables=True)
    out = []
    for j in range(n):
        d = tuple(np.flatnonzero(det[j] ^ det0[0]).tolist())
        o = bool((obs[j] ^ obs0[0]).any())
        if len(d) > 2:
            raise ValueError(f"record {j} flips {len(d)} detectors: not a graph edge")
        out.append((d, o))
    return out


def _xor(p, q):
    return p * (1 - q) + q * (1 - p)


class SoftMatcher:
    """Minimum-weight matching with per-shot weights on the misread edges."""

    def __init__(self, prior_without_misreads: stim.Circuit, clean: stim.Circuit):
        dem = prior_without_misreads.detector_error_model(decompose_errors=True)
        self.matching = pymatching.Matching.from_detector_error_model(dem)
        self.n_det = clean.num_detectors
        self.effects = record_effects(clean)
        # Group records that hit the same edge; remember the edge's prior probability.
        self.edges: dict[tuple[int, ...], dict] = {}
        for j, (dets, obs) in enumerate(self.effects):
            if not dets:
                continue                       # flips nothing a detector sees
            key = dets
            e = self.edges.setdefault(key, {"records": [], "obs": obs})
            if e["obs"] != obs:
                raise ValueError(f"records on edge {key} disagree on the observable")
            e["records"].append(j)
        for key, e in self.edges.items():
            e["p_rest"] = self._edge_probability(key)
            e["fault_ids"] = {0} if e["obs"] else set()

    def _edge_probability(self, key) -> float:
        try:
            if len(key) == 2:
                data = self.matching.get_edge_data(*key)
            else:
                data = self.matching.get_boundary_edge_data(key[0])
            return float(data["error_probability"])
        except (ValueError, KeyError):
            return 0.0

    def _set_edges(self, p_records: np.ndarray):
        for key, e in self.edges.items():
            p = e["p_rest"]
            for j in e["records"]:
                p = _xor(p, p_records[j])
            p = float(np.clip(p, P_MIN, P_MAX))
            w = float(np.log((1 - p) / p))
            if len(key) == 2:
                self.matching.add_edge(key[0], key[1], fault_ids=e["fault_ids"], weight=w,
                                       error_probability=p, merge_strategy="replace")
            else:
                self.matching.add_boundary_edge(key[0], fault_ids=e["fault_ids"], weight=w,
                                                error_probability=p, merge_strategy="replace")

    def decode(self, det_events: np.ndarray, p_records: np.ndarray) -> np.ndarray:
        """det_events (shots, n_det); p_records (shots, n_meas) or (n_meas,) for
        one weight set for every shot. Returns predicted observable flips."""
        det_events = np.asarray(det_events, dtype=bool)
        p_records = np.asarray(p_records, dtype=float)
        if p_records.ndim == 1:
            self._set_edges(p_records)
            return self.matching.decode_batch(det_events)
        out = np.zeros((len(det_events), self.matching.num_fault_ids or 1), dtype=bool)
        for k in range(len(det_events)):
            self._set_edges(p_records[k])
            out[k] = self.matching.decode(det_events[k])
        return out


def without_misreads(model):
    """Copy of a NoiseModel with every CLASSICAL misread removed (persisting
    state flips during readout are kept): the soft information replaces it."""
    from copy import deepcopy
    m = deepcopy(model)
    m.p_meas_record_q = {}
    return m
