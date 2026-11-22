"""Logical error rate per round from a sweep over round counts.

Model (Google Quantum AI, Nature 2021 / 2023): the logical fidelity
F(n) = 1 - 2 P_L(n) decays exponentially with the number of rounds n,

    F(n) = A (1 - 2 eps_L)^n,

where eps_L is the logical error rate per round. The challenge README uses
A = 1; letting A float absorbs state-preparation and final-readout error so
that eps_L reflects only the per-round cost. We report both.

Fitting is a binomial maximum likelihood over all (n, errors, shots) points,
with the 1-sigma uncertainty from the observed Fisher information (inverse
Hessian of the negative log-likelihood).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def p_logical_model(n, eps, amp=1.0):
    n = np.asarray(n, dtype=float)
    return 0.5 * (1 - amp * (1 - 2 * eps) ** n)


def eps_from_single(p_l: float, n: int) -> float:
    """Per-round rate from one round count, A = 1 (README closed form)."""
    f = 1 - 2 * p_l
    if f <= 0:
        return 0.5
    return 0.5 * (1 - f ** (1 / n))


@dataclass
class LERFit:
    eps: float
    eps_err: float
    amp: float
    amp_err: float
    free_amplitude: bool
    rounds: np.ndarray
    p_obs: np.ndarray
    # Profile-likelihood intervals for eps (A profiled out when free), the
    # likelihood-ratio (deviance) goodness-of-fit statistic G^2 against the
    # saturated binomial model, its degrees of freedom and p-value, and the
    # Pearson residuals (obs - model) / sigma_model per fitted point.
    ci68: tuple[float, float] = (np.nan, np.nan)
    ci95: tuple[float, float] = (np.nan, np.nan)
    g2: float = np.nan
    dof: int = 0
    p_gof: float = np.nan
    residuals: np.ndarray | None = None

    @property
    def model_ok(self) -> bool:
        """False when the exponential model is rejected at the 1 % level: then
        eps_L and its interval should not be quoted as a per-round rate."""
        return not (self.dof > 0 and self.p_gof < 0.01)

    def __repr__(self):
        s = (f"eps_L = {100*self.eps:.3f} +/- {100*self.eps_err:.3f} % per round "
             f"(68% PL [{100*self.ci68[0]:.3f}, {100*self.ci68[1]:.3f}])")
        if self.free_amplitude:
            s += f"  (A = {self.amp:.3f} +/- {self.amp_err:.3f})"
        if self.dof > 0:
            s += f"  G2 = {self.g2:.1f}/{self.dof} dof, p = {self.p_gof:.2g}"
            if not self.model_ok:
                s += "  MODEL REJECTED"
        return s


def clopper_pearson(k: int, n: int, cl: float = 0.6827) -> tuple[float, float]:
    """Exact binomial confidence interval."""
    from scipy.stats import beta
    a = 1 - cl
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def wilson(k: int, n: int, z: float = 1.0) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def fit_ler(rounds, errors, shots, free_amplitude: bool = False,
            min_round: int = 1) -> LERFit:
    """Binomial-MLE fit of P_L(n) = (1 - A (1-2 eps)^n) / 2.

    min_round : drop round counts below this (Google excludes the first
                cycle(s), which carry extra boundary/state-prep error).
    """
    rounds = np.asarray(rounds, dtype=float)
    errors = np.asarray(errors, dtype=float)
    shots = np.asarray(shots, dtype=float)
    keep = rounds >= min_round
    r, k, N = rounds[keep], errors[keep], shots[keep]

    def unpack(theta):
        eps = 0.5 / (1 + np.exp(-theta[0]))          # eps in (0, 0.5)
        amp = 1 / (1 + np.exp(-theta[1])) if free_amplitude else 1.0
        return eps, amp

    def nll(theta):
        eps, amp = unpack(theta)
        p = np.clip(p_logical_model(r, eps, amp), 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(p) + (N - k) * np.log(1 - p))

    # Initial guess from the pooled single-round estimates.
    eps0 = np.clip(np.median([eps_from_single(ki / Ni, int(ri))
                              for ri, ki, Ni in zip(r, k, N)]), 1e-5, 0.45)
    th0 = [np.log(2 * eps0 / (1 - 2 * eps0))]
    if free_amplitude:
        th0.append(2.0)
    res = minimize(nll, th0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    eps, amp = unpack(res.x)

    # Uncertainty: numerical Hessian in the natural parameters (eps, amp).
    def nll_nat(v):
        e = v[0]
        a = v[1] if free_amplitude else 1.0
        p = np.clip(p_logical_model(r, e, a), 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(p) + (N - k) * np.log(1 - p))

    x0 = np.array([eps, amp]) if free_amplitude else np.array([eps])
    H = _hessian(nll_nat, x0)
    try:
        cov = np.linalg.inv(H)
        errs = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        errs = np.full(len(x0), np.nan)

    # Profile likelihood for eps (amplitude re-optimised at each eps if free).
    nll_min = nll_nat(x0)

    def profile(e):
        if not free_amplitude:
            return nll_nat([e]) - nll_min
        from scipy.optimize import minimize_scalar
        r_ = minimize_scalar(lambda a: nll_nat([e, a]), bounds=(1e-6, 1.0), method="bounded")
        return r_.fun - nll_min

    def bound(level, direction):
        from scipy.optimize import brentq
        lo_lim, hi_lim = 1e-9, 0.5 - 1e-9
        f = lambda e: profile(e) - level  # noqa: E731
        edge = lo_lim if direction < 0 else hi_lim
        if f(edge) < 0:
            return edge
        return brentq(f, edge, eps) if direction < 0 else brentq(f, eps, edge)

    try:
        ci68 = (bound(0.5, -1), bound(0.5, +1))
        ci95 = (bound(1.920729, -1), bound(1.920729, +1))
    except ValueError:
        ci68 = ci95 = (np.nan, np.nan)

    # Goodness of fit: deviance vs the saturated model.
    from scipy.stats import chi2
    pm = np.clip(p_logical_model(r, eps, amp), 1e-12, 1 - 1e-12)
    ph = k / N
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = 2 * np.sum(np.where(k > 0, k * np.log(ph / pm), 0.0)
                        + np.where(N - k > 0, (N - k) * np.log((1 - ph) / (1 - pm)), 0.0))
    dof = len(r) - len(x0)
    p_gof = float(chi2.sf(g2, dof)) if dof > 0 else np.nan
    resid = (ph - pm) / np.sqrt(pm * (1 - pm) / N)

    return LERFit(eps=float(eps), eps_err=float(errs[0]), amp=float(amp),
                  amp_err=float(errs[1]) if free_amplitude else 0.0,
                  free_amplitude=free_amplitude, rounds=rounds,
                  p_obs=errors / shots, ci68=tuple(map(float, ci68)),
                  ci95=tuple(map(float, ci95)), g2=float(g2), dof=int(dof),
                  p_gof=p_gof, residuals=resid)


def _hessian(f, x, rel=1e-4):
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = np.maximum(np.abs(x) * rel, 1e-7)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h[i]
            ej = np.zeros(n); ej[j] = h[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej)
                       + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def plot_ler(ax, rounds, p_obs, p_err, fit: LERFit | None = None, label="", color=None):
    """P_L vs rounds with optional fitted curve (Fig. 3 style, 2023 paper)."""
    ax.errorbar(rounds, p_obs, yerr=p_err, fmt="o", color=color, label=label,
                capsize=3, ms=5)
    if fit is not None:
        n = np.linspace(min(rounds), max(rounds), 200)
        ax.plot(n, p_logical_model(n, fit.eps, fit.amp), "-", color=color, alpha=0.8,
                label=f"fit: $\\varepsilon_L$ = {100*fit.eps:.2f}%")
    ax.set_xlabel("rounds $n$")
    ax.set_ylabel("logical error probability $P_L$")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
