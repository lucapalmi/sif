#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
The complete emulator: inputs to multiplicity function, in one place.

This module is the SPECIFICATION. The C implementation has to reproduce these
feature definitions exactly -- not approximately, since the network was fitted
against these numbers and a feature computed a little differently is simply a
different input. Training and inference both call compute_features here, so
there is no second definition anywhere that could drift from this one.

The pipeline:

    radii, S, barrier, deriv_variance
      -> local description        (nu, gamma^2, y)          per radius
      -> up-crossing rate f_up    (Musso & Sheth)           per radius
      -> baseline hazard Lambda   (log-linear in the bin)   per bin
      -> history features         (lookback, cumulative)    per bin
      -> network                  log correction            per bin
      -> corrected hazard         Lambda * exp(correction)
      -> survival recursion       multiplicity              per bin

The correction multiplies a HAZARD, and the multiplicity is rebuilt through the
survival recursion, so the result is non-negative and integrates to at most one
whatever the network predicts. That bound is structural, not a clamp.
"""

import numpy as np
from scipy.special import erfc

FEATURE_NAMES = ["nu", "gamma2", "y", "dlnnu_dlnS", "nu_back", "lag",
                 "cum_lam", "nu_origin"]

# Above this the mean-excess bracket is taken from its asymptotic series;
# the closed form is a difference of two nearly equal numbers there.
Y_ASYMPTOTIC = 5.0

# Fraction of S looked back to, and the floor the lookback is clamped at.
LOOKBACK_FRACTION = 0.5


def upper_tail(x):
    """1 - Phi(x)."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def mean_excess(y):
    """E[(z - y)^+] for standard normal z: phi(y) - y (1 - Phi(y))."""
    y = np.asarray(y, dtype=float)
    phi = np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi)
    out = phi - y * upper_tail(y)

    big = y > Y_ASYMPTOTIC
    if np.any(big):
        yb = y[big]
        y2 = yb * yb
        y4 = y2 * y2
        out[big] = (np.exp(-0.5 * y2) / np.sqrt(2.0 * np.pi)) / y2 * (
            1.0 - 3.0 / y2 + 15.0 / y4 - 105.0 / (y4 * y2))
    return out


def _mid(a):
    return 0.5 * (a[:-1] + a[1:])


def baseline_rate(S, barrier, dvar):
    """nu, gamma^2, y and the up-crossing rate, per radius."""
    S = np.asarray(S, dtype=float)
    B = np.asarray(barrier, dtype=float)
    V = np.asarray(dvar, dtype=float)

    nu = B / np.sqrt(S)
    gamma2 = 1.0 / (4.0 * S * V)

    # Conditional on delta = B the slope has mean B/2S -- from <delta delta'>
    # = 1/2 exactly -- and variance V - 1/(4S).
    mu = B / (2.0 * S)
    slope_var = np.maximum(V - 1.0 / (4.0 * S), 0.0)
    sigma_slope = np.sqrt(slope_var)

    dB = np.gradient(B, S, edge_order=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(sigma_slope > 0, (dB - mu) / sigma_slope, 0.0)

    f_up = (np.exp(-0.5 * nu * nu) / np.sqrt(2.0 * np.pi * S)
            * sigma_slope * mean_excess(y))
    return nu, gamma2, y, np.maximum(f_up, 0.0)


def baseline_hazard(S, f_up):
    """Per-bin integrated hazard, log-linear in the rate.

    The rate carries exp(-nu^2/2) and varies exponentially across a bin, so a
    trapezoid makes the answer depend on how finely the caller sampled radii.
    Integrating the log-linear interpolant is exact for a pure exponential:
    the logarithmic mean of the endpoints.
    """
    S = np.asarray(S, dtype=float)
    f0 = f_up[1:]    # entered first: larger radius, smaller S
    f1 = f_up[:-1]
    dS = S[:-1] - S[1:]

    lam = 0.5 * (f0 + f1) * dS
    ok = (f0 > 0.0) & (f1 > 0.0)
    lr = np.zeros_like(lam)
    lr[ok] = np.log(f1[ok] / f0[ok])
    use = ok & (np.abs(lr) > 1e-6)
    lam[use] = (f1[use] - f0[use]) * dS[use] / lr[use]
    return np.maximum(lam, 0.0)


def compute_features(radii, S, barrier, dvar):
    """The eight features on the n-1 bin centres, plus the baseline hazard.

    Returns (X, lam_up, nu_at_largest_radius) where X has shape (n-1, 8) in
    the order of FEATURE_NAMES.
    """
    radii = np.asarray(radii, dtype=float)
    S = np.asarray(S, dtype=float)

    nu_r, gamma2_r, y_r, f_up = baseline_rate(S, barrier, dvar)
    lam_up = baseline_hazard(S, f_up)

    nu = _mid(nu_r)
    gamma2 = _mid(gamma2_r)
    y = _mid(y_r)
    S_mid = _mid(S)

    # Local shape of the barrier in the walk's own time.
    dlnnu_dlnS = np.gradient(np.log(nu), np.log(S_mid), edge_order=2)

    # Lookback, clamped at the walk's origin: the walk BEGINS at the largest
    # radius, so there is no history before it to extrapolate into.
    target = np.maximum(LOOKBACK_FRACTION * S_mid, S[-1])
    lag = np.log(S_mid / target)
    nu_back = np.interp(target, S_mid[::-1], nu[::-1])

    # Hazard accumulated BEFORE entering each bin. The walk enters at the
    # largest radius (highest index) and works down, so this is the reverse
    # cumulative sum, excluding the bin itself. It summarizes the whole prior
    # history in one number, which is what the surviving population is
    # conditioned on.
    cum_lam = np.concatenate(
        [np.cumsum(lam_up[::-1])[::-1][1:], [0.0]])

    nu_origin = np.full(len(nu), nu[-1])

    X = np.column_stack([nu, gamma2, y, dlnnu_dlnS, nu_back, lag,
                         cum_lam, nu_origin])
    return X, lam_up, nu_r[-1]


def mc_hazard(counts, n_paths):
    """Per-bin hazard from raw first-crossing counts: the training target.

    The inverse of the survival recursion below. counts[i] is the number of
    walks first crossing at radius i, ascending; the walk runs from the highest
    index down, so the population still walking when it reaches i is everything
    that has not crossed above it. The conditional crossing probability there
    is h = q / alive, and the integrated hazard is -ln(1 - h).

    counts[n-1] is the first-step point mass. It has no bin of its own but it
    does deplete the population for every bin below, which is why it enters
    `alive` and is then dropped from the returned arrays.

    Returns (lambda, relative_error, alive) on the n-1 bins.
    """
    counts = np.asarray(counts, dtype=float)
    q = counts / float(n_paths)

    above = np.concatenate(([0.0], np.cumsum(q[::-1])[:-1]))[::-1]
    alive = 1.0 - above

    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(alive > 0.0, q / alive, 0.0)
    lam = -np.log1p(-np.minimum(h, 1.0 - 1e-15))

    # Binomial error on the conditional hazard, propagated: dLambda = dh/(1-h).
    n_eff = np.maximum(alive * n_paths, 1.0)
    dh = np.sqrt(np.maximum(h * (1.0 - h), 0.0) / n_eff)
    dlam = dh / np.maximum(1.0 - h, 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(lam > 0.0, dlam / lam, np.inf)

    return lam[:-1], rel[:-1], alive[:-1]


def multiplicity_from_hazard(lam, radii, nu_at_largest):
    """Survival recursion: per-bin hazards to the multiplicity function.

    Walks that begin above the barrier cross on the first step and never enter
    a bin. That point mass is the one-point tail at the walk's origin, which is
    exactly analytic, so it is removed here rather than estimated.
    """
    radii = np.asarray(radii, dtype=float)
    n_bins = len(lam)
    f = np.zeros(n_bins)

    alive = 1.0 - float(upper_tail(nu_at_largest))
    for i in range(n_bins - 1, -1, -1):
        p = alive * (1.0 - np.exp(-lam[i]))
        alive -= p
        dr = radii[i + 1] - radii[i]
        f[i] = p / dr if dr > 0 else 0.0
    return f


class EPEmulator:
    """The trained correction, plus everything needed to apply and police it."""

    def __init__(self, mu, sd, layers, box=None, nu_origin_min=None,
                 accuracy=None):
        self.mu = np.asarray(mu, dtype=float)
        self.sd = np.asarray(sd, dtype=float)
        self.layers = [(np.asarray(W, dtype=float), np.asarray(b, dtype=float))
                       for (W, b) in layers]
        self.box = box                    # (lo, hi) per feature, from training
        self.nu_origin_min = nu_origin_min
        self.accuracy = accuracy or {}

    # --- network ------------------------------------------------------------

    def correction(self, X):
        """log(Lambda_MC / Lambda_up) for each bin."""
        h = (np.asarray(X, dtype=float) - self.mu) / self.sd
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h.ravel()

    # --- the whole thing ----------------------------------------------------

    def multiplicity(self, radii, S, barrier, dvar, check=True):
        """The emulated multiplicity function on the n-1 bin centres.

        Returns (f, info) where info carries the domain report.
        """
        X, lam_up, nu_large = compute_features(radii, S, barrier, dvar)
        info = self.check_domain(X, nu_large) if check else {}

        lam = lam_up * np.exp(self.correction(X))
        f = multiplicity_from_hazard(lam, radii, nu_large)
        return f, info

    # --- domain -------------------------------------------------------------

    def check_domain(self, X, nu_large):
        """What the training actually covered, and where this call sits.

        Two distinct failures. A feature outside the trained box is ordinary
        extrapolation. A walk origin that is too low is different in kind: a
        sizeable fraction of walks then START above the barrier, which is the
        one configuration the emulator was measured to handle badly (0.8%
        typical, 11% at the tail). It is also entirely the caller's to fix, by
        extending the radius grid outward.
        """
        info = dict(ok=True, messages=[], outside=0)

        first_step = float(upper_tail(nu_large))
        if self.nu_origin_min is not None and nu_large < self.nu_origin_min:
            info["ok"] = False
            info["messages"].append(
                f"nu at the largest radius is {nu_large:.2f}, below the "
                f"trained minimum of {self.nu_origin_min:.2f}: {first_step:.1%} "
                f"of walks start above the barrier and never enter a bin. "
                f"Extend the radius grid outward.")

        if self.box is not None:
            lo, hi = self.box
            below = X < lo
            above = X > hi
            n_out = int(np.any(below | above, axis=1).sum())
            info["outside"] = n_out
            if n_out:
                worst = np.argmax(np.maximum(
                    (lo - X).max(axis=0), (X - hi).max(axis=0)))
                info["ok"] = False
                info["messages"].append(
                    f"{n_out} of {len(X)} bins fall outside the trained "
                    f"feature range; {FEATURE_NAMES[worst]} is the furthest "
                    f"out (trained {lo[worst]:.4g} to {hi[worst]:.4g}, "
                    f"given {X[:, worst].min():.4g} to "
                    f"{X[:, worst].max():.4g}).")

        info["first_step_mass"] = first_step
        info["nu_origin"] = float(nu_large)
        return info

    # --- persistence ---------------------------------------------------------

    def save(self, path):
        store = dict(mu=self.mu, sd=self.sd, n_layers=len(self.layers),
                     features=np.array(FEATURE_NAMES))
        for i, (W, b) in enumerate(self.layers):
            store[f"W{i}"] = W
            store[f"b{i}"] = b
        if self.box is not None:
            store["box_lo"], store["box_hi"] = self.box
        if self.nu_origin_min is not None:
            store["nu_origin_min"] = self.nu_origin_min
        for k, v in self.accuracy.items():
            store[f"acc__{k}"] = v
        np.savez(path, **store)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))]
        box = ((d["box_lo"], d["box_hi"]) if "box_lo" in d else None)
        nmin = float(d["nu_origin_min"]) if "nu_origin_min" in d else None
        acc = {k[5:]: float(d[k]) for k in d.files if k.startswith("acc__")}
        return EPEmulator(d["mu"], d["sd"], layers, box, nmin, acc)

    def n_parameters(self):
        return sum(W.size + b.size for (W, b) in self.layers)
