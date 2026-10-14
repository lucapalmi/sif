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
      -> up-crossing rate f_up    (Verza+ 2024, eq. 3.15)   per radius
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
                 "cum_lam", "nu_origin", "gamma2_back"]

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

    # dB/dS through the logarithms, (B/S) d ln B / d ln S: a barrier is close
    # to a power law in sigma, so ln B against ln S is nearly straight and the
    # three-point stencil is exact for it, where on B against S it carries an
    # error of second order in the spacing -- the larger part of what made
    # f_up, and so the answer, depend on the caller's radius sampling.
    if np.all(B > 0.0):
        dB = B / S * np.gradient(np.log(B), np.log(S), edge_order=2)
    else:
        dB = np.gradient(B, S, edge_order=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(sigma_slope > 0, (dB - mu) / sigma_slope, 0.0)

    f_up = (np.exp(-0.5 * nu * nu) / np.sqrt(2.0 * np.pi * S)
            * sigma_slope * mean_excess(y))
    return nu, gamma2, y, np.maximum(f_up, 0.0)


# 4-point Gauss-Legendre on [-1, 1].
GL_X = np.array([-0.8611363115940526, -0.3399810435848563,
                 0.3399810435848563, 0.8611363115940526])
GL_W = np.array([0.3478548451374538, 0.6521451548625461,
                 0.6521451548625461, 0.3478548451374538])


def _baseline_hazard_logmean(S, f_up):
    """Logarithmic mean of the endpoints: the rule for fewer than four radii."""
    f0, f1 = f_up[1:], f_up[:-1]
    dS = S[:-1] - S[1:]
    lam = 0.5 * (f0 + f1) * dS
    ok = (f0 > 0.0) & (f1 > 0.0)
    lr = np.zeros_like(lam)
    lr[ok] = np.log(f1[ok] / f0[ok])
    use = ok & (np.abs(lr) > 1e-6)
    lam[use] = (f1[use] - f0[use]) * dS[use] / lr[use]
    return np.maximum(lam, 0.0)


def baseline_hazard(S, f_up):
    """Per-bin integrated hazard: ln f_up interpolated, then Gauss-Legendre.

    The rate carries exp(-nu^2/2), so ln f_up is smooth where f_up itself is
    steep. Over each bin it is replaced by the cubic through the four radii
    nearest the bin -- the bin's two ends and one neighbour either side, or the
    four at the end of the grid for the outermost bins -- and the exponential
    of that cubic is integrated with four Gauss-Legendre nodes. Needs n >= 4,
    which the entry point already demands.

    The earlier rule -- the logarithmic mean of the two endpoints, exact only
    for a pure exponential -- left a quadrature error that depended on the bin
    width: 0.1-0.5% at nu > 3 on a 129-radius grid, 2% on 65. The training
    target ln(Lambda_MC / Lambda_up) absorbed it, and so did cum_lam, and the
    emulator inherited a dependence on the caller's radius sampling that no
    input could tell it about. This rule leaves 0.01-0.05% at 129 radii, and
    what remains is the finite-difference barrier slope inside f_up itself.
    """
    S = np.asarray(S, dtype=float)
    f_up = np.asarray(f_up, dtype=float)
    n = len(S)
    if n < 4:
        return _baseline_hazard_logmean(S, f_up)
    i = np.arange(n - 1)

    dS = S[:-1] - S[1:]                   # positive: S descends with the index
    j0 = np.clip(i - 1, 0, n - 4)
    idx = j0[:, None] + np.arange(4)      # (n-1, 4) stencil
    xs = S[idx]
    usable = np.all(f_up[idx] > 0.0, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ys = np.log(f_up[idx])
        nodes = (0.5 * (S[:-1] + S[1:]))[:, None] + 0.5 * dS[:, None] * GL_X
        val = np.zeros_like(nodes)
        for a in range(4):
            basis = np.ones_like(nodes)
            for b in range(4):
                if b != a:
                    basis *= ((nodes - xs[:, b, None])
                              / (xs[:, a] - xs[:, b])[:, None])
            val += ys[:, a, None] * basis
        gl = 0.5 * dS * np.sum(GL_W * np.exp(val), axis=1)

    # Where the rate is exactly zero somewhere on the stencil, the trapezoid,
    # which is what a vanishing rate deserves.
    trap = 0.5 * (f_up[:-1] + f_up[1:]) * dS
    lam = np.where(usable, gl, trap)
    return np.maximum(lam, 0.0)


def compute_features(radii, S, barrier, dvar):
    """The nine features on the n-1 bin centres, plus the baseline hazard.

    Returns (X, lam_up, nu_at_largest_radius) where X has shape (n-1, 9) in
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

    # gamma^2 at the same lookback. gamma^2(R) peaks near R ~ 10 Mpc/h, so a
    # given local value is reached from two directions, and a walk descending
    # from hundreds of Mpc/h has a different history from one that started
    # near the peak. Without this the network could not tell them apart: at
    # delta_NL = -0.2 it cut the residual from 3-6 times the Monte Carlo noise
    # to the noise.
    gamma2_back = np.interp(target, S_mid[::-1], gamma2[::-1])

    # Hazard accumulated up to each bin's CENTRE: everything above the bin
    # plus half the bin itself. The walk enters at the largest radius (highest
    # index) and works down, so this is the reverse cumulative sum. It
    # summarises the whole prior history in one number, which is what the
    # surviving population is conditioned on.
    #
    # Measured at the centre because that is where the other features are.
    # The first definition stopped at the bin's upper edge, so it left out half
    # the bin's own hazard -- eight times more of it on a 65-radius grid than on
    # 513 -- and the network, rightly sensitive to depletion (d correction /
    # d cum_lam ~ -1), turned that into a 0.5% dependence on the sampling.
    cum_lam = (np.concatenate([np.cumsum(lam_up[::-1])[::-1][1:], [0.0]])
               + 0.5 * lam_up)

    # nu at the largest RADIUS, where the walk starts -- not at the outermost
    # bin centre, which sits half a grid step inside it and so moved with the
    # caller's sampling (4.31 on a 65-radius grid against 4.48 on 513, for the
    # same walk), carrying the emulated answer 0.4% with it.
    nu_origin = np.full(len(nu), nu_r[-1])

    X = np.column_stack([nu, gamma2, y, dlnnu_dlnS, nu_back, lag,
                         cum_lam, nu_origin, gamma2_back])
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


def _grad2(f, x):
    """d f / d x on a non-uniform grid, second order, numpy's edge_order=2."""
    return np.gradient(np.asarray(f, dtype=float), np.asarray(x, dtype=float),
                       edge_order=2)


def multiplicity_at_radii(radii, S, f_up, corr, lam, nu_at_largest):
    """The first-crossing density per unit R AT the radii, n values.

    multiplicity_from_hazard gives the probability of first crossing inside
    each bin divided by its width -- a bin average, on the n-1 centres, which
    is what a Monte Carlo histogram measures. This is the density itself at
    each radius:

        f(R_i) = r(S_i) * alive(S_i) * |dS/dR|_i

    with r = f_up * exp(correction) the corrected crossing rate per unit S,
    alive the fraction of walks that have not crossed by R_i -- exact at a
    grid point, from the same recursion as the bins -- and dS/dR the
    second-order derivative of S over the radii, taken in ln S and ln R. The correction is fitted per
    bin; it is a smooth O(1) function, so it is carried from the bin centres
    to the radii by linear interpolation in ln S, extrapolated over the half
    bin at either end. The two forms agree to second order in the spacing.
    """
    radii = np.asarray(radii, dtype=float)
    S = np.asarray(S, dtype=float)
    lnS_mid = np.log(0.5 * (S[1:] + S[:-1]))
    lnS = np.log(S)
    corr = np.asarray(corr, dtype=float)

    # S descends with the index, so ln S_mid does too; interpolate on the
    # reversed (ascending) arrays and extrapolate linearly at both ends.
    xs, ys = lnS_mid[::-1], corr[::-1]
    c = np.interp(lnS, xs, ys)
    lo, hi = lnS < xs[0], lnS > xs[-1]
    c[lo] = ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (lnS[lo] - xs[0])
    c[hi] = ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (lnS[hi] - xs[-1])

    n = len(radii)
    alive = np.empty(n)
    alive[n - 1] = 1.0 - float(upper_tail(nu_at_largest))
    for i in range(n - 2, -1, -1):
        alive[i] = alive[i + 1] * np.exp(-lam[i])

    # Through the logarithms for the same reason as dB/dS: S is close to a
    # power law in R. Differenced directly, dS/dR was 0.5% off at 48 radii;
    # this way, 0.007%.
    dS_dR = S / radii * np.abs(_grad2(np.log(S), np.log(radii)))
    return np.asarray(f_up, dtype=float) * np.exp(c) * alive * dS_dR


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
        """log(Lambda_MC / Lambda_up) for each bin.

        nu_origin is held at the trained maximum: above nu ~ 3.5 nothing has
        crossed, so a walk starting further out is the same walk, and the
        network extrapolated past its origins moved the answer by up to 0.4%
        at nu_origin = 8 (under 0.1% held). Training rows are inside the box,
        so this never touches them.
        """
        X = np.array(X, dtype=float)
        if self.box is not None:
            j = FEATURE_NAMES.index("nu_origin")
            X[:, j] = np.minimum(X[:, j], self.box[1][j])
        h = (X - self.mu) / self.sd
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h.ravel()

    # --- the whole thing ----------------------------------------------------

    def multiplicity(self, radii, S, barrier, dvar, check=True, binned=False):
        """The emulated multiplicity function.

        At the n radii by default; with binned=True, the bin averages on the
        n-1 centres that a Monte Carlo histogram measures. Returns (f, info)
        where info carries the domain report.
        """
        X, lam_up, nu_large = compute_features(radii, S, barrier, dvar)
        info = self.check_domain(X, nu_large) if check else {}

        corr = self.correction(X)
        lam = lam_up * np.exp(corr)
        if binned:
            return multiplicity_from_hazard(lam, radii, nu_large), info
        f_up = baseline_rate(S, barrier, dvar)[3]
        return multiplicity_at_radii(radii, S, f_up, corr, lam,
                                     nu_large), info

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
                f"walk origin too low: nu at the largest radius is "
                f"{nu_large:.3f}, {first_step:.2%} of walks start above the "
                f"barrier (expected nu >= {self.nu_origin_min:.3f}; extend "
                f"the radii outward)")

        if self.box is not None:
            lo, hi = self.box
            below = X < lo
            above = X > hi
            # The upper edges of nu_back and nu_origin are where training walks
            # started, not a limit of the physics: nu_origin is held there
            # (correction) and nu_back passes it only at nu > 4, where under
            # 3e-5 of the walks cross. Bins above the trained nu are skipped
            # altogether: nothing crosses there, and the network stays within
            # a few per cent of one out to nu = 50.
            for name in ("nu_back", "nu_origin"):
                above[:, FEATURE_NAMES.index(name)] = False
            # The low end of nu_origin is the origin check above, said once.
            below[:, FEATURE_NAMES.index("nu_origin")] = False
            skip = X[:, FEATURE_NAMES.index("nu")] > hi[0]
            below[skip] = False
            above[skip] = False
            n_out = int(np.any(below | above, axis=1).sum())
            info["outside"] = n_out
            if n_out:
                span = np.where(hi > lo, hi - lo, 1.0)
                excess = np.where(below, lo - X, 0.0) + np.where(above,
                                                                 X - hi, 0.0)
                rel = excess / span
                worst = int(np.argmax(rel.max(axis=0)))
                value = X[int(np.argmax(rel[:, worst])), worst]
                info["ok"] = False
                info["messages"].append(
                    f"{n_out} of {len(X)} bins outside the trained range: "
                    f"{FEATURE_NAMES[worst]} = {value:.4g} (trained "
                    f"{lo[worst]:.4g} to {hi[worst]:.4g})")

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
