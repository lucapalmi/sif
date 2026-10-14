#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
The training-set design: which cosmologies, which barriers, which radii.

Shared by the local spectrum generator and the cluster worker, so that the
design is defined in exactly one place and the worker can reproduce any curve
from its index alone.

Two decisions are encoded here and both matter.

Radii are chosen PER CURVE so that nu = B/sigma sweeps a fixed window. A fixed
radius range cannot work: over the requested barrier box nu spans twelve orders
of magnitude, so most curves would either never cross or cross instantly, and
the paths spent on them would buy nothing. The window is intersected with a
sigma range, because the nu window alone would reach radii where linear theory
is meaningless and the P(k) table is pure extrapolation.

Barrier parameters are drawn independently for every (cosmology, barrier) pair
rather than from a shared list, so N_COSMO x N_BARRIER distinct points fill the
three-dimensional barrier box instead of N_BARRIER of them.
"""

import numpy as np

# --- cosmology box -----------------------------------------------------------
# Wider than a realistic prior: the emulator's domain should contain the chains
# with room to spare, and cosmology is nearly free here because the feature
# manifold is thin along it.
COSMO_BOX = dict(
    omch2=(0.08, 0.18),
    ombh2=(0.018, 0.028),
    H0=(55.0, 80.0),
    ns=(0.90, 1.02),
    mnu=(0.0, 0.5),
    z=(0.0, 2.0),
)

# --- barrier box -------------------------------------------------------------
# B(sigma) = alpha [1 + (beta / sigma)^gamma].
# Endpoints are floored strictly positive: alpha = 0 puts the barrier at zero,
# which every walk crosses on its first step, and beta = 0 is rejected by
# sif_ep_barrier_smt because (beta/sigma)^gamma is not real for a negative base.
BARRIER_BOX = dict(
    alpha=(0.05, 2.50),
    beta=(0.01, 1.50),
    gamma=(0.50, 3.50),
)

# Optional smooth perturbation on top of the SMT shape, as insurance against
# the shape-generalisation failure seen across spectral families. Kept small on
# purpose: the emulator's contract is the SMT family, and this only widens the
# neighbourhood around it rather than claiming arbitrary barriers.
PERTURB_FRACTION = 0.25   # share of curves carrying one
PERTURB_AMPLITUDE = 0.05  # relative, peak

N_COSMO = 64
N_BARRIER = 24
N_RADII = 128
N_PATHS = 1_000_000_000

# A curve is kept only if its nu window, AFTER the radius clip below, reaches
# down to this. It is a compute argument, not a physics one: a walk that never
# crosses still steps through every radius, so a curve crossing 0.1% of its
# paths is the most expensive kind to run and the least informative to have.
#
# The (cosmology, barrier) pairs this rejects are those whose barrier is out of
# reach at every physical scale -- no voids form, and any likelihood is zero
# there. What the emulator does in that corner is the nu >> 1 asymptote, which
# is exactly where the up-crossing baseline is already exact and needs no
# correction, and it is covered anyway by the high-nu tail of every kept curve.
NU_REACH = 1.2  # Phi_bar(1.2) = 11.5%, the worst crossing fraction accepted
MAX_DRAWS = 200

# nu at the walk's origin (largest radius). 4.5 leaves a first-step point mass
# of 3.4e-6, far below the smallest bin the training will use.
NU_LO, NU_HI = 0.25, 4.5

# Where linear theory is worth modelling, and where the P(k) table is real
# rather than padded. Intersected with the nu window above.
SIGMA_LO, SIGMA_HI = 0.15, 4.0

# And a hard bound on the radius itself. The sigma window alone does not bite
# for a high-redshift or low-amplitude cosmology -- sigma never reaches
# SIGMA_HI, the clip does nothing, and the nu window then reaches radii of
# 10^-3 Mpc/h where the P(k) table is pure power-law padding and gamma^2 drifts
# far below anything a real analysis will query.
R_MIN, R_MAX = 0.5, 150.0

SEED = 20260809


def latin_hypercube(n, box, rng):
    keys = list(box)
    cube = (rng.permuted(np.tile(np.arange(n), (len(keys), 1)), axis=1).T
            + rng.random((n, len(keys)))) / n
    return [{k: box[k][0] + cube[i, j] * (box[k][1] - box[k][0])
             for j, k in enumerate(keys)} for i in range(n)]


def build_cosmologies(n_cosmo=N_COSMO, seed=SEED):
    """The cosmology hypercube. Deterministic in `seed`."""
    return latin_hypercube(n_cosmo, COSMO_BOX, np.random.default_rng(seed))


def draw_barriers(sigma_scan, radii_scan, cosmo_index, n_barrier=N_BARRIER,
                  seed=SEED):
    """n_barrier barriers for one cosmology, each reachable at physical radii.

    Rejection sampling over the barrier box: draws are proposed from the
    requested ranges and kept only when the resulting curve actually crosses
    (see NU_REACH). The acceptance depends on the cosmology through sigma, so
    this needs the spectrum -- the design is reproducible from `seed` PLUS the
    spectrum table, not from the seed alone.

    Returns (accepted, n_proposed) so the caller can report how much of the
    box a cosmology could reach.
    """
    rng = np.random.default_rng(seed + 1000 + cosmo_index)
    keys = list(BARRIER_BOX)

    accepted = []
    proposed = 0

    while len(accepted) < n_barrier and proposed < MAX_DRAWS * n_barrier:
        p = {k: rng.uniform(*BARRIER_BOX[k]) for k in keys}
        proposed += 1

        j = len(accepted)
        p["perturb_seed"] = (
            seed + 50000 + cosmo_index * n_barrier + j
            if j % int(round(1 / PERTURB_FRACTION)) == 0 else 0)

        if reaches(sigma_scan, radii_scan, p):
            accepted.append(p)

    return accepted, proposed


def reaches(sigma_scan, radii_scan, params):
    """Does this barrier come within NU_REACH of the field at physical radii?"""
    radii = radii_for_curve(sigma_scan, radii_scan, params)
    if radii is None:
        return False

    sig = np.asarray(sigma_scan, dtype=float)
    lnR = np.log(np.asarray(radii_scan, dtype=float))
    nu = barrier_values(sig, params) / sig

    # nu at the smallest radius the clipped window actually reached
    order = np.argsort(lnR)
    nu_at_lo = np.interp(np.log(float(radii[0])), lnR[order], nu[order])
    return nu_at_lo <= NU_REACH


def barrier_values(sigma, params):
    """SMT barrier, optionally with a smooth relative perturbation.

    The perturbation is a low-order Chebyshev-like ripple in ln sigma: smooth,
    bounded, and zero-mean in the log, so it changes the barrier's shape
    without moving its overall normalisation.
    """
    a, b, g = params["alpha"], params["beta"], params["gamma"]
    B = a * (1.0 + (b / sigma) ** g)

    if params.get("perturb_seed", 0):
        rng = np.random.default_rng(int(params["perturb_seed"]))
        x = np.log(sigma)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        amp = rng.uniform(-1.0, 1.0, 3)
        ripple = (amp[0] * x + amp[1] * (2 * x ** 2 - 1)
                  + amp[2] * (4 * x ** 3 - 3 * x))
        ripple /= np.abs(ripple).max() + 1e-12
        B = B * (1.0 + PERTURB_AMPLITUDE * ripple)

    return B


def radii_for_curve(sigma_scan, radii_scan, params, n_radii=N_RADII):
    """Radii sweeping the nu window, clipped to the sigma window.

    Returns None when the two windows do not intersect over the scan, which
    the caller should treat as a dropped curve rather than an error.
    """
    sig = np.asarray(sigma_scan, dtype=float)
    lnR = np.log(np.asarray(radii_scan, dtype=float))

    B = barrier_values(sig, params)
    nu = B / sig

    # nu rises with R for any positive barrier of this family; enforce
    # monotonicity by sorting rather than assuming it.
    order = np.argsort(nu)
    nu_s, lnR_s = nu[order], lnR[order]

    ln_r_nu_lo = np.interp(NU_LO, nu_s, lnR_s)
    ln_r_nu_hi = np.interp(NU_HI, nu_s, lnR_s)

    # sigma falls with R, so the sigma window maps to an R window too.
    order_s = np.argsort(-sig)
    ln_r_sig_lo = np.interp(SIGMA_HI, sig[order_s][::-1], lnR[order_s][::-1])
    ln_r_sig_hi = np.interp(SIGMA_LO, sig[order_s][::-1], lnR[order_s][::-1])

    lo = max(ln_r_nu_lo, ln_r_sig_lo, np.log(R_MIN))
    hi = min(ln_r_nu_hi, ln_r_sig_hi, np.log(R_MAX))

    if not (hi > lo + 0.2):  # need at least a modest span in ln R
        return None

    return np.logspace(lo / np.log(10), hi / np.log(10),
                       n_radii).astype(np.float32)


# =============================================================================
# Campaign 2: walks placed by where they start
# =============================================================================
#
# Campaign 1 (everything above) chose radii from the nu window and so placed
# walks wherever the barrier draw happened to put them. Measured afterwards,
# none of its 1536 curves started beyond R = 51 Mpc/h: SIGMA_LO cut every walk
# off where sigma reaches 0.15 (R ~ 52 at z = 0), and reaching larger radii
# needed a small alpha AND a small beta at once, a corner of the box nobody
# drew. gamma^2 below 0.2 was therefore only ever seen on the small-R side of
# its peak, and the emulator was 1-6% wrong for walks starting at 150-700 --
# while the per-feature box still reported most of them in domain.
#
# Campaign 2 draws the walk's ORIGIN instead: a radius log-uniform over
# [R_ORIGIN_LO, R_ORIGIN_HI] and a nu there uniform over [NU_ORIGIN_LO,
# NU_ORIGIN_HI], then solves alpha so the barrier sits at that nu. The emulator
# never sees alpha, only B/sigma and its shape, so this loses nothing and
# covers the walk's history evenly in the quantities it does see.
#
# The origin nu is random rather than fixed at NU_HI because the answer must
# not depend on where the grid starts: walks starting above nu ~ 3.5 have
# essentially never crossed (first-step mass < 2.3e-4), so the Monte Carlo is
# origin-independent to that level. With every campaign-1 curve starting at
# nu = 4.5, the network could not separate the lookback clamp at the origin from
# the physics, and the emulated f moved 0.2-0.5% between a grid started at 4.5
# and one started at 4.8.
#
# Cosmologies are campaign 1's (same seed), so cross-validation can hold a
# cosmology out of both campaigns at once. The barrier stream has its own seed.

R_ORIGIN_LO, R_ORIGIN_HI = 2.0, 700.0
NU_ORIGIN_LO, NU_ORIGIN_HI = 3.5, 5.5
N_BARRIER_2 = 24
SEED_2 = SEED + 2_000_000

# The inner edge: nu down to NU_LO, but never below R_MIN or past SIGMA_HI.
# There is no SIGMA_LO and no R_MAX here; the origin is the outer edge.


def barrier_shape(sigma, params):
    """B / alpha for these (beta, gamma), ripple included."""
    return barrier_values(sigma, dict(params, alpha=1.0))


def radii_from_origin(sigma_scan, radii_scan, params, r_origin, nu_origin,
                      n_radii=N_RADII):
    """Radii for a walk starting at r_origin with nu = nu_origin there.

    Returns (radii, alpha), or None when nu never falls to NU_REACH inside the
    allowed range -- a barrier too steep to be crossed before R_MIN.
    """
    sig = np.asarray(sigma_scan, dtype=float)
    lnR = np.log(np.asarray(radii_scan, dtype=float))

    # alpha from the SMOOTH shape at the origin. The ripple is normalised to
    # the ln sigma span of whatever array barrier_values is handed -- here the
    # scan, in the worker the curve's own radii -- so it cannot be matched
    # exactly in advance; a rippled curve lands within PERTURB_AMPLITUDE of the
    # drawn nu_origin, which the random origin window absorbs.
    shape = barrier_shape(sig, dict(params, perturb_seed=0))
    s_o = np.exp(np.interp(np.log(r_origin), lnR, np.log(sig)))
    shape_o = np.exp(np.interp(np.log(r_origin), lnR, np.log(shape)))
    alpha = nu_origin * s_o / shape_o

    nu = alpha * shape / sig
    inside = lnR <= np.log(r_origin)
    # Reachable only over the radii the walk will actually visit: inside the
    # origin, above R_MIN, and below SIGMA_HI.
    visited = inside & (lnR >= np.log(R_MIN)) & (sig < SIGMA_HI)
    if not np.any(visited) or nu[visited].min() > NU_REACH:
        return None

    lo = np.log(R_MIN)
    below = np.where(inside & (nu <= NU_LO))[0]
    if len(below):
        lo = max(lo, lnR[below.max()])
    too_high = np.where(inside & (sig >= SIGMA_HI))[0]
    if len(too_high):
        lo = max(lo, lnR[too_high.max()])

    hi = np.log(r_origin)
    if not (hi > lo + 0.2):
        return None
    radii = np.exp(np.linspace(lo, hi, n_radii)).astype(np.float32)
    return radii, float(alpha)


def draw_barriers_2(sigma_scan, radii_scan, cosmo_index,
                    n_barrier=N_BARRIER_2, seed=SEED_2,
                    origin_range=(R_ORIGIN_LO, R_ORIGIN_HI), r_reach=None,
                    beta_log=None):
    """Campaign-2 barriers for one cosmology: origin, origin nu, beta, gamma.

    With r_reach, a draw is kept only if its walk comes down below that
    radius; with beta_log = (lo, hi), beta is log-uniform over it instead of
    uniform over the box (campaign 3). The defaults reproduce campaign 2 draw
    for draw.

    Returns a list of dicts carrying alpha (solved), beta, gamma,
    perturb_seed, r_origin and nu_origin.
    """
    rng = np.random.default_rng(seed + cosmo_index)
    out = []
    for j in range(n_barrier):
        r_o = float(np.exp(rng.uniform(np.log(origin_range[0]),
                                       np.log(origin_range[1]))))
        perturb = (seed + 50000 + cosmo_index * n_barrier + j
                   if j % int(round(1 / PERTURB_FRACTION)) == 0 else 0)
        for _ in range(MAX_DRAWS):
            beta = (rng.uniform(*BARRIER_BOX["beta"]) if beta_log is None
                    else float(np.exp(rng.uniform(np.log(beta_log[0]),
                                                  np.log(beta_log[1])))))
            p = dict(beta=beta, gamma=rng.uniform(*BARRIER_BOX["gamma"]),
                     perturb_seed=perturb)
            nu_o = rng.uniform(NU_ORIGIN_LO, NU_ORIGIN_HI)
            res = radii_from_origin(sigma_scan, radii_scan, p, r_o, nu_o)
            if res is not None and (r_reach is None or res[0][0] < r_reach):
                p.update(alpha=res[1], r_origin=r_o, nu_origin=nu_o)
                out.append(p)
                break
    return out


# =============================================================================
# Campaign 3: long walks from far out
# =============================================================================
#
# Campaign 2 drew its origins log-uniform over 2-700 Mpc/h, but a walk that
# starts beyond 150 Mpc/h only comes down below 80 if its barrier is nearly
# flat in S: nu has to fall by a factor ~5 while R falls by less than one, and
# most barriers in the box are steep enough that their walks end well above
# that. So the bins below 80 Mpc/h of walks starting beyond 150 were 0.4% of
# the training set, and cross-validation put them at 0.45% against 0.08%
# elsewhere. They are what a shallow void threshold at low redshift produces.
#
# Campaign 3 fills that corner directly: origins over [R_ORIGIN_LO_3,
# R_ORIGIN_HI_3], kept only when the walk reaches below R_REACH_3. The range
# brackets 200 Mpc/h, the largest origin a realistic void analysis needs (a
# shallow threshold at low redshift), with margin on both sides so it is not
# at the edge of what was trained.
#
# beta is log-uniform down to 1e-3 here rather than uniform from 0.01, which
# puts more of the draws on flat barriers -- the only ones whose walks span
# this range -- and reaches the constant barrier, a standard choice in its own
# right (the Sheth-van de Weygaert delta_v) that sat at the edge of the box.

R_ORIGIN_LO_3, R_ORIGIN_HI_3 = 120.0, 250.0
R_REACH_3 = 60.0
BETA_LOG_3 = (1e-3, 1.5)
N_BARRIER_3 = 12
SEED_3 = SEED + 3_000_000


def draw_barriers_3(sigma_scan, radii_scan, cosmo_index,
                    n_barrier=N_BARRIER_3, seed=SEED_3):
    """Campaign-3 barriers: far origins whose walks still reach R_REACH_3."""
    return draw_barriers_2(sigma_scan, radii_scan, cosmo_index, n_barrier,
                           seed, origin_range=(R_ORIGIN_LO_3, R_ORIGIN_HI_3),
                           r_reach=R_REACH_3, beta_log=BETA_LOG_3)


# =============================================================================
# Campaign 4: the middle of the range
# =============================================================================
#
# With the contract at walks starting 1-250 Mpc/h, campaigns 1 and 2 left the
# training density uneven: 700-1000 curves per e-fold of origin at 3-30 Mpc/h,
# ~300 at 30-120, and campaign 3 brings 120-250 to ~1000. Campaign 4 evens out
# 30-120, with campaign 3's log-uniform beta so flat and constant barriers are
# as well represented there. (Below ~2 Mpc/h nothing can be added: a walk
# starting at 1 Mpc/h ends at R_MIN = 0.5 before nu can fall to crossing
# values, i.e. it would describe voids smaller than 0.5 Mpc/h.)

R_ORIGIN_LO_4, R_ORIGIN_HI_4 = 30.0, 120.0
N_BARRIER_4 = 8
SEED_4 = SEED + 4_000_000


def draw_barriers_4(sigma_scan, radii_scan, cosmo_index,
                    n_barrier=N_BARRIER_4, seed=SEED_4):
    """Campaign-4 barriers: origins over the middle of the range."""
    return draw_barriers_2(sigma_scan, radii_scan, cosmo_index, n_barrier,
                           seed, origin_range=(R_ORIGIN_LO_4, R_ORIGIN_HI_4),
                           beta_log=BETA_LOG_3)
