#!/usr/bin/env python
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
# sif_barrier_smt because (beta/sigma)^gamma is not real for a negative base.
BARRIER_BOX = dict(
    alpha=(0.05, 2.50),
    beta=(0.01, 1.50),
    gamma=(0.50, 3.50),
)

# Optional smooth perturbation on top of the SMT shape, as insurance against
# the shape-generalization failure seen across spectral families. Kept small on
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
    without moving its overall normalization.
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
