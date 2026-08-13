#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
Pilot study for the EP multiplicity emulator.

Question under test: is the ratio between the Monte Carlo first-crossing
hazard and the Musso-Sheth up-crossing hazard a smooth, O(1) target that
local features (nu, gamma^2, y) organize across spectra and barriers?

Runs ~27 first-crossing ensembles at modest n_paths (about 2 minutes on a
laptop), extracts the per-bin hazard ratio with its Monte Carlo error, and
plots:

  1. ratio curves per barrier family across spectra (smoothness, spread)
  2. the same points in feature space (do cosmologies collapse?)
  3. noise validation (two seeds) and grid dependence (51/101/201 radii)

Usage:  python tools/ep_ratio_pilot.py [output_dir]
"""

import os
import sys
import time

import numpy as np

import pysif.model as model

# The baseline, the hazard extraction and the feature definitions all live in
# ep_model, which is what the deployed emulator calls. This script predates
# that module and originally carried its own copies; they are imported now so
# there is exactly one definition of each in the tree.
from ep_model import baseline_hazard, mc_hazard, upper_tail as _upper_tail
from ep_model import baseline_rate as baseline_features


def cov_diag(cov_packed, n):
    idx = np.arange(n)
    return cov_packed[idx * (idx + 1) // 2 + idx]


# ----------------------------------------------------------------------
# Spectra
# ----------------------------------------------------------------------


def power_law_pk(ns):
    k = np.logspace(-5, 4, 4000)
    return k.astype(np.float32), (k ** ns).astype(np.float32)


def eh_zero_baryon_pk(shape_gamma, ns=0.96):
    """Eisenstein & Hu (1998) zero-baryon transfer, k in h/Mpc."""
    k = np.logspace(-4, 3, 3000)
    q = k / shape_gamma
    L = np.log(2.0 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T = L / (L + C * q * q)
    return k.astype(np.float32), (k ** ns * T * T).astype(np.float32)


def normalize_sigma8(k, pk, sigma8=0.8):
    _, s8, _, _ = model.delta_covariance_pk(k, pk, np.array([8.0], np.float32))
    return (pk * (sigma8 / s8[0]) ** 2).astype(np.float32)


def radii_for_sigma_range(k, pk, s_hi=2.5, s_lo=0.30, n_radii=101):
    """Log-spaced radii spanning sigma in [s_lo, s_hi] for this spectrum."""
    scan = np.logspace(-2, 3.2, 240).astype(np.float32)
    _, sig, _, _ = model.delta_covariance_pk(k, pk, scan)
    good = sig > 0
    ln_r = np.log(scan[good].astype(float))
    ln_s = np.log(sig[good].astype(float))
    # sigma decreases with R; interpolate ln R against ln sigma
    r_of_s = lambda s: np.exp(np.interp(np.log(s), ln_s[::-1], ln_r[::-1]))
    r_lo, r_hi = r_of_s(s_hi), r_of_s(s_lo)
    return np.logspace(np.log10(r_lo), np.log10(r_hi), n_radii).astype(
        np.float32)


# ----------------------------------------------------------------------
# The pilot grid
# ----------------------------------------------------------------------

SPECTRA = [
    ("pl-2.5", lambda: power_law_pk(-2.5)),
    ("pl-2.0", lambda: power_law_pk(-2.0)),
    ("pl-1.5", lambda: power_law_pk(-1.5)),
    ("pl-1.0", lambda: power_law_pk(-1.0)),
    ("eh0.14", lambda: eh_zero_baryon_pk(0.14)),
    ("eh0.20", lambda: eh_zero_baryon_pk(0.20)),
]

BARRIERS = [
    ("flat", dict(alpha=0.70, beta=0.30, gamma=0.0)),
    ("smt_mild", dict(alpha=0.50, beta=0.30, gamma=0.6)),
    ("smt_steep", dict(alpha=0.40, beta=0.50, gamma=1.0)),
    ("smt_heavy", dict(alpha=0.70, beta=0.20, gamma=0.3)),
]

N_PATHS = 10_000_000
SEED = 20260809


def run_case(radii, cov, sigma, dvar, bpars, n_paths=N_PATHS, seed=SEED):
    """One ensemble: returns a dict with features, hazards, ratio, errors."""
    n = len(radii)
    barrier = model.barrier_smt(sigma, **bpars)
    S = cov_diag(cov, n).astype(float)
    nu, gamma2, y, f_up = baseline_features(S, barrier.astype(float), dvar)
    lam_up = baseline_hazard(S, f_up)

    t0 = time.perf_counter()
    counts = model.first_crossing_counts_ep(
        radii, cov, barrier, n_paths=n_paths, seed=seed)
    dt = time.perf_counter() - t0

    lam_mc, rel_err, alive = mc_hazard(counts.astype(float), n_paths)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lam_up > 0.0, lam_mc / lam_up, np.nan)

    # bin-centre features (radii ascending; bin i between i and i+1)
    mid = lambda a: 0.5 * (a[:-1] + a[1:])
    return dict(
        radii=radii.astype(float), S=S, counts=counts,
        nu=mid(nu), gamma2=mid(gamma2), y=mid(y),
        lam_up=lam_up, lam_mc=lam_mc, ratio=ratio, rel_err=rel_err,
        alive=alive, crossed=counts.sum() / n_paths,
        dropped=counts[-1] / n_paths, walk_seconds=dt,
    )


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    spectra_data = {}

    for sname, build in SPECTRA:
        k, pk = build()
        pk = normalize_sigma8(k, pk)
        radii = radii_for_sigma_range(k, pk)
        cov, sigma, high, dvar = model.delta_covariance_pk(k, pk, radii)
        spectra_data[sname] = (k, pk)
        print(f"[{sname}] R in [{radii[0]:.3g}, {radii[-1]:.3g}], "
              f"sigma in [{sigma[-1]:.3f}, {sigma[0]:.3f}], "
              f"max high-k fraction {high.max():.3f}")

        for bname, bpars in BARRIERS:
            case = run_case(radii, cov, sigma, dvar, bpars)
            results[(sname, bname)] = case
            good = case["rel_err"] < 0.2
            r = case["ratio"][good]
            print(f"    {bname:10s} crossed {case['crossed']:.3f} "
                  f"dropped {case['dropped']:.2e} "
                  f"ratio [{np.nanmin(r):.3f}, {np.nanmax(r):.3f}] "
                  f"median|r-1| {np.nanmedian(np.abs(r - 1)):.3f} "
                  f"({case['walk_seconds']:.1f}s)")

    # --- repeat one case at a different seed, for the noise check ---
    k, pk = spectra_data["eh0.14"]
    radii = radii_for_sigma_range(k, pk)
    cov, sigma, high, dvar = model.delta_covariance_pk(k, pk, radii)
    noise_a = results[("eh0.14", "flat")]
    noise_b = run_case(radii, cov, sigma, dvar, dict(BARRIERS[0][1]),
                       seed=SEED + 1)

    # --- one case at three grid resolutions ---
    grid_cases = {}
    for n_r in (51, 101, 201):
        radii_g = radii_for_sigma_range(k, pk, n_radii=n_r)
        try:
            cov_g, sigma_g, _, dvar_g = model.delta_covariance_pk(
                k, pk, radii_g)
            grid_cases[n_r] = run_case(
                radii_g, cov_g, sigma_g, dvar_g, dict(BARRIERS[1][1]))
        except RuntimeError as e:
            print(f"grid n={n_r} failed: {e}")

    np.savez(os.path.join(out_dir, "ep_pilot_results.npz"),
             **{f"{s}__{b}__{key}": val
                for (s, b), case in results.items()
                for key, val in case.items() if key != "walk_seconds"})

    make_plots(results, noise_a, noise_b, grid_cases, out_dir)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------


def make_plots(results, noise_a, noise_b, grid_cases, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec_names = [s for s, _ in SPECTRA]
    cmap = plt.get_cmap("viridis")
    spec_color = {s: cmap(i / (len(spec_names) - 1))
                  for i, s in enumerate(spec_names)}

    def good_mask(case, max_rel=0.2):
        return (case["rel_err"] < max_rel) & np.isfinite(case["ratio"])

    # --- 1. ratio curves per barrier family ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (bname, _) in zip(axes.ravel(), BARRIERS):
        for sname in spec_names:
            case = results[(sname, bname)]
            g = good_mask(case)
            ax.plot(case["nu"][g], case["ratio"][g], "-", lw=1.2,
                    color=spec_color[sname], label=sname)
            ax.fill_between(
                case["nu"][g],
                case["ratio"][g] * (1 - case["rel_err"][g]),
                case["ratio"][g] * (1 + case["rel_err"][g]),
                color=spec_color[sname], alpha=0.15, lw=0)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_title(bname)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"$\nu = B/\sigma$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Lambda_{\rm MC}/\Lambda_{\rm up}$")
    fig.suptitle("Hazard ratio vs nu: 6 spectra x 4 barriers "
                 f"({N_PATHS:.0e} paths each; bins with <5% error)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pilot_1_ratio_curves.png"), dpi=150)

    # --- 2. feature-space collapse ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    pooled = {k: [] for k in ("nu", "gamma2", "y", "ratio")}
    for case in results.values():
        g = good_mask(case, max_rel=0.05)
        for key in pooled:
            pooled[key].append(case[key][g])
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}

    for ax, (xkey, ckey) in zip(
            axes, [("nu", "gamma2"), ("y", "gamma2"), ("gamma2", "nu")]):
        sc = ax.scatter(pooled[xkey], pooled["ratio"], c=pooled[ckey],
                        s=4, cmap="plasma", alpha=0.6)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xlabel(xkey)
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label=ckey)
    axes[0].set_ylabel(r"$\Lambda_{\rm MC}/\Lambda_{\rm up}$")
    fig.suptitle("All bins with <5% MC error, pooled over every run")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pilot_2_feature_space.png"), dpi=150)

    # --- 3. noise and grid dependence ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ga = good_mask(noise_a) & good_mask(noise_b)
    pull = (noise_a["ratio"][ga] - noise_b["ratio"][ga]) / np.sqrt(
        (noise_a["ratio"][ga] * noise_a["rel_err"][ga]) ** 2
        + (noise_b["ratio"][ga] * noise_b["rel_err"][ga]) ** 2)
    axes[0].hist(pull, bins=30, density=True, alpha=0.7)
    xs = np.linspace(-4, 4, 200)
    axes[0].plot(xs, np.exp(-xs * xs / 2) / np.sqrt(2 * np.pi), "k-", lw=1)
    axes[0].set_title(f"seed-to-seed pull (std {pull.std():.2f})")
    axes[0].set_xlabel("(ratio$_a$ - ratio$_b$) / expected error")

    for n_r, case in grid_cases.items():
        g = good_mask(case)
        axes[1].plot(case["nu"][g], case["ratio"][g], "-", lw=1.2,
                     label=f"{n_r} radii")
    axes[1].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[1].legend()
    axes[1].set_title("grid dependence (eh0.14, smt_mild)")
    axes[1].set_xlabel(r"$\nu$")
    axes[1].set_ylabel(r"$\Lambda_{\rm MC}/\Lambda_{\rm up}$")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pilot_3_noise_grid.png"), dpi=150)

    print(f"\nplots and ep_pilot_results.npz written to {out_dir}")


if __name__ == "__main__":
    main()
