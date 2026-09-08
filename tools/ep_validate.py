#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
End-to-end validation of the trained emulator, against fresh Monte Carlo.

Everything up to here was cross-validation inside one training set. This runs
NEW cosmologies and NEW barriers, drawn from a different seed, through a fresh
first-crossing ensemble, and compares the emulator with the Monte Carlo the way
a user would: same inputs in, multiplicity out.

Four things are tested, and the last two need no Monte Carlo at all:

  1. accuracy against a fresh ensemble
  2. grid independence -- the same physical problem sampled at 64, 128 and 256
     radii must give the same answer. Nothing in the fit enforced this, so it
     is a real check on whether the features are properties of the field or of
     the grid
  3. the domain guard, deliberately provoked
  4. speed

The `plot` stage draws the first of those across the barrier box, which is the
part a pooled number cannot show.

Usage:
  python tools/ep_validate.py --stage mc      # runs the fresh ensemble
  python tools/ep_validate.py --stage analyse # everything else
  python tools/ep_validate.py --stage plot --out-dir <dir>
"""

import argparse
import os
import time

import numpy as np

import pysif.model as model
from ep_model import EPEmulator, FEATURE_NAMES, compute_features, upper_tail
from ep_trainset_design import (BARRIER_BOX, COSMO_BOX, MAX_DRAWS, NU_REACH,
                                barrier_values, latin_hypercube,
                                radii_for_curve)
from ep_trainset_worker import SCAN

# The data lives next to this script, and the script is run from wherever the
# caller happens to be standing. Bare filenames as defaults would resolve
# against the shell's cwd and fail from the repository root, which is the one
# place anybody actually runs it from.
HERE = os.path.dirname(os.path.abspath(__file__))

VALID_SEED = 777_000        # deliberately unrelated to the training seed
N_COSMO = 8
N_BARRIER = 3
N_PATHS = 100_000_000
N_RADII = 128


def camb_spectrum(p):
    import camb
    from ep_make_spectra import K_MAX, K_MIN, N_K, pad_grid, pad_pk
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"],
                       mnu=p["mnu"], omk=0.0)
    pars.InitPower.set_params(ns=p["ns"], As=2.1e-9)
    pars.set_matter_power(redshifts=[p["z"]], kmax=K_MAX * 1.2)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    kh, _, pk = res.get_matter_power_spectrum(minkh=K_MIN, maxkh=K_MAX,
                                              npoints=N_K)
    k_full, n_lo, n_hi = pad_grid(kh)
    return (k_full.astype(np.float32),
            pad_pk(kh, pk[0], k_full, n_lo, n_hi).astype(np.float32))


def draw_reachable(sigma_scan, rng, n):
    out = []
    tries = 0
    while len(out) < n and tries < MAX_DRAWS * n:
        p = {k: rng.uniform(*BARRIER_BOX[k]) for k in BARRIER_BOX}
        p["perturb_seed"] = 0
        tries += 1
        radii = radii_for_curve(sigma_scan, SCAN, p)
        if radii is None:
            continue
        sig = np.asarray(sigma_scan, dtype=float)
        lnR = np.log(np.asarray(SCAN, dtype=float))
        nu = barrier_values(sig, p) / sig
        o = np.argsort(lnR)
        if np.interp(np.log(float(radii[0])), lnR[o], nu[o]) <= NU_REACH:
            out.append(p)
    return out


def stage_mc(out_path):
    rng = np.random.default_rng(VALID_SEED)
    cosmologies = latin_hypercube(N_COSMO, COSMO_BOX, rng)

    store = {}
    t0 = time.perf_counter()

    for i, p in enumerate(cosmologies):
        k, pk = camb_spectrum(p)
        _, sigma_scan, _, _ = model.delta_covariance_pk(k, pk, SCAN)
        barriers = draw_reachable(sigma_scan,
                                  np.random.default_rng(VALID_SEED + 17 + i),
                                  N_BARRIER)
        print(f"[{i}] omch2={p['omch2']:.4f} h={p['H0'] / 100:.3f} "
              f"ns={p['ns']:.3f} mnu={p['mnu']:.3f} z={p['z']:.3f}", flush=True)

        for j, bp in enumerate(barriers):
            radii = radii_for_curve(sigma_scan, SCAN, bp, n_radii=N_RADII)
            cov, sigma, _, dvar = model.delta_covariance_pk(k, pk, radii)
            barrier = barrier_values(sigma.astype(float), bp).astype(np.float32)

            t = time.perf_counter()
            counts = model.first_crossing_counts_ep(
                radii, cov, barrier, n_paths=N_PATHS,
                seed=VALID_SEED + 1000 * i + j)
            dt = time.perf_counter() - t

            n = len(radii)
            idx = np.arange(n)
            tag = f"v{i:02d}_{j}"
            store[f"{tag}__radii"] = radii
            store[f"{tag}__S"] = cov[idx * (idx + 1) // 2 + idx]
            store[f"{tag}__barrier"] = barrier
            store[f"{tag}__dvar"] = dvar
            store[f"{tag}__counts"] = counts
            for key in ("alpha", "beta", "gamma"):
                store[f"{tag}__{key}"] = np.float64(bp[key])
            # keep the spectrum so the grid test can resample it
            store[f"c{i:02d}__k"] = k
            store[f"c{i:02d}__pk"] = pk
            print(f"    a={bp['alpha']:.3f} b={bp['beta']:.3f} "
                  f"g={bp['gamma']:.3f} crossed {counts.sum() / N_PATHS:.3f} "
                  f"({dt:.0f}s)", flush=True)

    store["n_paths"] = np.uint64(N_PATHS)
    np.savez_compressed(out_path, **store)
    print(f"\n{N_COSMO * N_BARRIER} curves in "
          f"{(time.perf_counter() - t0) / 60:.1f} min -> {out_path}")


def stage_analyse(mc_path, weights):
    emu = EPEmulator.load(weights)
    d = np.load(mc_path)
    n_paths = int(d["n_paths"])
    tags = sorted({k.split("__")[0] for k in d.files if k.startswith("v")})

    print(f"{'=' * 84}\n1. accuracy against a FRESH ensemble "
          f"({len(tags)} curves, {n_paths:.0e} paths)\n{'=' * 84}")
    print(f"{'curve':>8}{'alpha':>8}{'beta':>7}{'gamma':>7}"
          f"{'median':>10}{'95th':>9}{'nu<2':>9}{'nu>2':>9}{'outside':>9}")
    print("-" * 84)

    all_rel, all_nu, all_noise = [], [], []
    for tag in tags:
        radii = d[f"{tag}__radii"].astype(float)
        S = d[f"{tag}__S"]
        barrier = d[f"{tag}__barrier"].astype(float)
        dvar = d[f"{tag}__dvar"]
        counts = d[f"{tag}__counts"].astype(float)

        f_emu, info = emu.multiplicity(radii, S, barrier, dvar)
        dr = np.diff(radii)
        f_mc = counts[:-1] / (n_paths * dr)

        # Poisson error on the Monte Carlo's own bins, for context
        with np.errstate(divide="ignore", invalid="ignore"):
            mc_err = np.where(counts[:-1] > 0,
                              1.0 / np.sqrt(np.maximum(counts[:-1], 1)), np.inf)
            rel = np.where(f_mc > 0, (f_emu - f_mc) / f_mc, np.nan)

        X, _, _ = compute_features(radii, S, barrier, dvar)
        nu = X[:, 0]
        good = np.isfinite(rel) & (mc_err < 0.05)

        a = np.abs(rel[good])
        lo = np.median(a[nu[good] < 2]) if (nu[good] < 2).any() else np.nan
        hi = np.median(a[nu[good] >= 2]) if (nu[good] >= 2).any() else np.nan
        print(f"{tag:>8}{float(d[f'{tag}__alpha']):>8.3f}"
              f"{float(d[f'{tag}__beta']):>7.3f}{float(d[f'{tag}__gamma']):>7.3f}"
              f"{np.median(a):>10.3%}{np.percentile(a, 95):>9.2%}"
              f"{lo:>9.2%}{hi:>9.2%}{info['outside']:>9}")

        all_rel.append(a)
        all_nu.append(nu[good])
        all_noise.append(mc_err[good])

    a = np.concatenate(all_rel)
    nu = np.concatenate(all_nu)
    ns = np.concatenate(all_noise)
    print("-" * 84)
    print(f"{'POOLED':>8}{'':>22}{np.median(a):>10.3%}"
          f"{np.percentile(a, 95):>9.2%}"
          f"{np.median(a[nu < 2]):>9.2%}{np.median(a[nu >= 2]):>9.2%}")
    print(f"\n  the Monte Carlo's own error on these bins: median "
          f"{np.median(ns):.2%} -- at {n_paths:.0e} paths it is a "
          f"co-contributor,\n  not a clean yardstick, so the emulator's true "
          f"error is below the number above.")


def stage_grid(mc_path, weights):
    """Same physics, three radius grids. Nothing in the fit enforced this."""
    emu = EPEmulator.load(weights)
    d = np.load(mc_path)
    tags = sorted({k.split("__")[0] for k in d.files if k.startswith("v")})

    print(f"\n{'=' * 84}\n2. grid independence -- the same problem at 64, 128 "
          f"and 256 radii\n{'=' * 84}")
    print(f"{'curve':>8}{'read at R':>12}{'f(64)':>13}{'f(128)':>13}"
          f"{'f(256)':>13}{'spread':>10}")
    print("-" * 84)

    worst = 0.0
    for tag in tags[:6]:
        ci = "c" + tag[1:3]
        k, pk = d[f"{ci}__k"], d[f"{ci}__pk"]
        a, b, g = (float(d[f"{tag}__alpha"]), float(d[f"{tag}__beta"]),
                   float(d[f"{tag}__gamma"]))
        bp = dict(alpha=a, beta=b, gamma=g, perturb_seed=0)

        _, sigma_scan, _, _ = model.delta_covariance_pk(k, pk, SCAN)

        curves = {}
        for n_r in (64, 128, 256):
            radii = radii_for_curve(sigma_scan, SCAN, bp, n_radii=n_r)
            cov, sigma, _, dvar = model.delta_covariance_pk(k, pk, radii)
            S = cov[np.arange(n_r) * (np.arange(n_r) + 1) // 2 + np.arange(n_r)]
            barrier = barrier_values(sigma.astype(float), bp)
            f, _ = emu.multiplicity(radii, S, barrier, dvar, check=False)
            centres = 0.5 * (radii[:-1] + radii[1:])
            curves[n_r] = (centres, f)

        # compare at a common radius: the midpoint of the 64-grid range
        R = float(np.exp(np.mean(np.log(curves[64][0]))))
        vals = [float(np.interp(R, c, f)) for (c, f) in curves.values()]
        spread = (max(vals) - min(vals)) / np.mean(vals)
        worst = max(worst, spread)
        print(f"{tag:>8}{R:>12.3f}{vals[0]:>13.5e}{vals[1]:>13.5e}"
              f"{vals[2]:>13.5e}{spread:>10.2%}")

    print(f"\n  worst spread across grid resolutions: {worst:.2%}")
    print("  (the Monte Carlo itself moves by ~1% between 50 and 100 radii;\n"
          "   the emulator predicts a continuum quantity and should not)")


def stage_guard(mc_path, weights):
    emu = EPEmulator.load(weights)
    d = np.load(mc_path)
    tag = sorted({k.split("__")[0] for k in d.files if k.startswith("v")})[0]
    radii = d[f"{tag}__radii"].astype(float)
    S = d[f"{tag}__S"]
    barrier = d[f"{tag}__barrier"].astype(float)
    dvar = d[f"{tag}__dvar"]

    print(f"\n{'=' * 84}\n3. the domain guard, deliberately provoked\n{'=' * 84}")

    _, info = emu.multiplicity(radii, S, barrier, dvar)
    print(f"  as generated             ok={info['ok']}  outside={info['outside']}"
          f"  first-step mass={info['first_step_mass']:.1e}")

    # Truncate the grid from the outside: the walk then starts too low.
    for keep in (100, 80, 60):
        r2, S2, b2, v2 = radii[:keep], S[:keep], barrier[:keep], dvar[:keep]
        _, info2 = emu.multiplicity(r2, S2, b2, v2)
        msg = info2["messages"][0][:66] if info2["messages"] else ""
        print(f"  grid cut to {keep:>3} radii     ok={info2['ok']}  "
              f"first-step mass={info2['first_step_mass']:.1e}")
        if msg:
            print(f"      -> {msg}...")

    # A barrier far outside the trained amplitude.
    _, info3 = emu.multiplicity(radii, S, barrier * 4.0, dvar)
    print(f"  barrier x4               ok={info3['ok']}  "
          f"outside={info3['outside']} of {len(radii) - 1} bins")
    if info3["messages"]:
        print(f"      -> {info3['messages'][0][:70]}...")


def stage_speed(mc_path, weights):
    emu = EPEmulator.load(weights)
    d = np.load(mc_path)
    tag = sorted({k.split("__")[0] for k in d.files if k.startswith("v")})[0]
    radii = d[f"{tag}__radii"].astype(float)
    S = d[f"{tag}__S"]
    barrier = d[f"{tag}__barrier"].astype(float)
    dvar = d[f"{tag}__dvar"]

    for _ in range(3):
        emu.multiplicity(radii, S, barrier, dvar, check=False)
    t0 = time.perf_counter()
    N = 200
    for _ in range(N):
        emu.multiplicity(radii, S, barrier, dvar, check=False)
    dt = (time.perf_counter() - t0) / N

    print(f"\n{'=' * 84}\n4. speed\n{'=' * 84}")
    print(f"  emulator, 128 radii, numpy:   {dt * 1e3:.2f} ms")
    print(f"  Monte Carlo at 1e9 paths:     ~{1e9 / 2.4e6:.0f} s")
    print(f"  speed-up:                     {(1e9 / 2.4e6) / dt:.3e}x")
    print("  (the C version has no numpy overhead and should be well under "
          "0.1 ms)")


# The cross-validated accuracy quoted in the text, and the same number
# ep_train_final.py stores in the weights file as acc__median: the median
# relative error on the multiplicity over 143256 usable bins of the pure-SMT
# curves, five-fold, held out by whole cosmology.
#
# Drawn as a reference line rather than recomputed here, and the reason is the
# path count. That measurement is against the training runs at 1e9 paths, where
# the counting noise on those bins has a median of 0.060% -- less than half the
# 0.128% it reports, so the emulator is what is being measured. This stage runs
# a fresh ensemble at 1e8, where the same noise is 0.185% and sits ABOVE the
# quantity of interest. Twenty-four curves at 1e8 can bound the emulator and
# show that the bound holds everywhere in the barrier box, which is what the
# figure is for; they cannot re-measure the central value, and a line drawn
# from them would be a line drawn from noise.
CV_ACCURACY = 0.00128


def curve_accuracy(mc_path, weights):
    """Per-curve median accuracy and per-curve median Monte Carlo noise.

    One number per curve rather than per bin, because the question the figure
    answers is whether any corner of the barrier box is worse than the others,
    and a per-bin cloud buries that under the nu dependence the other figure
    already shows.

    Both numbers are medians over the same bins, which is what makes them
    comparable: the counting error varies by a factor of twenty within a single
    curve, so its RMS is set by a handful of tail bins and would not describe
    the noise the median accuracy is actually measured against.
    """
    emu = EPEmulator.load(weights)
    d = np.load(mc_path)
    n_paths = int(d["n_paths"])
    tags = sorted({k.split("__")[0] for k in d.files if k.startswith("v")})

    out = []
    for tag in tags:
        radii = d[f"{tag}__radii"].astype(float)
        S = d[f"{tag}__S"]
        barrier = d[f"{tag}__barrier"].astype(float)
        dvar = d[f"{tag}__dvar"]
        counts = d[f"{tag}__counts"].astype(float)

        f_emu, info = emu.multiplicity(radii, S, barrier, dvar)
        f_mc = counts[:-1] / (n_paths * np.diff(radii))
        c = counts[:-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            noise = np.where(c > 0, 1.0 / np.sqrt(np.maximum(c, 1)), np.inf)
            resid = np.where((f_mc > 0) & (f_emu > 0),
                             np.log(f_emu) - np.log(f_mc), np.nan)

        # The same 5% cut stage_analyse uses: past it the reference carries no
        # information about the emulator at all.
        good = np.isfinite(resid) & (noise < 0.05)

        out.append(dict(
            tag=tag,
            alpha=float(d[f"{tag}__alpha"]),
            beta=float(d[f"{tag}__beta"]),
            gamma=float(d[f"{tag}__gamma"]),
            err=float(np.median(np.abs(np.expm1(resid[good])))),
            noise=float(np.median(noise[good])),
            outside=int(info["outside"]) > 0))

    return out


def stage_plot(mc_path, weights, out_dir, png):
    """Accuracy against each barrier parameter, one point per curve.

    Gold is the Monte Carlo and deep blue the emulator, as in the validation
    figure, and the stem joins the two numbers belonging to the SAME curve --
    without it the eye pairs points by height and reads a spread that is not
    there.

    For almost every curve the measured departure sits at or below the
    reference's own counting noise, which is the honest reading of this test:
    at 1e8 paths it bounds the emulator rather than resolving it, and the bound
    it gives is consistent with the cross-validated accuracy drawn as the
    dotted line.

    Every font size is left to voidstyle. Setting them locally is what makes a
    figure look foreign next to the others in the same chapter.

    The alpha panel is the one with something to say. Three of the four curves
    that stand clear of the noise are the three lowest barrier amplitudes in
    the set, and the domain guard rejects one of those without having been told
    about any of this. Against beta and gamma there is no trend to see, which
    is the point of drawing all three.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import voidstyle as vs
        vs.use_style()
        C_MC, C_EMU = vs.gold, vs.deep_blue
    except ImportError:
        C_MC, C_EMU = "#FFB84D", "#163F6E"

    rows = curve_accuracy(mc_path, weights)
    err = 100.0 * np.array([r["err"] for r in rows])
    noise = 100.0 * np.array([r["noise"] for r in rows])
    outside = np.array([r["outside"] for r in rows])

    # Tall enough that the y label at the shared axes.labelsize fits along the
    # axis rather than running off it, which is what forces the row to be this
    # deep for three short panels.
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 4.4), sharey=True,
                           gridspec_kw=dict(wspace=0.07))

    panels = [(r"$\alpha$", "alpha", (-0.05, 2.62)),
              (r"$\beta$", "beta", (-0.05, 1.58)),
              (r"$\gamma$", "gamma", (0.42, 3.28))]

    for a, (label, key, xlim) in zip(ax, panels):
        x = np.array([r[key] for r in rows])

        a.axhline(100.0 * CV_ACCURACY, color="0.35", lw=0.9, ls=":", zorder=2,
                  label="cross-validated accuracy")
        a.vlines(x, np.minimum(err, noise), np.maximum(err, noise),
                 color="0.72", lw=0.9, zorder=3)
        a.scatter(x, noise, s=27, facecolors="none", edgecolors=C_MC, lw=1.3,
                  zorder=4, label="Monte Carlo counting noise")
        a.scatter(x[~outside], err[~outside], s=23, color=C_EMU, lw=0,
                  zorder=5, label="emulator")
        # Kept in rather than dropped: the guard rejecting the curve is part of
        # the result, and a gap where a bad point used to be would hide it.
        if outside.any():
            a.scatter(x[outside], err[outside], s=46, facecolors="none",
                      edgecolors=C_EMU, lw=1.5, zorder=6,
                      label="rejected by the domain guard")

        a.set(xlabel=label, xlim=xlim, ylim=(0.10, 0.40))

    ax[0].set_ylabel(r"median per-bin error  [%]")

    # One legend for the row, above it: four entries will not fit inside a
    # panel this size without landing on the points they describe.
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.90), handletextpad=0.5,
               columnspacing=1.6)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf"] + (["png"] if png else []):
        path = os.path.join(out_dir, f"ep_domain_accuracy.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    print(f"\n  {len(rows)} curves: median accuracy {np.median(err):.3f}%, "
          f"worst {err.max():.3f}%")
    print(f"  their Monte Carlo noise: median {np.median(noise):.3f}%, "
          f"worst {noise.max():.3f}%")
    over = err > 1.25 * noise
    print(f"  departure exceeds the noise by >25% on {int(over.sum())} curves, "
          f"at alpha = "
          + ", ".join(f"{r['alpha']:.3f}" for r, o in zip(rows, over) if o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["mc", "analyse", "plot", "all"],
                    default="analyse")
    ap.add_argument("--mc", default=os.path.join(HERE, "ep_validation_mc.npz"))
    ap.add_argument("--weights", default=os.path.join(HERE, "ep_emulator.npz"))
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    args.out_dir = os.path.expanduser(args.out_dir)

    if args.stage == "mc":
        stage_mc(args.mc)
        return

    if args.stage == "plot":
        stage_plot(args.mc, args.weights, args.out_dir, args.png)
        return

    stage_analyse(args.mc, args.weights)
    stage_grid(args.mc, args.weights)
    stage_guard(args.mc, args.weights)
    stage_speed(args.mc, args.weights)

    if args.stage == "all":
        stage_plot(args.mc, args.weights, args.out_dir, args.png)


if __name__ == "__main__":
    main()
