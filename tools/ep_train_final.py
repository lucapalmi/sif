#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
Trains the final emulator on the whole training set and writes the weights.

Both the training targets and the deployed model go through
ep_model.compute_features, so the fit and inference cannot disagree about what
a feature means -- there is no second definition anywhere to drift.

Reports cross-validated accuracy first (held out by whole cosmologies, scored
against the Monte Carlo's own multiplicity rather than any reconstruction),
then refits on everything and saves the weights plus the domain box.

The two campaigns share cosmologies by task id, so a fold holds a cosmology
out of both at once. Accuracy is reported over all bins, and over the bins
below --rmax (80 Mpc/h by default), which is where the accuracy target is set.

Usage:
  python tools/ep_train_final.py runs runs_v2 runs_v3 [-o ep_emulator.npz]
  python tools/ep_train_final.py runs runs_v2 --hidden 32,32 --cv-only \\
      --json cv_32x32.json
  # folds as separate jobs, then pooled into the saved model's accuracy:
  python tools/ep_train_final.py runs runs_v2 --fold 0 --fold-out f0.npz
  python tools/ep_train_final.py --merge f0.npz f1.npz f2.npz f3.npz \\
      -o ep_emulator.npz
"""

import argparse
import json
import os

# The fitter parallelises over blocks of rows itself (ep_emulator.BLOCK_ROWS);
# a multithreaded BLAS underneath it oversubscribes the cores and was measured
# to stall outright. Must be set before numpy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402

from ep_emulator import EmulatorMLP  # noqa: E402
from ep_model import (EPEmulator, FEATURE_NAMES,  # noqa: E402
                      multiplicity_from_hazard, upper_tail)
from ep_trainset_load import MAX_FIRST_STEP, load_runs  # noqa: E402

HIDDEN = (48, 48)

# Walks starting beyond this are left out of the fit. The emulator's contract
# is walks starting at 1-250 Mpc/h -- 250 is about where the shallowest
# realistic void threshold at low redshift starts its walk -- and this keeps
# a margin past it while spending none of the network on the 300-700 Mpc/h
# end that campaign 2 also sampled. Beyond the margin the inputs (gamma2 first)
# leave the trained box, so the guard reports such calls.
MAX_ORIGIN = 300.0

# The guard's upper edge for d ln nu / d ln S, below the trained maximum. An
# SMT barrier never exceeds -0.5 (a constant one sits on it); only the 5%
# ripples reach above, at their local peaks, so past -0.40 the training holds
# 0.14% of its bins and its maximum (-0.215) is set by 14 of them. Measured
# against the Monte Carlo, barriers rising with sigma were sound at -0.44
# (0.16%) and wrong at -0.375 (0.35%) and -0.25 (0.83%), with the box passing
# them all.
DLNNU_DLNS_MAX = -0.40
# Fit y / rms(y) and divide the output layer back. Off, the data term near the
# optimum (~1e-6) sat below the L2 term (~1e-3), which then set the weights:
# measured, CV below 80 Mpc/h 0.081% unscaled against 0.061% scaled.
SCALE_TARGET = os.environ.get("EP_SCALE_TARGET", "1") == "1"
N_TRAIN = 0           # bins drawn for the fit; 0 = every usable bin
MAXITER = 6000
N_FOLDS = 4
SEED = 11
R_MAX = 80.0


def fit(curves, hidden=HIDDEN, maxiter=MAXITER, n_train=N_TRAIN, seed=SEED):
    """Unweighted L-BFGS fit of log(Lambda_MC / Lambda_up).

    Flat weights, not inverse-variance: the Monte Carlo error varies 25x across
    the nu range, and inverse-variance weighting was measured to underfit the
    high-nu end by 2.5x while leaving the overall RMS unchanged. The large-
    radius end is where the cosmological information sits, so it gets equal say.
    """
    X = np.vstack([c["X"][c["mask"]] for c in curves])
    y = np.concatenate([np.log(c["ratio"][c["mask"]]) for c in curves])

    if n_train and len(y) > n_train:
        idx = np.random.default_rng(seed).choice(len(y), n_train,
                                                 replace=False)
        X, y = X[idx], y[idx]

    if not SCALE_TARGET:
        return EmulatorMLP(hidden=hidden, seed=seed).fit(
            X, y, np.ones(len(y)), maxiter=maxiter)

    # Fitted on y / rms(y), then the output layer divided back: the same
    # network, but with the data term O(1) rather than ~1e-6 near the optimum,
    # where the L2 term (~1e-3) would otherwise set the weights.
    c = 1.0 / np.sqrt(np.mean(y * y))
    m = EmulatorMLP(hidden=hidden, seed=seed).fit(
        X, c * y, np.ones(len(y)), maxiter=maxiter)
    layers = m._unpack(m.theta_)
    W, b = layers[-1]
    W /= c
    b /= c
    m.theta_ = np.concatenate([np.concatenate([Wl.ravel(), bl.ravel()])
                               for (Wl, bl) in layers])
    print(f"fit: {m.nit_} iterations, loss {m.loss_:.4e} ({m.message_})",
          flush=True)
    return m


def as_emulator(mlp, curves):
    """Wrap the fitted network with the domain the training actually covered.

    `lag` and `cum_lam` get a floor of zero rather than their trained minimum.
    At the walk's first bin both are set by half a grid step and nothing more
    -- lag is ln(S/S_o) clamped at the origin, cum_lam is half that bin's
    hazard -- so any grid finer than training's put that one bin "outside",
    and every call on a 200-radius grid was reported out of domain.

    The box spans every bin the training walks visited, not only the bins
    noisy enough to have been fitted. Bins beyond nu ~ 5 see almost no
    crossings and fail the noise cut, but the walks did start there (the
    design draws nu_origin up to 5.5); their hazard is ~1e-6 of the total, so
    what the network does there cannot move the answer, and flagging them only
    made the guard report sound calls as out of domain.
    """
    mu, sd, layers = mlp.weights()
    X = np.vstack([c["X"][np.all(np.isfinite(c["X"]), axis=1)] for c in curves])
    lo, hi = X.min(axis=0), X.max(axis=0)
    for name in ("lag", "cum_lam"):
        lo[FEATURE_NAMES.index(name)] = 0.0
    j = FEATURE_NAMES.index("dlnnu_dlnS")
    hi[j] = min(hi[j], DLNNU_DLNS_MAX)
    nu_origin_min = min(c["nu_large"] for c in curves)
    return EPEmulator(mu, sd, layers, (lo, hi), nu_origin_min)


def score(emu, curves):
    """Per-bin relative error on the multiplicity, against the Monte Carlo.

    f_mc comes straight from the counts, not from running the counts back
    through the survival recursion: reconstructing both sides the same way
    would hide an error common to both.
    """
    keys = ("rel", "nu", "noise", "logres", "r", "origin", "campaign")
    out = {k: [] for k in keys}
    for c in curves:
        m = c["mask"]
        if m.sum() == 0:
            continue
        pred = emu.correction(c["X"])
        lam = c["lam_up"] * np.exp(pred)
        f_emu = multiplicity_from_hazard(lam, c["radii"], c["nu_large"])
        f_mc = c["counts"][:-1] / (c["n_paths"] * np.diff(c["radii"]))

        with np.errstate(divide="ignore", invalid="ignore"):
            out["rel"].append(np.where(f_mc > 0, (f_emu - f_mc) / f_mc,
                                       np.nan)[m])
        out["nu"].append(c["nu"][m])
        out["noise"].append(c["rel_err"][m])
        out["logres"].append(pred[m] - np.log(c["ratio"][m]))
        out["r"].append((0.5 * (c["radii"][1:] + c["radii"][:-1]))[m])
        out["origin"].append(np.full(m.sum(), c["r_origin"]))
        out["campaign"].append(np.full(m.sum(), c["campaign"]))
    return {k: np.concatenate(v) for k, v in out.items()}


def summarise(s, sel, label):
    """One line: median and 95th |error|, noise-free model error, nu bands."""
    a = np.abs(s["rel"][sel])
    ok = np.isfinite(a)
    if ok.sum() == 0:
        print(f"  {label:<34}  no bins")
        return None
    nu = s["nu"][sel]
    r = np.sqrt(np.mean(s["logres"][sel] ** 2))
    n = np.sqrt(np.mean(s["noise"][sel] ** 2))
    model = np.sqrt(max(r * r - n * n, 0.0))
    bands = []
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 99)]:
        b = ok & (nu >= lo) & (nu < hi)
        bands.append(np.median(a[b]) if b.sum() else np.nan)
    print(f"  {label:<34}{ok.sum():>8}{np.median(a[ok]):>9.3%}"
          f"{np.percentile(a[ok], 95):>8.2%}{model:>10.2e}"
          + "".join(f"{b:>8.2%}" for b in bands))
    return dict(n=int(ok.sum()), median=float(np.median(a[ok])),
                p95=float(np.percentile(a[ok], 95)), model_error=float(model))


def report(s, r_max, title):
    print(f"\n{'=' * 98}\n{title}\n{'=' * 98}")
    print(f"  {'evaluated on':<34}{'bins':>8}{'median':>9}{'95th':>8}"
          f"{'model err':>10}{'nu<1':>8}{'1-2':>8}{'2-3':>8}{'>3':>8}")
    print("  " + "-" * 96)
    one = s["campaign"] == 1
    later = s["campaign"] > 1
    low = s["r"] < r_max
    acc = {}
    acc["all"] = summarise(s, np.ones_like(one), "all bins")
    acc["campaign1"] = summarise(s, one, "campaign 1 (origin < 51)")
    acc["campaigns2_4"] = summarise(s, later, "campaigns 2-4 (origin-placed)")
    acc["below_rmax"] = summarise(s, low, f"all bins, R < {r_max:g}")
    for lo, hi in ((0, 30), (30, 120), (120, 1000)):
        sel = low & (s["origin"] >= lo) & (s["origin"] < hi)
        acc[f"below_rmax_origin_{lo}_{hi}"] = summarise(
            s, sel, f"  R < {r_max:g}, origin {lo}-{hi}")
    acc["above_rmax"] = summarise(s, ~low, f"all bins, R >= {r_max:g}")
    return acc


CV_TITLE = (f"cross-validated ({N_FOLDS}-fold, held out by cosmology across "
            f"both campaigns), against the Monte Carlo")


def cv_fold(curves, k, hidden, maxiter, n_train):
    """Held-out scores of fold k: fit on the other folds, score this one."""
    tasks = sorted({c["task"] for c in curves})
    held = set(np.random.default_rng(SEED).permutation(tasks)[k::N_FOLDS])
    tr = [c for c in curves if c["task"] not in held]
    emu = as_emulator(fit(tr, hidden, maxiter, n_train), tr)
    return score(emu, [c for c in curves if c["task"] in held])


def cross_validate(curves, hidden, maxiter, n_train, r_max):
    parts = [cv_fold(curves, k, hidden, maxiter, n_train)
             for k in range(N_FOLDS)]
    s = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return report(s, r_max, CV_TITLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*",
                    default=["../runs", "../runs_v2", "../runs_v3",
                             "../runs_v4"])
    ap.add_argument("-o", "--out", default="ep_emulator.npz")
    ap.add_argument("--hidden", default=",".join(map(str, HIDDEN)),
                    help="hidden layer widths, comma separated")
    ap.add_argument("--maxiter", type=int, default=MAXITER)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--rmax", type=float, default=R_MAX)
    ap.add_argument("--max-origin", type=float, default=MAX_ORIGIN,
                    help="leave out walks starting beyond this radius")
    ap.add_argument("--cv-only", action="store_true")
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--json", help="write the accuracy summary here")
    ap.add_argument("--fold", type=int,
                    help="run only this CV fold and save its held-out scores "
                         "to --fold-out, so folds can run as separate jobs")
    ap.add_argument("--fold-out", default="cv_fold.npz")
    ap.add_argument("--merge", nargs="+", metavar="FOLD_NPZ",
                    help="pool the --fold outputs and report; with -o, also "
                         "record the result as that model's accuracy")
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))

    if args.merge:
        parts = [dict(np.load(p)) for p in args.merge]
        s = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        cv = report(s, args.rmax, CV_TITLE)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(dict(cv=cv), fh, indent=1)
        emu = EPEmulator.load(args.out)
        emu.accuracy.update({k: v for k, v in (
            ("median", cv["below_rmax"]["median"]),
            ("median_all", cv["all"]["median"]),
            ("p95", cv["below_rmax"]["p95"]),
            ("model_error", cv["below_rmax"]["model_error"]),
            ("rmax", args.rmax))})
        emu.save(args.out)
        print(f"\n  CV accuracy recorded in {args.out}")
        return

    curves = load_runs(args.runs, max_origin=args.max_origin)
    if args.fold is not None:
        np.savez(args.fold_out, **cv_fold(curves, args.fold, hidden,
                                          args.maxiter, args.n_train))
        print(f"fold {args.fold} held-out scores -> {args.fold_out}")
        return
    pure = [c for c in curves if not c["perturbed"]]
    print(f"{len(curves)} curves ({len(pure)} pure SMT) from "
          f"{' '.join(args.runs)}, {sum(int(c['mask'].sum()) for c in curves)} "
          f"usable bins; network {len(FEATURE_NAMES)} -> "
          f"{' -> '.join(map(str, hidden))} -> 1")

    summary = dict(hidden=list(hidden), maxiter=args.maxiter,
                   n_train=args.n_train)
    if not args.no_cv:
        summary["cv"] = cross_validate(curves, hidden, args.maxiter,
                                       args.n_train, args.rmax)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)
    if args.cv_only:
        return

    emu = as_emulator(fit(curves, hidden, args.maxiter, args.n_train), curves)
    cv = summary.get("cv") or {}
    # What the header advertises: the held-out median below rmax, where the
    # target is set, and the overall one.
    emu.accuracy = {k: v for k, v in (
        ("median", (cv.get("below_rmax") or {}).get("median")),
        ("median_all", (cv.get("all") or {}).get("median")),
        ("p95", (cv.get("below_rmax") or {}).get("p95")),
        ("model_error", (cv.get("below_rmax") or {}).get("model_error")),
        ("rmax", args.rmax)) if v is not None}
    report(score(emu, curves), args.rmax, f"final fit on all {len(curves)} "
           f"curves (in-sample)")

    # The figure a caller is quoted when the guard fires: the error on the
    # curves training excluded for starting too close to the barrier, which is
    # the out-of-domain case the guard exists for.
    outside = [c for c in load_runs(args.runs, keep_truncated=True,
                                    max_origin=args.max_origin)
               if c["first_step"] > MAX_FIRST_STEP]
    if outside:
        s = score(emu, outside)
        a = np.abs(s["rel"][np.isfinite(s["rel"])])
        emu.accuracy["median_outside"] = float(np.median(a))
        print(f"\n  {len(outside)} curves starting below nu_origin_min: median "
              f"error {np.median(a):.3%}, 95th {np.percentile(a, 95):.2%}")

    print(f"\n  parameters      {emu.n_parameters()}")
    print(f"  architecture    {len(FEATURE_NAMES)} -> "
          f"{' -> '.join(str(h) for h in hidden)} -> 1  (tanh)")
    print(f"  features        {', '.join(FEATURE_NAMES)}")
    print(f"  nu_origin_min   {emu.nu_origin_min:.4f} "
          f"(first-step mass {float(upper_tail(emu.nu_origin_min)):.2%})")
    print("  trained ranges:")
    lo, hi = emu.box
    for j, name in enumerate(FEATURE_NAMES):
        print(f"    {name:<12}{lo[j]:>13.5f}  to {hi[j]:>13.5f}")

    emu.save(args.out)
    print(f"\nweights -> {args.out}")


if __name__ == "__main__":
    main()
