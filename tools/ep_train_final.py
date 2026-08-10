#!/usr/bin/env python
"""
Trains the final emulator on the whole training set and writes the weights.

Both the training targets and the deployed model go through
ep_model.compute_features, so the fit and inference cannot disagree about what
a feature means -- there is no second definition anywhere to drift.

Reports cross-validated accuracy first (held out by whole cosmologies, scored
against the Monte Carlo's own multiplicity rather than any reconstruction),
then refits on everything and saves the weights plus the domain box.

Usage:  python tools/ep_train_final.py [runs_dir] [-o ep_emulator.npz]
"""

import argparse

import numpy as np

from ep_emulator import EmulatorMLP
from ep_model import (EPEmulator, FEATURE_NAMES, multiplicity_from_hazard,
                      upper_tail)
from ep_trainset_load import load_runs

HIDDEN = (16, 16)
N_TRAIN = 60000       # bins drawn for the fit; L-BFGS runs over all of them
MAXITER = 1500
N_FOLDS = 4
SEED = 11


def fit(curves, seed=SEED, n_train=N_TRAIN):
    """Weighted-uniform L-BFGS fit of log(Lambda_MC / Lambda_up).

    Flat weights, not inverse-variance: the Monte Carlo error varies 25x across
    the nu range, and inverse-variance weighting was measured to underfit the
    high-nu end by 2.5x while leaving the overall RMS unchanged. The large-
    radius end is where the cosmological information sits, so it gets equal say.
    """
    X = np.vstack([c["X"][c["mask"]] for c in curves])
    y = np.concatenate([np.log(c["ratio"][c["mask"]]) for c in curves])

    if len(y) > n_train:
        idx = np.random.default_rng(seed).choice(len(y), n_train,
                                                 replace=False)
        X, y = X[idx], y[idx]

    return EmulatorMLP(hidden=HIDDEN, seed=seed).fit(
        X, y, np.ones(len(y)), maxiter=MAXITER)


def as_emulator(mlp, curves):
    """Wrap the fitted network with the domain the training actually covered."""
    mu, sd, layers = mlp.weights()
    X = np.vstack([c["X"][c["mask"]] for c in curves])
    box = (X.min(axis=0), X.max(axis=0))
    nu_origin_min = min(c["nu_large"] for c in curves)
    return EPEmulator(mu, sd, layers, box, nu_origin_min)


def score(emu, curves):
    """Relative error on the multiplicity, against the Monte Carlo directly.

    f_mc comes straight from the counts, not from running the counts back
    through the survival recursion: reconstructing both sides the same way
    would hide an error common to both.
    """
    rel, nus, noise, logres = [], [], [], []
    for c in curves:
        m = c["mask"]
        if m.sum() == 0:
            continue
        pred = emu.correction(c["X"])
        lam = c["lam_up"] * np.exp(pred)
        f_emu = multiplicity_from_hazard(lam, c["radii"], c["nu_large"])
        f_mc = c["counts"][:-1] / (c["n_paths"] * np.diff(c["radii"]))

        with np.errstate(divide="ignore", invalid="ignore"):
            rel.append(np.where(f_mc > 0, (f_emu - f_mc) / f_mc, np.nan)[m])
        nus.append(c["nu"][m])
        noise.append(c["rel_err"][m])
        logres.append(pred[m] - np.log(c["ratio"][m]))

    return tuple(np.concatenate(a) for a in (rel, nus, noise, logres))


def summarize(rel, nu, noise, logres, label):
    a = np.abs(rel)
    ok = np.isfinite(a)
    r = np.sqrt(np.mean(logres ** 2))
    n = np.sqrt(np.mean(noise ** 2))
    model = np.sqrt(max(r * r - n * n, 0.0))
    bands = [np.median(a[ok & (nu >= lo) & (nu < hi)])
             for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 99)]]
    print(f"  {label:<26}{np.median(a[ok]):>9.3%}"
          f"{np.percentile(a[ok], 95):>9.2%}{model:>11.2e}"
          + "".join(f"{b:>9.2%}" for b in bands))
    return dict(median=float(np.median(a[ok])),
                p95=float(np.percentile(a[ok], 95)),
                model_error=float(model))


def cross_validate(curves, pure):
    tasks = sorted({c["task"] for c in curves})
    sh = np.random.default_rng(SEED).permutation(tasks)
    folds = [set(sh[i::N_FOLDS]) for i in range(N_FOLDS)]

    acc = {"pure": [[], [], [], []], "all": [[], [], [], []]}
    for held in folds:
        tr = [c for c in curves if c["task"] not in held]
        emu = as_emulator(fit(tr), tr)
        for key, pool in (("pure", pure), ("all", curves)):
            te = [c for c in pool if c["task"] in held]
            if not te:
                continue
            for slot, arr in zip(acc[key], score(emu, te)):
                slot.append(arr)

    out = {}
    for key, label in (("pure", "pure SMT (the contract)"),
                       ("all", "all curves")):
        parts = [np.concatenate(s) for s in acc[key]]
        out[key] = summarize(*parts, label)
    return out["pure"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="../runs")
    ap.add_argument("-o", "--out", default="ep_emulator.npz")
    args = ap.parse_args()

    curves = load_runs(args.runs)
    pure = [c for c in curves if not c["perturbed"]]
    print(f"{len(curves)} curves ({len(pure)} pure SMT), "
          f"{sum(int(c['mask'].sum()) for c in curves)} usable bins")

    print(f"\n{'=' * 88}\ncross-validated ({N_FOLDS}-fold, held out by "
          f"cosmology), against the Monte Carlo directly\n{'=' * 88}")
    print(f"  {'evaluated on':<26}{'median':>9}{'95th':>9}{'model err':>11}"
          f"{'nu<1':>9}{'1-2':>9}{'2-3':>9}{'>3':>9}")
    print("  " + "-" * 86)
    accuracy = cross_validate(curves, pure)

    print(f"\n{'=' * 88}\nfinal fit on all {len(curves)} curves\n{'=' * 88}")
    emu = as_emulator(fit(curves), curves)
    emu.accuracy = accuracy
    summarize(*score(emu, pure), "in-sample, pure SMT")

    print(f"\n  parameters      {emu.n_parameters()}")
    print(f"  architecture    {len(FEATURE_NAMES)} -> "
          f"{' -> '.join(str(h) for h in HIDDEN)} -> 1  (tanh)")
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
