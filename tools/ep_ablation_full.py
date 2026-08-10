#!/usr/bin/env python
"""
The feature ablation, repeated at full statistics.

At 2e7 paths the pilot said local features saturate and history adds nothing.
That conclusion was drawn against a Monte Carlo noise floor of 8e-3, and the
fit sat at 1.1e-2 -- only 1.4x above it. The floor is now about thirty times
lower, so a history dependence of a few parts per thousand, invisible then,
would show here. This is the test that decides the feature set the C
implementation will carry.

Scored two ways, because they answer different questions: the weighted RMS in
log hazard is what the fit minimizes, and the relative error on the
multiplicity rebuilt through the survival recursion is what a likelihood sees.
Both are reported per nu band, since the large-radius (high nu) end is where
the cosmological information sits and where an error budget matters most.

Usage:  python tools/ep_ablation_full.py [runs_dir]
"""

import sys

import numpy as np

from ep_emulator import EmulatorMLP, build_matrix
from ep_model import multiplicity_from_hazard, upper_tail
from ep_trainset_load import MAX_FIRST_STEP as DROP_TRUNCATED
from ep_trainset_load import MAX_REL_ERR, load_runs

N_TRAIN_MAX = 40000     # bins per fit; the fit is L-BFGS over all of them
SEED = 3

# Nested sets, in the order the ablation walks them. Every name is a column of
# the canonical feature matrix, so nothing is recomputed here.
FEATURE_SETS = {
    "nu": ["nu"],
    "nu+y": ["nu", "y"],
    "nu+g2": ["nu", "gamma2"],
    "nu+g2+y": ["nu", "gamma2", "y"],
    "nu+g2+y+dnu": ["nu", "gamma2", "y", "dlnnu_dlnS"],
    "nu+g2+y+hist": ["nu", "gamma2", "y", "nu_back", "lag"],
    "full (8)": ["nu", "gamma2", "y", "dlnnu_dlnS", "nu_back", "lag",
                 "cum_lam", "nu_origin"],
}


def subsample(X, y, w, meta, n_max, seed=SEED):
    if X.shape[0] <= n_max:
        return X, y, w
    idx = np.random.default_rng(seed).choice(X.shape[0], n_max, replace=False)
    return X[idx], y[idx], w[idx]


def evaluate(model, curves, feats):
    """Residuals in log hazard and in the reconstructed multiplicity."""
    log_res, rel_f, nus, noise = [], [], [], []
    for c in curves:
        m = c["mask"]
        if m.sum() == 0:
            continue
        pred = model.predict(np.column_stack([c[f][m] for f in feats]))
        log_res.append(pred - np.log(c["ratio"][m]))
        nus.append(c["nu"][m])
        noise.append(c["rel_err"][m])

        lam = c["lam_up"].copy()
        lam[m] *= np.exp(pred)
        f_pred = multiplicity_from_hazard(lam, c["radii"], c["nu_large"])
        # Against the Monte Carlo's own multiplicity, straight from the counts.
        dr = np.diff(c["radii"])
        f_mc = c["counts"][:-1] / (c["n_paths"] * dr)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_f.append(np.where(f_mc > 0, (f_pred - f_mc) / f_mc,
                                  np.nan)[m])
    return tuple(np.concatenate(a) for a in (log_res, rel_f, nus, noise))


def weighted_rms(res, noise):
    w = 1.0 / np.maximum(noise, 1e-6) ** 2
    return np.sqrt(np.sum(w * res ** 2) / np.sum(w))


def folds_by_cosmology(curves, n_folds=4):
    tasks = sorted({c["task"] for c in curves})
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(tasks)
    return [set(shuffled[i::n_folds]) for i in range(n_folds)]


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "../runs"

    curves = load_runs(run_dir)
    n_all = len(curves)

    n_bins = sum(int(c["mask"].sum()) for c in curves)
    floor = np.concatenate([c["rel_err"][c["mask"]] for c in curves])
    floor_rms = weighted_rms(floor, floor)

    print(f"{len(curves)} curves (loader already drops those with a "
          f"first-step mass above {DROP_TRUNCATED:.0%}), {n_bins} bins")
    print(f"Monte Carlo noise floor: weighted RMS {floor_rms:.2e}, "
          f"median {np.median(floor):.2e}")

    fold_sets = folds_by_cosmology(curves)

    print(f"\n{'=' * 92}\nfeature ablation, {len(fold_sets)}-fold by cosmology"
          f"\n{'=' * 92}")
    print(f"{'features':<16}{'wRMS log':>10}{'/floor':>8}"
          f"{'med|df/f|':>11}{'95th':>9}"
          f"{'|df/f| nu<2':>13}{'nu 2-3':>9}{'nu>3':>9}{'params':>8}")
    print("-" * 92)

    results = {}
    for name, feats in FEATURE_SETS.items():
        L, R, NU, NS = [], [], [], []
        n_par = 0
        for held in fold_sets:
            tr = [c for c in curves if c["task"] not in held]
            te = [c for c in curves if c["task"] in held]

            X, y, w, meta = build_matrix(tr, feats)
            X, y, w = subsample(X, y, w, meta, N_TRAIN_MAX)

            m = EmulatorMLP(hidden=(16, 16), seed=SEED).fit(X, y, w)
            n_par = m.n_parameters()

            lr, rf, nu, ns = evaluate(m, te, feats)
            L.append(lr)
            R.append(rf)
            NU.append(nu)
            NS.append(ns)

        lr = np.concatenate(L)
        rf = np.concatenate(R)
        nu = np.concatenate(NU)
        ns = np.concatenate(NS)
        arf = np.abs(rf)
        ok = np.isfinite(arf)

        band = []
        for lo, hi in [(0, 2), (2, 3), (3, 99)]:
            s = ok & (nu >= lo) & (nu < hi)
            band.append(np.median(arf[s]) if s.sum() else np.nan)

        w_rms = weighted_rms(lr, ns)
        print(f"{name:<16}{w_rms:>10.2e}{w_rms / floor_rms:>8.1f}"
              f"{np.median(arf[ok]):>11.2%}{np.percentile(arf[ok], 95):>9.2%}"
              f"{band[0]:>13.2%}{band[1]:>9.2%}{band[2]:>9.2%}{n_par:>8}")

        results[name] = dict(log_res=lr, rel_f=rf, nu=nu, noise=ns)

    print("\n  /floor is the fit's weighted RMS divided by the Monte Carlo's "
          "own.\n  A value near 1 means the data, not the model, is the "
          "limitation.")
    return results


if __name__ == "__main__":
    main()
