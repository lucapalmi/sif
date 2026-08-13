#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
Where the remaining error lives, and why it concentrates at low nu.

Corrected for the Monte Carlo's own noise, the emulator sits at about 0.18%
across most of the nu range and roughly doubles below nu = 1. This asks what is
different about that region, and separates the explanations that would change
what we build from the ones that would not:

  - SELECTION. Low nu is the small-radius end, which the walk reaches LAST,
    after most paths have already crossed. The survivors are a strongly
    conditioned population and their hazard depends on the history of that
    conditioning. If this is the cause the error should track the surviving
    fraction, not nu itself, and more history features would be the remedy.
  - EDGES. Error concentrated at the boundary of feature coverage is
    extrapolation, and the remedy is more training curves out there.
  - A FEW BAD CURVES. If the error is carried by a handful of curves rather
    than spread thinly, whatever those have in common is the real target.
  - BIAS vs SCATTER. A systematic offset means the model is misspecified in a
    fixable way; pure scatter means we are near the limit of the features.

Usage:  python tools/ep_lownu_diagnosis.py [runs_dir] [--plot out.png]
"""

import argparse

import numpy as np

from ep_emulator import EmulatorMLP, build_matrix
from ep_ablation_full import SEED, subsample, weighted_rms
from ep_model import FEATURE_NAMES
from ep_trainset_load import load_runs

# The diagnosis predates cum_lam and nu_origin; it runs on the six features
# that were current when the low-nu weakness was found, which is the point --
# it is what motivated adding the other two.
FEATURES = FEATURE_NAMES[:6]


def collect(curves, n_folds=4, seed=SEED):
    """Cross-validated per-bin residuals, with everything needed to slice."""
    tasks = sorted({c["task"] for c in curves})
    sh = np.random.default_rng(seed).permutation(tasks)
    folds = [set(sh[i::n_folds]) for i in range(n_folds)]

    rows = []
    for held in folds:
        tr = [c for c in curves if c["task"] not in held]
        te = [c for c in curves if c["task"] in held]

        X, y, w, meta = build_matrix(tr, FEATURES)
        X, y, w = subsample(X, y, w, meta, 40000)
        m = EmulatorMLP(hidden=(16, 16), seed=seed).fit(X, y, w, maxiter=900)

        for ci, c in enumerate(te):
            k = c["mask"]
            if k.sum() == 0:
                continue
            pred = m.predict(np.column_stack([c[f][k] for f in FEATURES]))
            res = pred - np.log(c["ratio"][k])
            n = int(k.sum())
            rows.append(dict(
                res=res, nu=c["nu"][k], gamma2=c["gamma2"][k], y=c["y"][k],
                alive=c["alive"][k], noise=c["rel_err"][k],
                ratio=c["ratio"][k],
                alpha=np.full(n, c["alpha"]), beta=np.full(n, c["beta"]),
                gamma=np.full(n, c["gamma"]),
                perturbed=np.full(n, float(c["perturbed"])),
                z=np.full(n, c["z"]), omch2=np.full(n, c["omch2"]),
                curve=np.full(n, hash((c["task"], c["barrier_tag"])) % 10 ** 9),
            ))

    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def model_error(res, noise):
    """RMS residual with the reference's own noise removed in quadrature."""
    r = np.sqrt(np.mean(res ** 2))
    n = np.sqrt(np.mean(noise ** 2))
    return np.sqrt(max(r ** 2 - n ** 2, 0.0)), r, n


def sec(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def by_alive(d):
    sec("1. is it selection? error against the SURVIVING FRACTION")
    print("   low nu is reached last, so few walks are left; if the error "
          "tracks\n   survival rather than nu, the cause is the conditioning, "
          "not the scale")

    print(f"\n{'alive':>16}{'bins':>9}{'model err':>12}{'raw RMS':>10}"
          f"{'noise':>10}{'median nu':>11}")
    print("-" * 68)
    edges = [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 0.80, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (d["alive"] >= lo) & (d["alive"] < hi)
        if s.sum() < 200:
            continue
        me, raw, ns = model_error(d["res"][s], d["noise"][s])
        print(f"  {lo:.2f} - {hi:<8.2f}{s.sum():>9}{me:>12.2e}{raw:>10.2e}"
              f"{ns:>10.2e}{np.median(d['nu'][s]):>11.2f}")

    # The same split at FIXED nu, which is what separates the two explanations.
    print("\n   at fixed nu (0.5 - 1.5), split by survival:")
    print(f"{'alive':>16}{'bins':>9}{'model err':>12}")
    print("-" * 37)
    base = (d["nu"] >= 0.5) & (d["nu"] < 1.5)
    for lo, hi in [(0.0, 0.10), (0.10, 0.30), (0.30, 0.60), (0.60, 1.01)]:
        s = base & (d["alive"] >= lo) & (d["alive"] < hi)
        if s.sum() < 200:
            continue
        me, _, _ = model_error(d["res"][s], d["noise"][s])
        print(f"  {lo:.2f} - {hi:<8.2f}{s.sum():>9}{me:>12.2e}")


def bias_or_scatter(d):
    sec("2. bias or scatter?")
    print("   a systematic offset is a misspecification we can fix; pure "
          "scatter\n   means the features have run out")

    print(f"\n{'nu band':>12}{'bins':>9}{'mean (bias)':>14}{'RMS':>11}"
          f"{'bias/RMS':>11}")
    print("-" * 57)
    for lo, hi in [(0.2, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 3.0),
                   (3.0, 9.0)]:
        s = (d["nu"] >= lo) & (d["nu"] < hi)
        if s.sum() < 200:
            continue
        b = np.mean(d["res"][s])
        r = np.sqrt(np.mean(d["res"][s] ** 2))
        print(f"  {lo:.1f} - {hi:<5.1f}{s.sum():>9}{b:>+14.2e}{r:>11.2e}"
              f"{abs(b) / r:>11.2f}")


def concentration(d):
    sec("3. spread thinly, or carried by a few curves?")

    ids = np.unique(d["curve"])
    per = np.array([np.sqrt(np.mean(d["res"][d["curve"] == i] ** 2))
                    for i in ids])
    order = np.argsort(per)[::-1]

    total = np.sum(per ** 2)
    for frac in (0.05, 0.10, 0.25):
        n = max(1, int(frac * len(ids)))
        share = np.sum(per[order[:n]] ** 2) / total
        print(f"  worst {frac:.0%} of curves carry {share:.0%} of the squared "
              f"error")

    print(f"\n  per-curve RMS: median {np.median(per):.2e}, "
          f"90th {np.percentile(per, 90):.2e}, max {per.max():.2e}")

    # What do the worst curves have in common?
    worst = set(ids[order[:max(1, len(ids) // 10)]])
    is_worst = np.isin(d["curve"], list(worst))

    print(f"\n  {'quantity':>14}{'worst 10%':>12}{'rest':>10}{'shift':>10}")
    print("  " + "-" * 44)
    for name in ("alpha", "beta", "gamma", "perturbed", "z", "omch2", "nu",
                 "gamma2", "y", "alive"):
        a = np.median(d[name][is_worst])
        b = np.median(d[name][~is_worst])
        sd = np.std(d[name])
        print(f"  {name:>14}{a:>12.3f}{b:>10.3f}{(a - b) / sd:>+10.2f}")
    print("\n   shift is in units of the quantity's own spread; |shift| > 0.5 "
          "is a real\n   difference, near zero means that quantity does not "
          "single them out")


def edges(d):
    sec("4. is it extrapolation? error against distance from the coverage edge")

    print(f"\n{'feature':>12}{'lowest 5%':>12}{'middle':>10}{'highest 5%':>12}")
    print("-" * 46)
    for name in ("nu", "gamma2", "y", "dlnnu_dlnS" if "dlnnu_dlnS" in d
                 else "y"):
        if name not in d:
            continue
        v = d[name]
        lo_c, hi_c = np.percentile(v, [5, 95])
        parts = []
        for s in (v < lo_c, (v >= lo_c) & (v <= hi_c), v > hi_c):
            me, _, _ = model_error(d["res"][s], d["noise"][s])
            parts.append(me)
        print(f"{name:>12}{parts[0]:>12.2e}{parts[1]:>10.2e}{parts[2]:>12.2e}")


def correction_size(d):
    sec("5. does the error simply track how much work the correction does?")
    print("   the correction reaches +67% at low nu and vanishes at high nu;\n"
          "   if the error is a roughly constant FRACTION of it, that is "
          "just\n   multiplicative and not a low-nu pathology at all")

    print(f"\n{'|ratio - 1|':>16}{'bins':>9}{'model err':>12}"
          f"{'err/(ratio-1)':>15}{'median nu':>11}")
    print("-" * 63)
    m = np.abs(d["ratio"] - 1.0)
    for lo, hi in [(0.0, 0.01), (0.01, 0.05), (0.05, 0.12), (0.12, 0.25),
                   (0.25, 0.45), (0.45, 2.0)]:
        s = (m >= lo) & (m < hi)
        if s.sum() < 200:
            continue
        me, _, _ = model_error(d["res"][s], d["noise"][s])
        mid = np.median(m[s])
        print(f"  {lo:.2f} - {hi:<8.2f}{s.sum():>9}{me:>12.2e}"
              f"{me / mid:>15.3f}{np.median(d['nu'][s]):>11.2f}")


def make_plot(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    def banded(x, xlabel, ax_, log=False, bins=24):
        e = np.linspace(*np.percentile(x, [1, 99]), bins + 1)
        cen, me = [], []
        for i in range(bins):
            s = (x >= e[i]) & (x < e[i + 1])
            if s.sum() > 100:
                v, _, _ = model_error(d["res"][s], d["noise"][s])
                cen.append(0.5 * (e[i] + e[i + 1]))
                me.append(v)
        ax_.plot(cen, me, "-o", ms=3, color="steelblue")
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel("noise-corrected model error")
        ax_.grid(alpha=0.3)
        if log:
            ax_.set_xscale("log")

    banded(d["nu"], r"$\nu$", ax[0])
    ax[0].set_title("error vs barrier height")

    banded(d["alive"], "surviving fraction", ax[1])
    ax[1].set_title("error vs survival (the selection hypothesis)")

    banded(np.abs(d["ratio"] - 1.0), "|correction - 1|", ax[2])
    ax[2].set_title("error vs size of the correction")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nplot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="../runs")
    ap.add_argument("--plot")
    args = ap.parse_args()

    curves = load_runs(args.runs)

    d = collect(curves)
    print(f"{len(curves)} curves, {d['res'].size} cross-validated bins")

    by_alive(d)
    bias_or_scatter(d)
    concentration(d)
    edges(d)
    correction_size(d)

    if args.plot:
        make_plot(d, args.plot)


if __name__ == "__main__":
    main()
