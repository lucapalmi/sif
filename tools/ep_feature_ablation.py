#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
Feature ablation for the EP hazard-ratio correction.

Question under test: which features does the correction need, and does the
error saturate once local quantities are supplied -- or does first crossing
retain history dependence that no local description can capture?

Reads the pilot ensemble written by ep_ratio_pilot.py and fits nested feature
sets with a small MLP, held out AT THE CURVE LEVEL: whole spectra and whole
barrier families are removed from training, so nothing can be recovered by
interpolating between neighbouring bins of the same run. Point-level splits
would flatter every feature set equally and say nothing.

Scored on the quantity that matters -- the multiplicity rebuilt through the
survival recursion from the corrected hazards -- not only on the log-hazard
residual the fit minimizes.

Usage:  python tools/ep_feature_ablation.py [pilot_dir]
"""

import os
import sys
import warnings

import numpy as np
from scipy.special import erfc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

SPECTRA = ["pl-2.5", "pl-2.0", "pl-1.5", "pl-1.0", "eh0.14", "eh0.20"]
BARRIERS = ["flat", "smt_mild", "smt_steep", "smt_heavy"]

# Bins noisier than this are dropped: their target carries less information
# than the correction we are trying to resolve.
MAX_REL_ERR = 0.05

FEATURE_SETS = {
    "nu": ["nu"],
    "nu+g2": ["nu", "gamma2"],
    "nu+g2+y": ["nu", "gamma2", "y"],
    "nu+g2+y+dnu": ["nu", "gamma2", "y", "dlnnu_dlnS"],
    "nu+g2+y+dnu+hist": ["nu", "gamma2", "y", "dlnnu_dlnS", "nu_back", "lag"],
}


def upper_tail(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def load_curves(path):
    """One record per (spectrum, barrier), with derived features attached."""
    d = np.load(path)
    curves = []

    for s in SPECTRA:
        for b in BARRIERS:
            g = lambda key: d[f"{s}__{b}__{key}"]
            S_full = g("S")
            nu = g("nu")          # bin centres already
            gamma2 = g("gamma2")
            y = g("y")
            radii = g("radii")
            counts = g("counts")

            # S at the bin centres, to differentiate the bin-centred nu on
            S_mid = 0.5 * (S_full[:-1] + S_full[1:])

            # Local logarithmic slope of nu in the walk's own time. y already
            # carries dB/dS, but normalized by the slope scatter; this is the
            # bare shape, and the two are not the same number.
            dlnnu_dlnS = np.gradient(np.log(nu), np.log(S_mid), edge_order=2)

            # History: nu half an e-fold back in S, clamped at the walk's
            # origin, and how far back the lookback actually reached.
            S_origin = S_full[-1]
            target = np.maximum(0.5 * S_mid, S_origin)
            lag = np.log(S_mid / target)
            # S_mid descends with index, so reverse for np.interp
            nu_back = np.interp(target, S_mid[::-1], nu[::-1])

            curves.append(dict(
                spectrum=s, barrier=b, radii=radii, S_full=S_full,
                counts=counts, nu=nu, gamma2=gamma2, y=y,
                dlnnu_dlnS=dlnnu_dlnS, nu_back=nu_back, lag=lag,
                lam_up=g("lam_up"), lam_mc=g("lam_mc"),
                ratio=g("ratio"), rel_err=g("rel_err"),
            ))
    return curves


def curve_matrix(curve, feats, mask):
    return np.column_stack([curve[f][mask] for f in feats])


def clean_mask(curve):
    return (np.isfinite(curve["ratio"]) & (curve["ratio"] > 0)
            & (curve["rel_err"] < MAX_REL_ERR) & (curve["lam_up"] > 0))


def multiplicity_from_hazard(lam, radii, nu_origin):
    """f on the n-1 bins from per-bin integrated hazards, via survival.

    The walk enters the largest-radius bin first. Paths that start above the
    barrier cross on the first step and never enter a bin: that point mass is
    the one-point tail at the walk's origin, which is exactly analytic.
    """
    n_bins = len(lam)
    f = np.zeros(n_bins)
    alive = 1.0 - upper_tail(nu_origin)
    for i in range(n_bins - 1, -1, -1):
        p = alive * (1.0 - np.exp(-lam[i]))
        alive -= p
        dr = radii[i + 1] - radii[i]
        f[i] = p / dr if dr > 0 else 0.0
    return f


def fit_predict(train_curves, test_curves, feats, seed=0):
    """Fit log(ratio) on the training curves, predict on the test ones."""
    Xtr, ytr = [], []
    for c in train_curves:
        m = clean_mask(c)
        Xtr.append(curve_matrix(c, feats, m))
        ytr.append(np.log(c["ratio"][m]))
    Xtr = np.vstack(Xtr)
    ytr = np.concatenate(ytr)

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh",
                     solver="lbfgs", max_iter=4000, tol=1e-9,
                     alpha=1e-5, random_state=seed))
    model.fit(Xtr, ytr)

    out = []
    for c in test_curves:
        m = clean_mask(c)
        pred_log = model.predict(curve_matrix(c, feats, m))
        out.append((c, m, pred_log))
    return out


def score(predictions):
    """Residuals in log-hazard, and relative error on the multiplicity."""
    log_res, rel_f, nus, noise = [], [], [], []

    for c, m, pred_log in predictions:
        truth_log = np.log(c["ratio"][m])
        log_res.append(pred_log - truth_log)
        nus.append(c["nu"][m])
        noise.append(c["rel_err"][m])

        # Rebuild the multiplicity from corrected hazards, against the
        # multiplicity the Monte Carlo itself gives on the same bins.
        lam_corr = c["lam_up"].copy()
        lam_corr[m] *= np.exp(pred_log)
        nu_origin = c["nu"][-1]
        f_pred = multiplicity_from_hazard(lam_corr, c["radii"], nu_origin)
        f_mc = multiplicity_from_hazard(c["lam_mc"], c["radii"], nu_origin)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(f_mc > 0, (f_pred - f_mc) / f_mc, np.nan)
        rel_f.append(rel[m])

    return (np.concatenate(log_res), np.concatenate(rel_f),
            np.concatenate(nus), np.concatenate(noise))


def run_ablation(curves, group_key, group_values, label):
    print(f"\n{'=' * 76}\nleave-one-{label}-out\n{'=' * 76}")
    header = (f"{'features':<22} {'RMS log-haz':>12} {'med |df/f|':>11} "
              f"{'95th |df/f|':>12} {'med |df/f| nu>2':>16}")
    print(header)
    print("-" * len(header))

    results = {}
    for name, feats in FEATURE_SETS.items():
        all_log, all_relf, all_nu, all_noise = [], [], [], []

        for held in group_values:
            train = [c for c in curves if c[group_key] != held]
            test = [c for c in curves if c[group_key] == held]
            lr, rf, nu, ns = score(fit_predict(train, test, feats))
            all_log.append(lr)
            all_relf.append(rf)
            all_nu.append(nu)
            all_noise.append(ns)

        lr = np.concatenate(all_log)
        rf = np.abs(np.concatenate(all_relf))
        nu = np.concatenate(all_nu)
        ns = np.concatenate(all_noise)
        rf = rf[np.isfinite(rf)]

        hi = nu > 2.0
        med_hi = np.nanmedian(np.abs(np.concatenate(all_relf))[hi]) if hi.any() \
            else np.nan

        print(f"{name:<22} {np.sqrt(np.mean(lr ** 2)):>12.4f} "
              f"{np.median(rf):>11.4f} {np.percentile(rf, 95):>12.4f} "
              f"{med_hi:>16.4f}")

        results[name] = dict(log_res=lr, rel_f=np.concatenate(all_relf),
                             nu=nu, noise=ns)

    ns = results[list(FEATURE_SETS)[0]]["noise"]
    print(f"\n  Monte Carlo noise floor on the target: median {np.median(ns):.4f}, "
          f"RMS {np.sqrt(np.mean(ns ** 2)):.4f}")
    return results


def per_spectrum_breakdown(curves, feats):
    print(f"\n{'=' * 76}\nper-spectrum, held out entirely ({'+'.join(feats)})"
          f"\n{'=' * 76}")
    print(f"{'held-out spectrum':<20} {'RMS log-haz':>12} {'med |df/f|':>12} "
          f"{'95th |df/f|':>12}")
    print("-" * 60)
    for held in SPECTRA:
        train = [c for c in curves if c["spectrum"] != held]
        test = [c for c in curves if c["spectrum"] == held]
        lr, rf, nu, ns = score(fit_predict(train, test, feats))
        rf = np.abs(rf[np.isfinite(rf)])
        print(f"{held:<20} {np.sqrt(np.mean(lr ** 2)):>12.4f} "
              f"{np.median(rf):>12.4f} {np.percentile(rf, 95):>12.4f}")


def make_plots(res_spec, curves, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # 1. error saturation
    names = list(FEATURE_SETS)
    rms = [np.sqrt(np.mean(res_spec[n]["log_res"] ** 2)) for n in names]
    floor = np.sqrt(np.mean(res_spec[names[0]]["noise"] ** 2))
    axes[0].bar(range(len(names)), rms, color="steelblue")
    axes[0].axhline(floor, color="crimson", ls="--", lw=1.2,
                    label=f"MC noise floor ({floor:.3f})")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("RMS residual in log hazard")
    axes[0].set_title("error saturation (leave-one-spectrum-out)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    # 2. residual vs nu for the local set
    best = "nu+g2+y"
    r = res_spec[best]
    axes[1].scatter(r["nu"], r["log_res"], s=4, alpha=0.4, color="steelblue")
    axes[1].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[1].fill_between([0, r["nu"].max()], -floor, floor, color="crimson",
                         alpha=0.15, label="MC noise floor")
    axes[1].set_xlabel(r"$\nu$")
    axes[1].set_ylabel("log-hazard residual")
    axes[1].set_title(f"where the error lives ({best})")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3. multiplicity error vs nu, local vs full
    for name, color in [("nu+g2", "darkorange"), ("nu+g2+y", "steelblue"),
                        ("nu+g2+y+dnu+hist", "seagreen")]:
        rr = res_spec[name]
        order = np.argsort(rr["nu"])
        nu_s = rr["nu"][order]
        rel_s = np.abs(rr["rel_f"][order])
        # running median in nu
        nbin = 25
        edges = np.linspace(nu_s.min(), nu_s.max(), nbin + 1)
        cen, med = [], []
        for i in range(nbin):
            m = (nu_s >= edges[i]) & (nu_s < edges[i + 1])
            if m.sum() > 5:
                cen.append(0.5 * (edges[i] + edges[i + 1]))
                med.append(np.nanmedian(rel_s[m]))
        axes[2].semilogy(cen, med, "-o", ms=3, color=color, label=name)
    axes[2].set_xlabel(r"$\nu$")
    axes[2].set_ylabel(r"median $|\Delta f / f|$")
    axes[2].set_title("multiplicity error vs barrier height")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(out_dir, "ablation_summary.png")
    fig.savefig(path, dpi=150)
    print(f"\nplot written to {path}")


def main():
    pilot_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    curves = load_curves(os.path.join(pilot_dir, "ep_pilot_results.npz"))

    n_bins = sum(clean_mask(c).sum() for c in curves)
    print(f"{len(curves)} curves, {n_bins} bins with < {MAX_REL_ERR:.0%} "
          f"Monte Carlo error")

    res_spec = run_ablation(curves, "spectrum", SPECTRA, "spectrum")
    run_ablation(curves, "barrier", BARRIERS, "barrier")
    per_spectrum_breakdown(curves, FEATURE_SETS["nu+g2+y"])

    make_plots(res_spec, curves, pilot_dir)


if __name__ == "__main__":
    main()
