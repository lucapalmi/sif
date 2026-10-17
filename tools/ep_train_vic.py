#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
Trains the void-in-cloud factor K on top of the single-barrier emulator.

Every training curve was replayed with a collapse barrier and its own seed
(ep_vic_worker.py), so per bin K = counts_vic / counts is the fraction of its
voids whose walk was not inside a cloud at a larger radius, measured walk for
walk. The network learns ln K = -Lambda_cond exp(z), with Lambda_cond the
conditional collapse hazard of ep_model: K <= 1 by construction, and exactly 1
wherever the baseline sees no cloud.

Stages:
  features   one pass over runs + runs_vic: every candidate input, the
             baseline, the target and the single-barrier emulator's own
             prediction per bin, cached to one file
  fold k     cross-validation by whole cosmologies; held-out scores to a file
  merge      pool the folds and report
  fit        the final fit on everything -> ep_emulator_vic.npz

    python tools/ep_train_vic.py features --cache vic_cache.npz
    python tools/ep_train_vic.py fold 0 --cache vic_cache.npz -o f0.npz
    python tools/ep_train_vic.py merge f0.npz f1.npz f2.npz f3.npz
    python tools/ep_train_vic.py fit --cache vic_cache.npz -o ep_emulator_vic.npz
"""

import argparse
import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ep_emulator import EmulatorMLP  # noqa: E402
from ep_model import (EPEmulator, VICEmulator,  # noqa: E402
                      multiplicity_from_hazard, vic_candidates)

HIDDEN = (48, 48)
MAXITER = 6000
N_FOLDS = 4
SEED = 11
R_MAX = 80.0
MAX_ORIGIN = 300.0
MIN_VOIDS = 100        # bins with fewer single-barrier voids carry no K

FEATURES = ["nu", "nu_c", "gamma2", "log_lam_cond", "nu_c_back", "lag",
            "gamma2_back", "nu_back", "log_cum_lam_c", "dlnnu_dlnS",
            "dlnnuc_dlnS", "cum_lam", "nu_origin", "y", "y_c"]

CAMPAIGNS = {1: ("runs", "ep_spectra.npz"), 2: ("runs_v2", "ep_spectra_v2.npz"),
             3: ("runs_v3", "ep_spectra_v2.npz"),
             4: ("runs_v4", "ep_spectra_v2.npz")}


# --- features ------------------------------------------------------------------

def stage_features(root, cache, weights):
    import pysif.model as model
    from ep_trainset_load import load_runs

    emu = EPEmulator.load(weights)
    cols = {}
    n_curves = 0
    for camp, (rdir, spec) in CAMPAIGNS.items():
        sp = np.load(os.path.join(HERE, spec))
        k = np.ascontiguousarray(sp["k"])
        curves = load_runs([os.path.join(root, rdir)], max_origin=MAX_ORIGIN)
        vic_files = {}
        for c in curves:
            path = os.path.join(root, "runs_vic", f"c{camp}",
                                f"vic_{c['task']:03d}.npz")
            if path not in vic_files:
                vic_files[path] = (np.load(path) if os.path.exists(path)
                                   else None)
            v = vic_files[path]
            t = c["barrier_tag"]
            if v is None or f"{t}__counts_vic" not in v.files:
                continue
            radii = c["radii"].astype(np.float32)
            cov, sigma, _, dv = model.delta_covariance_pk(
                k, np.ascontiguousarray(sp["pk"][c["task"]]), radii)
            S = c["S_full"].astype(float)
            collapse = v[f"{t}__collapse"].astype(float)
            cand, lam_cond = vic_candidates(
                c["radii"], S, c["barrier"], collapse,
                c["deriv_variance"], cov)

            one = c["counts"][:-1].astype(float)
            vic = v[f"{t}__counts_vic"][:-1].astype(float)
            ok = c["mask"] & (one >= MIN_VOIDS)
            if not ok.any():
                continue
            K = vic[ok] / one[ok]

            corr = emu.correction(c["X"])
            f_one = multiplicity_from_hazard(c["lam_up"] * np.exp(corr),
                                             c["radii"], c["nu_large"])
            dr = np.diff(c["radii"].astype(float))
            n = float(c["n_paths"])

            row = {name: arr[ok] for name, arr in cand.items()}
            extra = dict(
                lam_cond=lam_cond[ok],
                lnk=np.log(np.maximum(K, 1e-300)),
                sig=np.sqrt(np.maximum(1.0 - K, 1.0 / one[ok])
                            / (np.maximum(K, 1e-12) * one[ok])),
                f_one=f_one[ok], f_vic_mc=vic[ok] / (n * dr[ok]),
                n_vic=vic[ok], r=0.5 * (c["radii"][1:] + c["radii"][:-1])[ok],
                cosmo=np.full(ok.sum(), c["task"]),
                campaign=np.full(ok.sum(), camp),
                curve=np.full(ok.sum(), n_curves),
                origin=np.full(ok.sum(), float(c.get("r_origin") or
                                               c["radii"][-1])),
                delta_c=np.full(ok.sum(), float(v[f"{t}__delta_c"])),
                beta_e=np.full(ok.sum(), float(v[f"{t}__beta_e"])))
            # The bookkeeping must never share a name with an input: a target
            # stored as "y" once overwrote the barrier's y, and the network was
            # fitted with the answer among its inputs.
            clash = set(extra) & set(row)
            if clash:
                raise SystemExit(f"cache columns {sorted(clash)} are also "
                                 f"input candidates")
            row.update(extra)
            for key, arr in row.items():
                cols.setdefault(key, []).append(np.asarray(arr))
            n_curves += 1
        print(f"campaign {camp}: {n_curves} curves so far", flush=True)

    out = {key: np.concatenate(arrs) for key, arrs in cols.items()}
    np.savez(cache, **out)
    print(f"{n_curves} curves, {len(out['y'])} bins -> {cache}")


# --- fitting and scoring ------------------------------------------------------

def load_cache(path, features):
    d = np.load(path)
    X = np.column_stack([d[f] for f in features])
    return d, X


def fit(X, y, lam, hidden, maxiter, seed=SEED):
    # Target and baseline scaled together so the data term is O(1): the
    # prediction -lam e^z is unchanged by it, and predict() takes the unscaled
    # baseline. ln K is ~1e-3 in a typical bin and zero in most, so the plain
    # mean square is ~1e-6 against an L2 term of ~1e-3: the fit minimised the
    # penalty and stopped within ~140 iterations, the data barely heard.
    c = 1.0 / np.sqrt(max(np.mean(y * y), 1e-300))
    m = EmulatorMLP(hidden=hidden, seed=seed, link="negexp").fit(
        X, c * y, np.ones(len(y)), maxiter=maxiter, scale=c * lam)
    print(f"fit: {X.shape[1]} inputs, {len(y)} rows, {m.nit_} iterations, "
          f"loss {m.loss_:.4e} ({m.message_})", flush=True)
    return m


def scores(d, idx, pred):
    """Per held-out bin: ln K residual, and the full f x K error."""
    f_pred = d["f_one"][idx] * np.exp(pred)
    return dict(dlnk=pred - d["lnk"][idx], sig=d["sig"][idx], lnk=d["lnk"][idx],
                rel=f_pred / d["f_vic_mc"][idx] - 1.0,
                rel_one=d["f_one"][idx] * np.exp(d["lnk"][idx])
                / d["f_vic_mc"][idx] - 1.0,
                rel_novic=d["f_one"][idx] / d["f_vic_mc"][idx] - 1.0,
                noise=1.0 / np.sqrt(np.maximum(d["n_vic"][idx], 1.0)),
                r=d["r"][idx], nu=d["nu"][idx], origin=d["origin"][idx],
                campaign=d["campaign"][idx], cosmo=d["cosmo"][idx])


def fold_rows(d, k):
    cosmos = np.unique(d["cosmo"])
    held = set(np.random.default_rng(SEED).permutation(cosmos)[k::N_FOLDS])
    test = np.isin(d["cosmo"], list(held))
    return ~test, test


def report(s, rmax):
    def line(label, m):
        if not m.any():
            return
        a = np.abs(s["dlnk"][m])
        big = m & (-s["lnk"] > 1e-3)
        me = (np.sqrt(max(np.mean(s["dlnk"][big] ** 2)
                          - np.mean(s["sig"][big] ** 2), 0.0))
              if big.any() else 0.0)
        print(f"  {label:<26}{m.sum():>8}  |dlnK| med {np.median(a):.2e} "
              f"95th {np.percentile(a, 95):.2e} | K<0.999: {big.sum():>6} "
              f"med {np.median(np.abs(s['dlnk'][big])) if big.any() else 0:.2e}"
              f" model {me:.2e} | f: med {np.median(np.abs(s['rel'][m])):.3%} "
              f"(K exact {np.median(np.abs(s['rel_one'][m])):.3%}, no VIC "
              f"{np.median(np.abs(s['rel_novic'][m])):.3%}) noise "
              f"{np.median(s['noise'][m]):.3%}")
    r = s["r"]
    print("held-out scores (whole cosmologies):")
    line("all", np.ones(len(r), bool))
    line(f"R < {rmax:g}", r < rmax)
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 99)):
        line(f"R < {rmax:g}, nu {lo}-{hi}", (r < rmax) & (s["nu"] >= lo)
             & (s["nu"] < hi))
    vicish = (r < rmax) & (-s["lnk"] > 1e-3)
    line(f"R < {rmax:g}, K < 0.999", vicish)
    line(f"R < {rmax:g}, K < 0.99", (r < rmax) & (-s["lnk"] > 1e-2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("features", "fold", "merge", "fit"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--root", default=os.path.join(HERE, ".."))
    ap.add_argument("--cache", default="vic_cache.npz")
    ap.add_argument("--weights", default=os.path.join(HERE, "ep_emulator.npz"))
    ap.add_argument("--features", default=",".join(FEATURES))
    ap.add_argument("--hidden", default=",".join(map(str, HIDDEN)))
    ap.add_argument("--maxiter", type=int, default=MAXITER)
    ap.add_argument("--rmax", type=float, default=R_MAX)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    features = a.features.split(",")
    hidden = tuple(int(h) for h in a.hidden.split(","))

    if a.stage == "features":
        stage_features(a.root, a.cache, a.weights)
        return

    if a.stage == "merge":
        parts = [dict(np.load(p)) for p in a.args]
        s = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        report(s, a.rmax)
        if a.out:
            # Recorded in the weights, as ep_train_final does: the median
            # error of f x K below rmax where void-in-cloud matters (K < 0.999)
            # and where it is strong (K < 0.99), held out by cosmology.
            emu = VICEmulator.load(a.out)
            r = s["r"] < a.rmax
            for key, cut in (("median", 1e-3), ("median_strong", 1e-2)):
                m = r & (-s["lnk"] > cut)
                emu.accuracy[key] = float(np.median(np.abs(s["rel"][m])))
            emu.accuracy["median_all"] = float(np.median(np.abs(s["rel"][r])))
            emu.save(a.out)
            print(f"  CV accuracy recorded in {a.out}: {emu.accuracy}")
        return

    if "lnk" in features or "lam_cond" in features:
        raise SystemExit("features: the target and the baseline are not inputs")
    d, X = load_cache(a.cache, features)
    if a.stage == "fold":
        k = int(a.args[0])
        train, test = fold_rows(d, k)
        m = fit(X[train], d["lnk"][train], d["lam_cond"][train], hidden,
                a.maxiter)
        pred = m.predict(X[test], d["lam_cond"][test])
        s = scores(d, np.flatnonzero(test), pred)
        np.savez(a.out or f"vic_fold{k}.npz", **s)
        report(s, a.rmax)
        return

    m = fit(X, d["lnk"], d["lam_cond"], hidden, a.maxiter)
    mu, sd, layers = m.weights()
    box = (X.min(axis=0), X.max(axis=0))
    VICEmulator(features, mu, sd, layers, box).save(
        a.out or "ep_emulator_vic.npz")
    s = scores(d, np.arange(len(d["lnk"])), m.predict(X, d["lam_cond"]))
    print("in-sample:")
    report(s, a.rmax)


if __name__ == "__main__":
    main()
