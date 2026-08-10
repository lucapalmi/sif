#!/usr/bin/env python
"""
Reads the cluster array's output back into training curves.

The worker stores only what cannot be recomputed -- the raw counts, and the
inputs they were produced from. Everything the fit consumes is derived here
through ep_model, which is also what the deployed emulator calls, so the
training features and the inference features cannot drift apart. A change to a
feature definition means re-running this, not re-running the cluster.

Usage:  python tools/ep_trainset_load.py runs/
"""

import glob
import os
import sys

import numpy as np

from ep_model import (FEATURE_NAMES, compute_features, mc_hazard, upper_tail)

# Bins noisier than this carry less information than the correction we are
# trying to resolve, and are excluded from fitting and scoring alike.
MAX_REL_ERR = 0.05

# Curves whose walk origin is low enough that this fraction of paths start
# above the barrier. Measured to be predicted badly (0.8% typical against
# 0.13%), and excluded; the same threshold becomes the deployed domain guard.
MAX_FIRST_STEP = 0.01


def load_runs(run_dir, max_rel_err=MAX_REL_ERR, max_first_step=MAX_FIRST_STEP,
              keep_truncated=False):
    """Every curve in every task file, with canonical features and targets.

    Each curve carries:
      X          (n-1, 8) canonical feature matrix, columns FEATURE_NAMES
      lam_up     baseline hazard per bin
      lam_mc     Monte Carlo hazard per bin (the target's numerator)
      ratio      lam_mc / lam_up, the quantity the network learns the log of
      rel_err    Monte Carlo relative error on lam_mc
      mask       bins usable for fitting and scoring
      plus the raw inputs, the barrier parameters and the cosmology.
    """
    curves = []

    for path in sorted(glob.glob(os.path.join(run_dir, "trainset_*.npz"))):
        d = np.load(path)
        n_paths = int(d["n_paths"])
        task = int(os.path.basename(path).split("_")[1].split(".")[0])
        cosmo = {k.split("__", 1)[1]: float(d[k]) for k in d.files
                 if k.startswith("cosmo__")}

        tags = sorted({k.split("__")[0] for k in d.files
                       if k.startswith("b") and "__" in k
                       and k.split("__")[0][1:].isdigit()})

        for tag in tags:
            g = lambda key: d[f"{tag}__{key}"]
            radii = g("radii").astype(float)
            S = g("S").astype(float)
            barrier = g("barrier").astype(float)
            dvar = g("deriv_variance").astype(float)
            counts = g("counts").astype(float)

            X, lam_up, nu_large = compute_features(radii, S, barrier, dvar)
            lam_mc, rel_err, alive = mc_hazard(counts, n_paths)

            first_step = float(upper_tail(nu_large))
            if first_step > max_first_step and not keep_truncated:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(lam_up > 0.0, lam_mc / lam_up, np.nan)

            mask = (np.isfinite(ratio) & (ratio > 0.0)
                    & (rel_err < max_rel_err) & (lam_up > 0.0)
                    & np.all(np.isfinite(X), axis=1))

            c = dict(task=task, barrier_tag=tag, cosmo=task, family="cdm",
                     radii=radii, S_full=S, barrier=barrier,
                     deriv_variance=dvar, counts=counts, n_paths=n_paths,
                     X=X, lam_up=lam_up, lam_mc=lam_mc, ratio=ratio,
                     rel_err=rel_err, alive=alive, mask=mask,
                     nu_large=nu_large, first_step=first_step,
                     alpha=float(g("alpha")), beta=float(g("beta")),
                     gamma=float(g("gamma")),
                     perturbed=bool(float(g("perturb_seed")) != 0),
                     **cosmo)

            # Named views on the feature columns, for slicing and plotting.
            for j, name in enumerate(FEATURE_NAMES):
                c[name] = X[:, j]

            curves.append(c)

    return curves


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    curves = load_runs(run_dir)
    if not curves:
        sys.exit(f"no trainset_*.npz under {run_dir}")

    n_bins = sum(int(c["mask"].sum()) for c in curves)
    pool = lambda k: np.concatenate([c[k][c["mask"]] for c in curves])
    ratios, errs = pool("ratio"), pool("rel_err")

    print(f"{len(curves)} curves from {len({c['task'] for c in curves})} "
          f"cosmologies, {n_bins} usable bins")
    print(f"  paths per curve : {curves[0]['n_paths']:.3e}")
    print(f"  perturbed       : {sum(c['perturbed'] for c in curves)} curves")
    print(f"  hazard ratio    : {ratios.min():.4f} - {ratios.max():.4f} "
          f"(median |r-1| {np.median(np.abs(ratios - 1)):.4f})")
    print(f"  noise floor     : median {np.median(errs):.2e}, "
          f"RMS {np.sqrt(np.mean(errs ** 2)):.2e}")
    print("  feature ranges:")
    for name in FEATURE_NAMES:
        v = pool(name)
        print(f"    {name:<12}{v.min():>12.5f}  to {v.max():>12.5f}")


if __name__ == "__main__":
    main()
