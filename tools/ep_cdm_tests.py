#!/usr/bin/env python
"""
Three tests on the CDM-only ensemble, all of which change what the training
set should look like.

A. Is gamma^2 needed at all? Inside the CDM box gamma^2 spans only
   [0.18, 0.25], so the correction may need it only as a weak modulation --
   or not at all, which would simplify both the model and its domain guard.

B. Does honest validation still look good? On a manifold this thin, holding
   out one scattered cosmology measures interpolation between near-duplicates.
   This compares that against holding out CONTIGUOUS REGIONS of the box.

C. Does the fit degrade gracefully off the manifold, or fall off a cliff?
   Trained on CDM alone, evaluated against the power-law spectra of the first
   pilot, which sit far below the CDM gamma^2 band. Never trained on: purely a
   diagnostic of what the domain guard has to catch.

Usage:  python tools/ep_cdm_tests.py [pilot_dir]
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

MAX_REL_ERR = 0.05

FEATURE_SETS = {
    "nu": ["nu"],
    "nu+y": ["nu", "y"],
    "nu+g2": ["nu", "gamma2"],
    "nu+g2+y": ["nu", "gamma2", "y"],
    "nu+g2+y+hist": ["nu", "gamma2", "y", "nu_back", "lag"],
}


def upper_tail(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def derive(curve):
    """Attach the derived features every set below draws from."""
    S_full = curve["S_full"]
    nu = curve["nu"]
    S_mid = 0.5 * (S_full[:-1] + S_full[1:])
    S_origin = S_full[-1]
    target = np.maximum(0.5 * S_mid, S_origin)
    curve["lag"] = np.log(S_mid / target)
    curve["nu_back"] = np.interp(target, S_mid[::-1], nu[::-1])
    return curve


def load_cdm(path):
    d = np.load(path)
    barriers = [str(b) for b in d["barrier_names"]]
    n_cosmo = len(d["param__index"])
    params = {k.split("__", 1)[1]: d[k] for k in d.files
              if k.startswith("param__")}

    curves = []
    for i in range(n_cosmo):
        for b in barriers:
            tag = f"c{i:02d}__{b}"
            c = dict(cosmo=i, barrier=b, family="cdm",
                     radii=d[f"{tag}__radii"], S_full=d[f"{tag}__S"],
                     nu=d[f"{tag}__nu"], gamma2=d[f"{tag}__gamma2"],
                     y=d[f"{tag}__y"], lam_up=d[f"{tag}__lam_up"],
                     lam_mc=d[f"{tag}__lam_mc"], ratio=d[f"{tag}__ratio"],
                     rel_err=d[f"{tag}__rel_err"])
            for k, v in params.items():
                c[k] = v[i]
            curves.append(derive(c))
    return curves


def load_first_pilot(path):
    """The spectral-family pilot, for the off-manifold diagnostic."""
    d = np.load(path)
    spectra = ["pl-2.5", "pl-2.0", "pl-1.5", "pl-1.0", "eh0.14", "eh0.20"]
    barriers = ["flat", "smt_mild", "smt_steep", "smt_heavy"]
    curves = []
    for s in spectra:
        for b in barriers:
            tag = f"{s}__{b}"
            c = dict(cosmo=-1, barrier=b, family=s,
                     radii=d[f"{tag}__radii"], S_full=d[f"{tag}__S"],
                     nu=d[f"{tag}__nu"], gamma2=d[f"{tag}__gamma2"],
                     y=d[f"{tag}__y"], lam_up=d[f"{tag}__lam_up"],
                     lam_mc=d[f"{tag}__lam_mc"], ratio=d[f"{tag}__ratio"],
                     rel_err=d[f"{tag}__rel_err"])
            curves.append(derive(c))
    return curves


def clean_mask(c):
    return (np.isfinite(c["ratio"]) & (c["ratio"] > 0)
            & (c["rel_err"] < MAX_REL_ERR) & (c["lam_up"] > 0))


def multiplicity_from_hazard(lam, radii, nu_origin):
    f = np.zeros(len(lam))
    alive = 1.0 - upper_tail(nu_origin)
    for i in range(len(lam) - 1, -1, -1):
        p = alive * (1.0 - np.exp(-lam[i]))
        alive -= p
        dr = radii[i + 1] - radii[i]
        f[i] = p / dr if dr > 0 else 0.0
    return f


def fit(train, feats, seed=0):
    X = np.vstack([np.column_stack([c[f][clean_mask(c)] for f in feats])
                   for c in train])
    y = np.concatenate([np.log(c["ratio"][clean_mask(c)]) for c in train])
    m = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh",
                     solver="lbfgs", max_iter=4000, tol=1e-9, alpha=1e-5,
                     random_state=seed))
    m.fit(X, y)
    return m


def evaluate(m, test, feats):
    log_res, rel_f, nus, g2s, noise = [], [], [], [], []
    for c in test:
        k = clean_mask(c)
        if k.sum() == 0:
            continue
        pred = m.predict(np.column_stack([c[f][k] for f in feats]))
        log_res.append(pred - np.log(c["ratio"][k]))
        nus.append(c["nu"][k])
        g2s.append(c["gamma2"][k])
        noise.append(c["rel_err"][k])

        lam = c["lam_up"].copy()
        lam[k] *= np.exp(pred)
        nu_o = c["nu"][-1]
        f_pred = multiplicity_from_hazard(lam, c["radii"], nu_o)
        f_mc = multiplicity_from_hazard(c["lam_mc"], c["radii"], nu_o)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_f.append(np.where(f_mc > 0, (f_pred - f_mc) / f_mc, np.nan)[k])
    return tuple(np.concatenate(a) for a in
                 (log_res, rel_f, nus, g2s, noise))


def summarize(log_res, rel_f, label, width=26):
    rf = np.abs(rel_f[np.isfinite(rel_f)])
    return (f"{label:<{width}} {np.sqrt(np.mean(log_res ** 2)):>11.4f} "
            f"{np.median(rf):>11.4f} {np.percentile(rf, 95):>12.4f}")


def test_a(curves):
    print(f"\n{'=' * 78}\nA. does the correction need gamma^2 inside the CDM "
          f"box?\n{'=' * 78}")
    print("   held out by barrier family, so gamma^2 coverage is never the "
          "limitation")
    print(f"{'features':<26} {'RMS log-haz':>11} {'med |df/f|':>11} "
          f"{'95th |df/f|':>12}")
    print("-" * 62)

    barriers = sorted({c["barrier"] for c in curves})
    for name, feats in FEATURE_SETS.items():
        L, R = [], []
        for held in barriers:
            tr = [c for c in curves if c["barrier"] != held]
            te = [c for c in curves if c["barrier"] == held]
            lr, rf, *_ = evaluate(fit(tr, feats), te, feats)
            L.append(lr)
            R.append(rf)
        print(summarize(np.concatenate(L), np.concatenate(R), name))

    ns = np.concatenate([c["rel_err"][clean_mask(c)] for c in curves])
    print(f"\n  Monte Carlo noise floor: median {np.median(ns):.4f}, "
          f"RMS {np.sqrt(np.mean(ns ** 2)):.4f}")


def test_b(curves):
    print(f"\n{'=' * 78}\nB. scattered hold-out vs contiguous-region "
          f"hold-out\n{'=' * 78}")
    feats = FEATURE_SETS["nu+g2+y"]

    print(f"{'validation scheme':<26} {'RMS log-haz':>11} {'med |df/f|':>11} "
          f"{'95th |df/f|':>12}")
    print("-" * 62)

    # scattered: one cosmology at a time
    cosmos = sorted({c["cosmo"] for c in curves})
    L, R = [], []
    for held in cosmos:
        tr = [c for c in curves if c["cosmo"] != held]
        te = [c for c in curves if c["cosmo"] == held]
        lr, rf, *_ = evaluate(fit(tr, feats), te, feats)
        L.append(lr)
        R.append(rf)
    print(summarize(np.concatenate(L), np.concatenate(R),
                    "one cosmology at a time"))

    # contiguous regions of the box
    regions = {
        "high mnu (> 0.25 eV)": lambda c: c["mnu"] > 0.25,
        "high z (> 1.0)": lambda c: c["z"] > 1.0,
        "low omch2 (< 0.105)": lambda c: c["omch2"] < 0.105,
        "high omch2 (> 0.14)": lambda c: c["omch2"] > 0.14,
        "low ns (< 0.94)": lambda c: c["ns"] < 0.94,
    }
    for label, sel in regions.items():
        te = [c for c in curves if sel(c)]
        tr = [c for c in curves if not sel(c)]
        n_held = len({c["cosmo"] for c in te})
        if n_held == 0 or len(tr) == 0:
            print(f"{label:<26}  (empty)")
            continue
        lr, rf, *_ = evaluate(fit(tr, feats), te, feats)
        print(summarize(lr, rf, f"{label} [{n_held}]"))


def test_c(cdm_curves, pilot_curves):
    print(f"\n{'=' * 78}\nC. off-manifold behaviour: trained on CDM only"
          f"\n{'=' * 78}")
    feats = FEATURE_SETS["nu+g2+y"]
    m = fit(cdm_curves, feats)

    g2_train = np.concatenate([c["gamma2"][clean_mask(c)]
                               for c in cdm_curves])
    lo, hi = g2_train.min(), g2_train.max()
    print(f"   training gamma^2 range: [{lo:.4f}, {hi:.4f}]")
    print(f"\n{'evaluated on':<26} {'RMS log-haz':>11} {'med |df/f|':>11} "
          f"{'95th |df/f|':>12}  {'gamma^2 range':>16} {'outside':>8}")
    print("-" * 92)

    families = sorted({c["family"] for c in pilot_curves})
    for fam in families:
        te = [c for c in pilot_curves if c["family"] == fam]
        lr, rf, nus, g2s, ns = evaluate(m, te, feats)
        frac_out = np.mean((g2s < lo) | (g2s > hi))
        row = summarize(lr, rf, fam)
        print(f"{row}  {g2s.min():>7.4f}-{g2s.max():<7.4f} "
              f"{100 * frac_out:>7.1f}%")

    # and on its own training domain, for reference
    lr, rf, *_ = evaluate(m, cdm_curves, feats)
    print(f"\n{summarize(lr, rf, 'CDM itself (in-sample)')}")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    cdm = load_cdm(os.path.join(d, "ep_cdm_pilot.npz"))
    n_bins = sum(clean_mask(c).sum() for c in cdm)
    print(f"{len(cdm)} CDM curves, {n_bins} bins with < {MAX_REL_ERR:.0%} "
          f"Monte Carlo error")

    test_a(cdm)
    test_b(cdm)

    pilot_path = os.path.join(d, "ep_pilot_results.npz")
    if os.path.exists(pilot_path):
        test_c(cdm, load_first_pilot(pilot_path))
    else:
        print(f"\n(skipping C: {pilot_path} not found)")


if __name__ == "__main__":
    main()
