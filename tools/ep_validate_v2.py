#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
Validation for the extended (campaign 2) emulator, against fresh Monte Carlo.

Three sets, every seed unrelated to training:

  V  generic: 8 new cosmologies x 6 origin-placed SMT barriers, walk origin
     log-uniform over 2-700 Mpc/h, 1e9 paths, 128 radii
  T  a calibrated void barrier at the thesis's thresholds (delta_NL -0.15 to
     -0.70 at z = 0), on the grid an analysis actually uses: 200 radii, nu = 4.8
     at the origin, inner edge 1.3 below R_L(nu_E = 2). 1e9 paths
  W  the far-origin corner: 8 new cosmologies x (3 SMT barriers with beta
     log-uniform + 1 near-constant barrier, beta = 1e-3), walk origin
     150-200 Mpc/h and reaching below 60 -- the long walks of a shallow void
     threshold at low redshift. 1e9 paths, 128 radii
  C  convergence of the Monte Carlo itself: four problems at 65, 129, 257 and
     513 radii over the same span, 2e8 paths. The grids are nested, so the
     finer runs' counts sum exactly into the coarse bins and the comparison
     carries no binning effect. The network learns the ~128-radius Monte
     Carlo; if that is not converged to the accuracy asked of the emulator,
     the emulator cannot be either

Stages:
  spectra   LOCAL (needs CAMB): writes ep_validation_spectra.npz
  mc        CLUSTER: --task i, see N_TASKS; writes val_v2/<set>_<i>.npz
  analyse   anywhere: scores an emulator against V and T, reports C
  invariance  anywhere, no Monte Carlo: dependence on where the grid starts
            and on the radius sampling

Usage:
  python ep_validate_v2.py spectra
  python ep_validate_v2.py mc --task 0            # 0 .. N_TASKS - 1
  python ep_validate_v2.py analyse --weights ep_emulator.npz [--rmax 80]
  python ep_validate_v2.py invariance --weights ep_emulator.npz
"""

import argparse
import glob
import os
import time

import numpy as np

from ep_model import (EPEmulator, compute_features, mc_hazard,
                      multiplicity_from_hazard, upper_tail)
from ep_trainset_design import (BARRIER_BOX, BETA_LOG_3, COSMO_BOX,
                                NU_ORIGIN_HI, NU_ORIGIN_LO, R_ORIGIN_HI,
                                R_ORIGIN_LO, R_REACH_3, barrier_values,
                                latin_hypercube, radii_from_origin)

HERE = os.path.dirname(os.path.abspath(__file__))
SPECTRA = os.path.join(HERE, "ep_validation_spectra.npz")
SCAN = np.logspace(-2.5, 3.0, 500).astype(np.float32)

SEED = 990_000
N_COSMO_V, N_BAR_V = 8, 6
N_PATHS = 1_000_000_000
N_PATHS_C = 200_000_000
C_RADII = (65, 129, 257, 513)   # nested: every grid contains the 65-point one

# The fiducial cosmology of the thesis analysis (Planck-like, massless
# neutrinos), and its calibrated barrier: B = alpha [1 + (beta/sigma)^gamma]
# with alpha = A t^n, log10 beta = B0 + B1 t, t = |delta_L| / G the threshold
# in the amplitude the barrier was fitted at. G = sigma12(z=0) / sigma12(anchor)
# for this cosmology. Fixed here so the set is a stable benchmark; it is one
# realistic member of the SMT family, not the only one the emulator serves.
FIDUCIAL = dict(H0=67.0, ombh2=0.022445, omch2=0.120567, mnu=0.0, ns=0.96,
                z=0.0)
THESIS = dict(A=3.826338, n=1.619684, B0=-1.722868, B1=-18.943732,
              gamma=0.530195, G=76.999)
T_THRESHOLDS = (-0.15, -0.20, -0.25, -0.30, -0.40, -0.50, -0.60, -0.70)
T_RADII = 200
T_NU_ORIGIN = 4.8

# The far-origin corner, set W.
N_BAR_W = 3
W_ORIGIN = (150.0, 200.0)
W_BETA_CONST = 1e-3

# task layout: V by cosmology, then T, then one task per C problem, then W by
# cosmology (appended after C so the first 13 tasks keep their meaning)
N_TASKS = N_COSMO_V + 1 + 4 + N_COSMO_V
W_FIRST = N_COSMO_V + 1 + 4


def linear_contrast(delta_nl, c=1.594):
    """|delta_L| for a non-linear void contrast, Bernardeau (1994)."""
    return abs(c * (1.0 - (1.0 + delta_nl) ** (-1.0 / c)))


def thesis_barrier(delta_nl):
    t = linear_contrast(delta_nl) / THESIS["G"]
    return dict(alpha=THESIS["A"] * t ** THESIS["n"] * THESIS["G"],
                beta=10 ** (THESIS["B0"] + THESIS["B1"] * t) * THESIS["G"],
                gamma=THESIS["gamma"], perturb_seed=0)


def thesis_radii(sig_scan, delta_nl, n=T_RADII):
    """The analysis grid: nu = 4.8 outermost, 1.3 inside R_L(nu_E = 2)."""
    sig = np.asarray(sig_scan, dtype=float)
    p = thesis_barrier(delta_nl)
    nu = barrier_values(sig, p) / sig
    hi = float(np.interp(T_NU_ORIGIN, nu, SCAN))
    r_e2 = float(np.interp(2.0, linear_contrast(delta_nl) / sig, SCAN))
    r_in = r_e2 * (1.0 + delta_nl) ** (1.0 / 3.0) / 1.3
    return np.geomspace(r_in, hi, n).astype(np.float32), p


# --- stage: spectra ----------------------------------------------------------

def stage_spectra():
    import camb
    from ep_make_spectra import K_MAX, K_MIN, N_K, pad_grid, pad_pk

    def one(p):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"],
                           mnu=p["mnu"], omk=0.0)
        pars.InitPower.set_params(ns=p["ns"], As=2.1e-9)
        pars.set_matter_power(redshifts=[p["z"]], kmax=K_MAX * 1.2)
        pars.NonLinear = camb.model.NonLinear_none
        kh, _, pk = camb.get_results(pars).get_matter_power_spectrum(
            minkh=K_MIN, maxkh=K_MAX, npoints=N_K)
        kf, nlo, nhi = pad_grid(kh)
        return kf.astype(np.float32), pad_pk(kh, pk[0], kf, nlo,
                                             nhi).astype(np.float32)

    store = {}
    cosmos = latin_hypercube(N_COSMO_V, COSMO_BOX, np.random.default_rng(SEED))
    for i, p in enumerate(cosmos):
        store["k"], store[f"v{i}"] = one(p)
        for key, val in p.items():
            store[f"v{i}__{key}"] = np.float64(val)
    store["k"], store["fid"] = one(FIDUCIAL)
    np.savez_compressed(SPECTRA, **store)
    print(f"{N_COSMO_V} + 1 spectra -> {SPECTRA}")


# --- stage: mc ---------------------------------------------------------------

def run_mc(k, pk, radii, p, n_paths, seed):
    import pysif.model as model
    cov, sigma, _, dvar = model.delta_covariance_pk(k, pk, radii)
    barrier = barrier_values(sigma.astype(np.float64), p).astype(np.float32)
    t0 = time.perf_counter()
    counts = model.ep_first_crossing_counts(radii, cov, barrier,
                                            n_paths=n_paths, seed=seed)
    n = len(radii)
    idx = np.arange(n)
    return dict(radii=radii, sigma=sigma, deriv_variance=dvar,
                barrier=barrier, counts=counts,
                S=cov[idx * (idx + 1) // 2 + idx]), time.perf_counter() - t0


def stage_mc(task, out_dir):
    import pysif.model as model
    d = np.load(SPECTRA)
    k = np.ascontiguousarray(d["k"])
    store = {}

    def keep(tag, r, p, n_paths, **extra):
        for key, val in r.items():
            store[f"{tag}__{key}"] = val
        for key in ("alpha", "beta", "gamma", "perturb_seed"):
            store[f"{tag}__{key}"] = np.float64(p[key])
        store[f"{tag}__n_paths"] = np.uint64(n_paths)
        for key, val in extra.items():
            store[f"{tag}__{key}"] = np.float64(val)

    if task < N_COSMO_V:                                   # set V
        pk = np.ascontiguousarray(d[f"v{task}"])
        _, ss, _, _ = model.delta_covariance_pk(k, pk, SCAN)
        rng = np.random.default_rng(SEED + 100 + task)
        name = f"V_{task}"
        for j in range(N_BAR_V):
            r_o = float(np.exp(rng.uniform(np.log(R_ORIGIN_LO),
                                           np.log(R_ORIGIN_HI))))
            while True:
                p = dict(beta=rng.uniform(*BARRIER_BOX["beta"]),
                         gamma=rng.uniform(*BARRIER_BOX["gamma"]),
                         perturb_seed=0)
                nu_o = rng.uniform(NU_ORIGIN_LO, NU_ORIGIN_HI)
                res = radii_from_origin(ss, SCAN, p, r_o, nu_o)
                if res is not None:
                    break
            radii, p["alpha"] = res
            r, dt = run_mc(k, pk, radii, p, N_PATHS, SEED + 1000 * task + j)
            keep(f"V{task}x{j}", r, p, N_PATHS, r_origin=r_o, nu_origin=nu_o,
                 cosmo=task)
            print(f"V{task}x{j} R_o={r_o:6.1f} nu_o={nu_o:.2f} "
                  f"radii {radii[0]:.2f}..{radii[-1]:.1f} {dt:.0f}s", flush=True)

    elif task == N_COSMO_V:                                # set T
        pk = np.ascontiguousarray(d["fid"])
        _, ss, _, _ = model.delta_covariance_pk(k, pk, SCAN)
        name = "T"
        for j, dnl in enumerate(T_THRESHOLDS):
            radii, p = thesis_radii(ss, dnl)
            r, dt = run_mc(k, pk, radii, p, N_PATHS, SEED + 50000 + j)
            keep(f"T{j}", r, p, N_PATHS, delta_nl=dnl,
                 r_lo=1.3 * float(radii[0]))
            print(f"T{j} dNL={dnl:+.2f} radii {radii[0]:.1f}..{radii[-1]:.1f} "
                  f"{dt:.0f}s", flush=True)

    elif task >= W_FIRST:                                  # set W
        i = task - W_FIRST
        pk = np.ascontiguousarray(d[f"v{i}"])
        _, ss, _, _ = model.delta_covariance_pk(k, pk, SCAN)
        rng = np.random.default_rng(SEED + 200 + i)
        name = f"W_{i}"
        for j in range(N_BAR_W + 1):
            const = j == N_BAR_W
            for _ in range(1000):
                r_o = float(np.exp(rng.uniform(*np.log(W_ORIGIN))))
                beta = (W_BETA_CONST if const else float(np.exp(
                    rng.uniform(*np.log(BETA_LOG_3)))))
                p = dict(beta=beta, perturb_seed=0,
                         gamma=1.0 if const else rng.uniform(
                             *BARRIER_BOX["gamma"]))
                nu_o = rng.uniform(NU_ORIGIN_LO, NU_ORIGIN_HI)
                res = radii_from_origin(ss, SCAN, p, r_o, nu_o)
                if res is not None and res[0][0] < R_REACH_3:
                    break
            radii, p["alpha"] = res
            r, dt = run_mc(k, pk, radii, p, N_PATHS, SEED + 3000 * i + j)
            keep(f"W{i}x{j}", r, p, N_PATHS, r_origin=r_o, nu_origin=nu_o,
                 cosmo=i, constant=float(const))
            print(f"W{i}x{j} R_o={r_o:6.1f} nu_o={nu_o:.2f} beta={beta:.4f} "
                  f"radii {radii[0]:.2f}..{radii[-1]:.1f} {dt:.0f}s", flush=True)

    else:                                                  # set C
        c = task - N_COSMO_V - 1
        pk = np.ascontiguousarray(d["fid"])
        _, ss, _, _ = model.delta_covariance_pk(k, pk, SCAN)
        if c < 2:
            radii0, p = thesis_radii(ss, (-0.20, -0.45)[c])
            span = (float(radii0[0]), float(radii0[-1]))
        else:
            p = dict(beta=0.3, gamma=1.2, perturb_seed=0)
            r_o = (30.0, 300.0)[c - 2]
            radii0, p["alpha"] = radii_from_origin(ss, SCAN, p, r_o, 4.5)
            span = (float(radii0[0]), float(radii0[-1]))
        name = f"C_{c}"
        for nr in C_RADII:
            radii = np.geomspace(*span, nr).astype(np.float32)
            r, dt = run_mc(k, pk, radii, p, N_PATHS_C, SEED + 70000 + 10 * c)
            keep(f"C{c}n{nr}", r, p, N_PATHS_C, problem=c, n_radii=nr)
            print(f"C{c} n={nr} radii {span[0]:.1f}..{span[1]:.1f} {dt:.0f}s",
                  flush=True)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **store)


# --- stage: analyse ----------------------------------------------------------

def load_curves(out_dir):
    curves = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.npz"))):
        d = np.load(path)
        for tag in sorted({k.split("__")[0] for k in d.files}):
            c = {key.split("__", 1)[1]: d[key] for key in d.files
                 if key.startswith(tag + "__")}
            c["tag"] = tag
            curves.append(c)
    return curves


def score_curve(emu, c):
    """Per-bin emulator error against the Monte Carlo's own counts."""
    radii = c["radii"].astype(float)
    S = c["S"].astype(float)
    X, lam_up, nu_large = compute_features(radii, S, c["barrier"].astype(float),
                                           c["deriv_variance"].astype(float))
    n_paths = int(c["n_paths"])
    lam_mc, rel, _ = mc_hazard(c["counts"], n_paths)
    f = multiplicity_from_hazard(lam_up * np.exp(emu.correction(X)), radii,
                                 nu_large)
    f_mc = c["counts"][:-1] / (n_paths * np.diff(radii))
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(f_mc > 0, f / f_mc - 1.0, np.nan)
        lr = np.log(lam_up * np.exp(emu.correction(X)) / lam_mc)
    r_mid = 0.5 * (radii[1:] + radii[:-1])
    m = np.isfinite(e) & (rel < 0.05)
    if "r_lo" in c:
        m &= r_mid >= float(c["r_lo"])
    info = emu.check_domain(X, nu_large)
    return dict(e=e, lr=lr, rel=rel, r=r_mid, nu=X[:, 0], m=m, info=info)


def summary(rows, label):
    e = np.concatenate([r["e"][r["m"]] for r in rows])
    lr = np.concatenate([r["lr"][r["m"]] for r in rows])
    rel = np.concatenate([r["rel"][r["m"]] for r in rows])
    if len(e) == 0:
        return f"  {label:<30} no bins"
    a = np.abs(e)
    model_err = np.sqrt(max(np.mean(lr ** 2) - np.mean(rel ** 2), 0.0))
    return (f"  {label:<30}{len(a):>7}{np.median(a):>9.3%}"
            f"{np.percentile(a, 95):>8.2%}{np.median(rel):>9.3%}"
            f"{model_err:>11.2e}{np.sqrt(np.mean((lr / rel) ** 2)):>7.2f}")


def stage_analyse(out_dir, weights, rmax, max_origin=250.0):
    emu = EPEmulator.load(weights)
    curves = load_curves(out_dir)
    far = [c for c in curves if "r_origin" in c
           and float(c["r_origin"]) > max_origin]
    curves = [c for c in curves if not any(c is f for f in far)]
    if far:
        print(f"({len(far)} curves starting beyond {max_origin:g} Mpc/h, "
              f"outside the contract, left out)")
    head = (f"  {'set':<30}{'bins':>7}{'median':>9}{'95th':>8}{'MC noise':>9}"
            f"{'model err':>11}{'chi':>7}")
    print(f"emulator {weights}\n{head}\n  " + "-" * (len(head) - 2))
    for prefix, label in (("V", "V generic"), ("T", "T thesis barrier"),
                          ("W", "W far origin (150-200)")):
        rows = [score_curve(emu, c) for c in curves
                if c["tag"].startswith(prefix)]
        if not rows:
            continue
        print(summary(rows, f"{label}, all R"))
        cut = [dict(r, m=r["m"] & (r["r"] < rmax)) for r in rows]
        print(summary(cut, f"{label}, R < {rmax:g}"))
        if prefix == "V":
            for lo, hi in ((2, 30), (30, 150), (150, max_origin)):
                sel = [dict(r, m=r["m"] & (r["r"] < rmax)) for r, c in
                       zip(rows, [c for c in curves if c["tag"][0] == "V"])
                       if lo <= float(c["r_origin"]) < hi]
                print(summary(sel, f"  origin {lo}-{hi}, R < {rmax:g}"))
            out = sum(not r["info"]["ok"] for r in rows)
            print(f"  {'':<30}guard flags {out} of {len(rows)} curves")
        if prefix == "T":
            for r, c in zip(rows, [c for c in curves if c["tag"][0] == "T"]):
                print(summary([r], f"  delta_NL {float(c['delta_nl']):+.2f}"))
        if prefix == "W":
            wc = [c for c in curves if c["tag"][0] == "W"]
            for flag, lab in ((0.0, "SMT"), (1.0, "near-constant")):
                sel = [dict(r, m=r["m"] & (r["r"] < rmax))
                       for r, c in zip(rows, wc)
                       if float(c["constant"]) == flag]
                print(summary(sel, f"  {lab}, R < {rmax:g}"))
            out = sum(not r["info"]["ok"] for r in rows)
            print(f"  {'':<30}guard flags {out} of {len(rows)} curves")

    conv = [c for c in curves if c["tag"].startswith("C")]
    if conv:
        print("\nMonte Carlo convergence: crossing probability per bin of the "
              "65-radius grid,\nP(n radii) / P(513) - 1, mean over nu bands "
              "(+- its error)")
        for prob in sorted({int(c["problem"]) for c in conv}):
            cs = sorted([c for c in conv if int(c["problem"]) == prob],
                        key=lambda c: int(c["n_radii"]))
            coarse = []
            for c in cs:
                m = (int(c["n_radii"]) - 1) // (int(cs[0]["n_radii"]) - 1)
                q = c["counts"][:-1].astype(float).reshape(-1, m).sum(axis=1)
                coarse.append(q / float(c["n_paths"]))
            nu = (cs[0]["barrier"] / np.sqrt(cs[0]["S"])).astype(float)
            nu = 0.5 * (nu[1:] + nu[:-1])
            ref = coarse[-1]
            n_ref = float(cs[-1]["n_paths"])
            line = f"  problem {prob}:"
            for c, q in zip(cs[:-1], coarse[:-1]):
                ok = (q > 0) & (ref > 0)
                ratio = q[ok] / ref[ok] - 1.0
                err = np.sqrt(1.0 / (q[ok] * float(c["n_paths"]))
                              + 1.0 / (ref[ok] * n_ref))
                parts = []
                for lo, hi in ((0, 1.5), (1.5, 3), (3, 9)):
                    sel = (nu[ok] >= lo) & (nu[ok] < hi)
                    if sel.sum() > 2:
                        w = 1.0 / err[sel] ** 2
                        mean = np.sum(w * ratio[sel]) / np.sum(w)
                        parts.append(f"nu {lo}-{hi}: {mean:+.3%}"
                                     f"+-{1.0 / np.sqrt(np.sum(w)):.3%}")
                line += f"\n    n={int(c['n_radii']):<4}" + "  ".join(parts)
            print(line)


def emulate(emu, k, pk, radii, p):
    """Emulated f on the bin centres of `radii`, for barrier parameters p."""
    import pysif.model as model
    _, s, _, dv = model.delta_covariance_pk(k, pk, radii)
    s = s.astype(float)
    X, lam, nu_large = compute_features(radii.astype(float), s ** 2,
                                        barrier_values(s, p),
                                        np.asarray(dv, dtype=float))
    f = multiplicity_from_hazard(lam * np.exp(emu.correction(X)),
                                 radii.astype(float), nu_large)
    return 0.5 * (radii[1:] + radii[:-1]).astype(float), f


def stage_invariance(weights):
    """Two properties the physics fixes and the fit does not enforce.

    origin: the Monte Carlo does not depend on where the grid starts once the
    walk starts above nu ~ 3.5 (first-step mass < 2.3e-4). Same spacing, grid
    started at nu 4.0, 4.8 and 5.2 against 4.5; max |change| over nu 1-3.3.
    grid: the density at the radii on nested 65-, 129- and 257-radius grids,
    max spread where the radii coincide, over nu 1-3.5.
    """
    import pysif.model as model
    emu = EPEmulator.load(weights)
    d = np.load(SPECTRA)
    k, pk = np.ascontiguousarray(d["k"]), np.ascontiguousarray(d["fid"])
    _, ss, _, _ = model.delta_covariance_pk(k, pk, SCAN)
    ss = ss.astype(float)

    cases = [(f"thesis {dnl:+.2f}", thesis_barrier(dnl))
             for dnl in (-0.15, -0.20, -0.45, -0.70)]
    for r_o in (30.0, 150.0, 400.0):
        p = dict(beta=0.3, gamma=1.2, perturb_seed=0)
        p["alpha"] = radii_from_origin(ss, SCAN, p, r_o, 4.5)[1]
        cases.append((f"SMT origin {r_o:.0f}", p))

    print(f"emulator {weights}\n  {'case':<20}{'origin 4.0':>11}"
          f"{'4.8':>8}{'5.2':>8}{'grid':>8}")
    for name, p in cases:
        nu = barrier_values(ss, p) / ss
        r_in = float(np.interp(1.0, nu, SCAN))
        rq = np.geomspace(1.3 * r_in, float(np.interp(3.3, nu, SCAN)), 25)
        step = np.log(float(np.interp(4.5, nu, SCAN)) / r_in) / 199
        f = {}
        for nu_o in (4.0, 4.5, 4.8, 5.2):
            n = int(round(np.log(float(np.interp(nu_o, nu, SCAN)) / r_in)
                          / step)) + 1
            radii = np.exp(np.log(r_in) + step * np.arange(n)).astype(
                np.float32)
            rc, fv = emulate(emu, k, pk, radii, p)
            f[nu_o] = np.exp(np.interp(np.log(rq), np.log(rc), np.log(fv)))
        origin = [np.max(np.abs(f[o] / f[4.5] - 1)) for o in (4.0, 4.8, 5.2)]

        # Grid: the density AT the radii on nested grids (65, 129, 257 radii
        # over nu 0.5-4.5), compared where the radii coincide -- no
        # interpolation, so what is measured is the emulator's own dependence
        # on the sampling.
        r0, r1 = float(np.interp(0.5, nu, SCAN)), float(np.interp(4.5, nu,
                                                                  SCAN))
        fs = []
        for n in (65, 129, 257):
            radii = np.geomspace(r0, r1, n).astype(np.float32)
            _, s, _, dv = model.delta_covariance_pk(k, pk, radii)
            f_r, _ = emu.multiplicity(radii.astype(float),
                                      s.astype(float) ** 2,
                                      barrier_values(s.astype(float), p),
                                      np.asarray(dv, dtype=float), check=False)
            fs.append(f_r[::(n - 1) // 64])
        rr = np.geomspace(r0, r1, 65)
        nn = np.interp(np.log(rr), np.log(SCAN), nu)
        sel = (nn > 1.0) & (nn < 3.5)
        fs = np.array(fs)[:, sel]
        grid = np.max(fs.max(axis=0) / fs.min(axis=0) - 1)
        print(f"  {name:<20}" + "".join(f"{o:>8.2%}" if i else f"{o:>11.2%}"
                                         for i, o in enumerate(origin))
              + f"{grid:>8.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("spectra", "mc", "analyse",
                                      "invariance"))
    ap.add_argument("--task", type=int)
    ap.add_argument("--out", default="val_v2")
    ap.add_argument("--weights", default=os.path.join(HERE, "ep_emulator.npz"))
    ap.add_argument("--rmax", type=float, default=80.0)
    ap.add_argument("--max-origin", type=float, default=250.0)
    a = ap.parse_args()
    if a.stage == "spectra":
        stage_spectra()
    elif a.stage == "mc":
        stage_mc(a.task, a.out)
    elif a.stage == "invariance":
        stage_invariance(a.weights)
    else:
        stage_analyse(a.out, a.weights, a.rmax, a.max_origin)


if __name__ == "__main__":
    main()
