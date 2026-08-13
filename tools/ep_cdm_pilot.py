#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
CDM-only pilot ensemble for the EP hazard-ratio correction.

The first pilot spanned spectral families to map the correction's shape. This
one stays inside the domain the emulator will actually be used in -- CAMB
spectra over a realistic parameter box, including massive neutrinos and
redshift -- so that generalization can be tested the way it will be used.

The cosmologies are sampled to fill the box, and the analysis that follows
holds out CONTIGUOUS REGIONS of it (high neutrino mass, high redshift, low
matter density). On a manifold this thin, holding out one scattered cosmology
at a time measures interpolation between near-duplicates and reports it as
generalization.

Usage:  python tools/ep_cdm_pilot.py [output_dir]
"""

import os
import sys
import time

import numpy as np
import camb

import pysif.model as model
from ep_ratio_pilot import (baseline_features, baseline_hazard, cov_diag,
                            mc_hazard)

BOX = dict(
    omch2=(0.09, 0.16),
    ombh2=(0.019, 0.026),
    H0=(60.0, 76.0),
    ns=(0.92, 1.00),
    mnu=(0.0, 0.4),
    z=(0.0, 1.5),
)

N_COSMO = 16
N_RADII = 101
N_PATHS = 20_000_000
K_MAX = 200.0

# sigma at the walk's origin (the largest radius). Small enough that few paths
# start above the barrier, which would otherwise be a point mass outside every
# bin; the first pilot's dropped fractions came out at 1e-6 to 4e-4 here.
SIGMA_LO, SIGMA_HI = 0.25, 2.2

BARRIERS = [
    ("flat_low", dict(alpha=0.55, beta=0.30, gamma=0.0)),
    ("flat_high", dict(alpha=0.75, beta=0.30, gamma=0.0)),
    ("mild", dict(alpha=0.50, beta=0.30, gamma=0.6)),
    ("steep", dict(alpha=0.40, beta=0.50, gamma=1.0)),
    ("heavy", dict(alpha=0.70, beta=0.20, gamma=0.3)),
]


def latin_hypercube(n, box, seed=11):
    rng = np.random.default_rng(seed)
    keys = list(box)
    cube = (rng.permuted(np.tile(np.arange(n), (len(keys), 1)), axis=1).T
            + rng.random((n, len(keys)))) / n
    return [{k: box[k][0] + cube[i, j] * (box[k][1] - box[k][0])
             for j, k in enumerate(keys)} for i in range(n)]


def camb_pk(p):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"],
                       mnu=p["mnu"], omk=0.0)
    pars.InitPower.set_params(ns=p["ns"], As=2.1e-9)
    pars.set_matter_power(redshifts=[p["z"]], kmax=K_MAX)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    kh, _, pk = res.get_matter_power_spectrum(
        minkh=1e-4, maxkh=K_MAX, npoints=1200)
    return kh, pk[0]


def extend_table(kh, pk, k_lo=1e-6, k_hi=1e4, n_pad=400):
    n_lo = np.log(pk[1] / pk[0]) / np.log(kh[1] / kh[0])
    n_hi = np.log(pk[-1] / pk[-2]) / np.log(kh[-1] / kh[-2])
    k_pad_lo = np.logspace(np.log10(k_lo), np.log10(kh[0]), n_pad,
                           endpoint=False)
    k_pad_hi = np.logspace(np.log10(kh[-1]), np.log10(k_hi), n_pad)[1:]
    k = np.concatenate([k_pad_lo, kh, k_pad_hi])
    p = np.concatenate([pk[0] * (k_pad_lo / kh[0]) ** n_lo, pk,
                        pk[-1] * (k_pad_hi / kh[-1]) ** n_hi])
    return k.astype(np.float32), p.astype(np.float32)


def radii_for_sigma_range(k, pk, n_radii=N_RADII):
    """Radii spanning sigma in [SIGMA_LO, SIGMA_HI], clipped to what exists."""
    scan = np.logspace(-1.5, 2.8, 200).astype(np.float32)
    _, sig, _, _ = model.delta_covariance_pk(k, pk, scan)
    good = sig > 0
    ln_r = np.log(scan[good].astype(float))
    ln_s = np.log(sig[good].astype(float))
    s_hi = min(SIGMA_HI, 0.97 * sig[good].max())
    s_lo = max(SIGMA_LO, 1.03 * sig[good].min())
    r_of_s = lambda s: np.exp(np.interp(np.log(s), ln_s[::-1], ln_r[::-1]))
    return np.logspace(np.log10(r_of_s(s_hi)), np.log10(r_of_s(s_lo)),
                       n_radii).astype(np.float32), s_lo, s_hi


def run_case(radii, cov, sigma, dvar, bpars, n_paths=N_PATHS, seed=98765):
    n = len(radii)
    barrier = model.barrier_smt(sigma, **bpars)
    S = cov_diag(cov, n).astype(float)
    nu, gamma2, y, f_up = baseline_features(S, barrier.astype(float), dvar)
    lam_up = baseline_hazard(S, f_up)

    counts = model.first_crossing_counts_ep(
        radii, cov, barrier, n_paths=n_paths, seed=seed)
    lam_mc, rel_err, alive = mc_hazard(counts.astype(float), n_paths)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lam_up > 0.0, lam_mc / lam_up, np.nan)

    mid = lambda a: 0.5 * (a[:-1] + a[1:])
    return dict(radii=radii.astype(float), S=S, counts=counts,
                nu=mid(nu), gamma2=mid(gamma2), y=mid(y),
                lam_up=lam_up, lam_mc=lam_mc, ratio=ratio, rel_err=rel_err,
                crossed=counts.sum() / n_paths, dropped=counts[-1] / n_paths)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    cosmologies = latin_hypercube(N_COSMO, BOX)
    store = {}
    meta = []
    t_start = time.perf_counter()

    for i, p in enumerate(cosmologies):
        kh, pk = camb_pk(p)
        k, pkx = extend_table(kh, pk)
        radii, s_lo, s_hi = radii_for_sigma_range(k, pkx)
        cov, sigma, high, dvar = model.delta_covariance_pk(k, pkx, radii)

        n = len(radii)
        idx = np.arange(n)
        g2 = 1.0 / (4.0 * cov[idx * (idx + 1) // 2 + idx] * dvar)

        print(f"[{i:02d}] omch2={p['omch2']:.4f} ombh2={p['ombh2']:.4f} "
              f"h={p['H0'] / 100:.3f} ns={p['ns']:.3f} mnu={p['mnu']:.3f} "
              f"z={p['z']:.3f} | R {radii[0]:.2f}-{radii[-1]:.1f} "
              f"sigma {sigma.min():.3f}-{sigma.max():.3f} "
              f"g2 {g2.min():.4f}-{g2.max():.4f}", flush=True)

        for bname, bpars in BARRIERS:
            case = run_case(radii, cov, sigma, dvar, bpars)
            tag = f"c{i:02d}__{bname}"
            for key, val in case.items():
                store[f"{tag}__{key}"] = val
            print(f"      {bname:<10} crossed {case['crossed']:.3f} "
                  f"dropped {case['dropped']:.2e} "
                  f"ratio {np.nanmin(case['ratio']):.3f}-"
                  f"{np.nanmax(case['ratio']):.3f}", flush=True)

        meta.append(dict(index=i, **p))

    for key in BOX:
        store[f"param__{key}"] = np.array([m[key] for m in meta])
    store["param__index"] = np.array([m["index"] for m in meta])
    store["barrier_names"] = np.array([b for b, _ in BARRIERS])

    path = os.path.join(out_dir, "ep_cdm_pilot.npz")
    np.savez(path, **store)
    print(f"\n{len(cosmologies) * len(BARRIERS)} runs in "
          f"{time.perf_counter() - t_start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
