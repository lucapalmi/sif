#!/usr/bin/env python
"""
Generates the P(k) tables the training set is built on. Runs LOCALLY, once.

CAMB is deliberately kept off the compute nodes: the tables are under a
megabyte, shipping them makes the cluster job depend on nothing but numpy and
pysif, and it pins the spectra so a re-run reproduces the same training set
rather than whatever CAMB version the cluster happens to have.

Every cosmology is evaluated on the SAME wavenumber grid -- same minkh, maxkh
and npoints -- so k is stored once and P(k) is a single 2D array, and no
interpolation ever touches the spectrum.

Usage:  python tools/ep_make_spectra.py [output.npz]
"""

import sys

import numpy as np
import camb

from ep_trainset_design import COSMO_BOX, build_cosmologies

K_MIN, K_MAX, N_K = 1e-4, 500.0, 2400
K_PAD_LO, K_PAD_HI, N_PAD = 1e-6, 1e4, 500


def camb_pk(p):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"],
                       mnu=p["mnu"], omk=0.0)
    pars.InitPower.set_params(ns=p["ns"], As=2.1e-9)
    pars.set_matter_power(redshifts=[p["z"]], kmax=K_MAX * 1.2)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    kh, _, pk = res.get_matter_power_spectrum(
        minkh=K_MIN, maxkh=K_MAX, npoints=N_K)
    return kh, pk[0]


def pad_grid(kh):
    """The padded k grid, identical for every cosmology."""
    lo = np.logspace(np.log10(K_PAD_LO), np.log10(kh[0]), N_PAD,
                     endpoint=False)
    hi = np.logspace(np.log10(kh[-1]), np.log10(K_PAD_HI), N_PAD)[1:]
    return np.concatenate([lo, kh, hi]), len(lo), len(hi)


def pad_pk(kh, pk, k_full, n_lo, n_hi):
    """Power-law continuation at both ends, in the spectrum's own slopes."""
    s_lo = np.log(pk[1] / pk[0]) / np.log(kh[1] / kh[0])
    s_hi = np.log(pk[-1] / pk[-2]) / np.log(kh[-1] / kh[-2])
    lo = pk[0] * (k_full[:n_lo] / kh[0]) ** s_lo
    hi = pk[-1] * (k_full[len(k_full) - n_hi:] / kh[-1]) ** s_hi
    return np.concatenate([lo, pk, hi])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "ep_spectra.npz"
    cosmologies = build_cosmologies()

    k_full = None
    rows = []

    for i, p in enumerate(cosmologies):
        kh, pk = camb_pk(p)
        if k_full is None:
            k_full, n_lo, n_hi = pad_grid(kh)
        rows.append(pad_pk(kh, pk, k_full, n_lo, n_hi))
        if i % 8 == 0 or i == len(cosmologies) - 1:
            print(f"  [{i:02d}/{len(cosmologies)}] omch2={p['omch2']:.4f} "
                  f"h={p['H0'] / 100:.3f} ns={p['ns']:.3f} "
                  f"mnu={p['mnu']:.3f} z={p['z']:.3f}", flush=True)

    pk_all = np.array(rows, dtype=np.float32)
    k_full = k_full.astype(np.float32)

    if not np.all(np.diff(k_full) > 0):
        raise SystemExit("k grid is not strictly increasing")
    if not np.all(pk_all >= 0):
        raise SystemExit("a spectrum went negative; the Gram covariance "
                         "requires P(k) >= 0")

    store = dict(k=k_full, pk=pk_all)
    for key in COSMO_BOX:
        store[f"param__{key}"] = np.array([c[key] for c in cosmologies])

    np.savez_compressed(out, **store)
    print(f"\n{len(cosmologies)} spectra on {len(k_full)} shared wavenumbers "
          f"({K_PAD_LO:.0e} to {K_PAD_HI:.0e} h/Mpc) -> {out}")


if __name__ == "__main__":
    main()
