#!/usr/bin/env python
"""
Feature coverage of realistic CDM spectra.

The ablation identified gamma^2 = 1 / (4 S <(d delta / dS)^2>) as the axis the
correction is most sensitive to, and the axis a training set must cover without
gaps. This measures what a realistic CDM parameter box actually spans, so the
training family can be designed against the domain the emulator will be used
in rather than against a guess.

No Monte Carlo: only the covariance and its derivative variance, which is
seconds per cosmology.

Usage:  python tools/ep_cdm_coverage.py [output_dir]
"""

import os
import sys

import numpy as np
import camb

import pysif.model as model

# Void-relevant smoothing scales, in Mpc/h.
R_MIN, R_MAX, N_RADII = 1.0, 50.0, 80

# A deliberately generous box around Planck: wider than a realistic prior, so
# the emulator's domain contains the chains rather than merely touching them.
BOX = dict(
    omch2=(0.09, 0.16),
    ombh2=(0.019, 0.026),
    H0=(60.0, 76.0),
    ns=(0.92, 1.00),
    mnu=(0.0, 0.4),
    z=(0.0, 1.5),
)

N_SAMPLES = 48
K_MAX = 200.0


def latin_hypercube(n, box, seed=7):
    rng = np.random.default_rng(seed)
    keys = list(box)
    d = len(keys)
    cube = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T
            + rng.random((n, d))) / n
    return [{k: box[k][0] + cube[i, j] * (box[k][1] - box[k][0])
             for j, k in enumerate(keys)} for i in range(n)]


def camb_pk(params):
    """Linear P(k) in (Mpc/h)^3 against k in h/Mpc."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params["H0"], ombh2=params["ombh2"],
                       omch2=params["omch2"], mnu=params["mnu"], omk=0.0)
    pars.InitPower.set_params(ns=params["ns"], As=2.1e-9)
    pars.set_matter_power(redshifts=[params["z"]], kmax=K_MAX)
    pars.NonLinear = camb.model.NonLinear_none

    results = camb.get_results(pars)
    kh, _, pk = results.get_matter_power_spectrum(
        minkh=1e-4, maxkh=K_MAX, npoints=1200)
    return kh, pk[0]


def extend_table(kh, pk, k_lo=1e-6, k_hi=1e4, n_pad=400):
    """Power-law continuation at both ends.

    The top-hat window oscillates as cos(kR) and the derivative variance
    weights the integrand by k^4, so the quadrature needs support well past
    the scales of interest; truncating the table biases gamma^2 rather than
    merely roughening it.
    """
    n_lo = np.log(pk[1] / pk[0]) / np.log(kh[1] / kh[0])
    n_hi = np.log(pk[-1] / pk[-2]) / np.log(kh[-1] / kh[-2])

    k_pad_lo = np.logspace(np.log10(k_lo), np.log10(kh[0]), n_pad,
                           endpoint=False)
    k_pad_hi = np.logspace(np.log10(kh[-1]), np.log10(k_hi), n_pad)[1:]

    p_pad_lo = pk[0] * (k_pad_lo / kh[0]) ** n_lo
    p_pad_hi = pk[-1] * (k_pad_hi / kh[-1]) ** n_hi

    k_all = np.concatenate([k_pad_lo, kh, k_pad_hi])
    p_all = np.concatenate([p_pad_lo, pk, p_pad_hi])
    return k_all.astype(np.float32), p_all.astype(np.float32)


def features(k, pk, radii):
    cov, sigma, high, dvar = model.delta_covariance_pk(k, pk, radii)
    n = len(radii)
    idx = np.arange(n)
    S = cov[idx * (idx + 1) // 2 + idx]
    gamma2 = 1.0 / (4.0 * S * dvar)
    return S, sigma.astype(float), gamma2, high.max()


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    radii = np.logspace(np.log10(R_MIN), np.log10(R_MAX),
                        N_RADII).astype(np.float32)
    samples = latin_hypercube(N_SAMPLES, BOX)

    rows = []
    for i, p in enumerate(samples):
        try:
            kh, pk = camb_pk(p)
        except Exception as e:
            print(f"  [{i:02d}] CAMB failed: {e}")
            continue
        k, pkx = extend_table(kh, pk)
        S, sigma, gamma2, high = features(k, pkx, radii)
        rows.append(dict(params=p, sigma=sigma, gamma2=gamma2, high=high))
        if i % 8 == 0:
            print(f"  [{i:02d}] Om h^2={p['omch2']:.3f} h={p['H0']/100:.2f} "
                  f"ns={p['ns']:.3f} mnu={p['mnu']:.2f} z={p['z']:.2f} -> "
                  f"gamma2 {gamma2.min():.4f}-{gamma2.max():.4f}, "
                  f"sigma {sigma.min():.3f}-{sigma.max():.3f}")

    g_all = np.concatenate([r["gamma2"] for r in rows])
    s_all = np.concatenate([r["sigma"] for r in rows])

    print(f"\n{'=' * 70}")
    print(f"{len(rows)} cosmologies, R in [{R_MIN}, {R_MAX}] Mpc/h")
    print(f"{'=' * 70}")
    print(f"gamma^2 overall range : {g_all.min():.4f} - {g_all.max():.4f}")
    print(f"gamma^2 within one curve (median span): "
          f"{np.median([r['gamma2'].max() - r['gamma2'].min() for r in rows]):.4f}")
    print(f"sigma overall range   : {s_all.min():.4f} - {s_all.max():.4f}")
    print(f"max high-k fraction   : {max(r['high'] for r in rows):.4f}")

    # How much of the gamma^2 axis does a single cosmology cover, against the
    # spread across cosmologies? If a single curve sweeps most of the range,
    # the axis is filled by the radius grid and not by the parameter sampling.
    per_curve = np.median([r["gamma2"].max() - r["gamma2"].min() for r in rows])
    across = g_all.max() - g_all.min()
    print(f"\nsingle-curve span / total span: {per_curve / across:.2f}")

    # gamma^2 at fixed radius across the box: the part the parameters control
    for R_probe in (5.0, 10.0, 20.0, 40.0):
        j = np.argmin(np.abs(radii - R_probe))
        vals = np.array([r["gamma2"][j] for r in rows])
        print(f"  at R = {radii[j]:5.1f} Mpc/h: gamma^2 = "
              f"{vals.min():.4f} - {vals.max():.4f} "
              f"(spread {100 * (vals.max() - vals.min()) / vals.mean():.1f}%)")

    np.savez(os.path.join(out_dir, "cdm_coverage.npz"),
             radii=radii, gamma2=np.array([r["gamma2"] for r in rows]),
             sigma=np.array([r["sigma"] for r in rows]))

    make_plot(rows, radii, out_dir)


def make_plot(rows, radii, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for r in rows:
        axes[0].plot(radii, r["gamma2"], lw=0.7, alpha=0.5, color="steelblue")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("R  [Mpc/h]")
    axes[0].set_ylabel(r"$\gamma^2$")
    axes[0].set_title(f"{len(rows)} CDM cosmologies")
    axes[0].grid(alpha=0.3)

    g_all = np.concatenate([r["gamma2"] for r in rows])
    axes[1].hist(g_all, bins=60, color="steelblue")
    axes[1].axvspan(0.179, 0.256, color="seagreen", alpha=0.2,
                    label="pilot EH coverage")
    for v, lab in [(0.045, "pl-1.0"), (0.121, "pl-2.5"), (0.167, "pl-2.0")]:
        axes[1].axvline(v, color="crimson", ls="--", lw=1)
        axes[1].text(v, axes[1].get_ylim()[1] * 0.9, lab, fontsize=7,
                     rotation=90, ha="right", color="crimson")
    axes[1].set_xlabel(r"$\gamma^2$")
    axes[1].set_ylabel("bins")
    axes[1].set_title("occupancy of the decisive axis")
    axes[1].legend(fontsize=8)

    for r in rows:
        axes[2].plot(r["sigma"], r["gamma2"], lw=0.7, alpha=0.5,
                     color="steelblue")
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"$\sigma(R)$")
    axes[2].set_ylabel(r"$\gamma^2$")
    axes[2].set_title(r"the manifold: is $\gamma^2$ a function of $\sigma$?")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "cdm_coverage.png")
    fig.savefig(path, dpi=150)
    print(f"\nplot written to {path}")


if __name__ == "__main__":
    main()
