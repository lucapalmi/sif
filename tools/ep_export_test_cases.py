#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
Emits stored reference cases for tests/test_ep_emu.c.

The C emulator has to reproduce ep_model.py, not merely resemble it: the
network was fitted against those exact feature expressions, so a stencil or a
lookback that differs in the last term is a different model, and no physical
invariant would catch it. These cases pin the whole chain -- baseline rate,
hazard integration, all nine features, the network, the survival recursion and
the density at the radii --
against values the Python produced.

The inputs are stored at the precision the C entry point actually receives
(sigma and the barrier as sif_real), and the expected outputs are computed from
those same reduced-precision inputs, so a mismatch means an implementation
difference rather than a rounding difference.

Usage:  python tools/ep_export_test_cases.py [-o ../tests/ep_emu_cases.h]
"""

import argparse
import datetime
import hashlib
import os

import numpy as np

from ep_model import EPEmulator, VICEmulator, compute_features

N_CASES = 3


def emit(name, values, fh):
    v = np.asarray(values, dtype=np.float64).ravel()
    body = ",\n  ".join(", ".join(f"{x:.17g}" for x in v[i:i + 4])
                        for i in range(0, len(v), 4))
    fh.write(f"static const double {name}[{len(v)}] = {{\n  {body}\n}};\n")


# Void-in-cloud: a constant delta_c and an ellipsoidal one, over a grid where
# K falls well below one, on a fixed training spectrum so the covariance can be
# rebuilt with the case. Barrier, collapse and cov are stored at the precision
# the C takes them.
VIC_CASES = (dict(cosmo=5, alpha=0.5, beta=0.3, gamma=0.9, beta_e=0.0),
             dict(cosmo=11, alpha=0.4, beta=0.2, gamma=1.3, beta_e=0.47))
VIC_N = 96


def write_vic_cases(fh, emu, args):
    import pysif.model as model
    vic = VICEmulator.load(args.vic)
    sp = np.load(args.spectra)
    k = np.ascontiguousarray(sp["k"])
    fh.write(f"/* Void-in-cloud: f with a collapse barrier, from "
             f"{os.path.basename(args.vic)}. */\n"
             f"#define EP_VIC_N_CASES {len(VIC_CASES)}\n"
             f"#define EP_VIC_CASE_N {VIC_N}\n\n")
    names = []
    for i, c in enumerate(VIC_CASES):
        radii = np.geomspace(0.6, 40.0, VIC_N).astype(np.float32)
        cov, sig, _, dv = model.delta_covariance_pk(
            k, np.ascontiguousarray(sp["pk"][c["cosmo"]]), radii)
        sigma = np.asarray(sig, np.float32)
        S = sigma.astype(np.float64) ** 2
        barrier = model.ep_barrier_smt(sigma, alpha=c["alpha"],
                                       beta=c["beta"], gamma=c["gamma"])
        sd = sigma.astype(np.float64)
        collapse = (1.686 * (1.0 + c["beta_e"] * (sd / 1.686) ** 1.23)
                    ).astype(np.float32)
        dvar = np.asarray(dv, np.float64)
        args_ = (radii.astype(np.float64), S, barrier.astype(np.float64),
                 dvar)
        kw = dict(collapse=collapse.astype(np.float64),
                  cov=np.asarray(cov, np.float64), vic=vic, check=False)
        f, _ = emu.multiplicity(*args_, **kw)
        fb, _ = emu.multiplicity(*args_, binned=True, **kw)
        for nm, v in (("RADII", radii), ("SIGMA", sigma), ("BARRIER", barrier),
                      ("COLLAPSE", collapse), ("DVAR", dvar),
                      ("COV", np.asarray(cov, np.float64)), ("EXPECT", f),
                      ("EXPECT_BINNED", fb)):
            emit(f"EP_VIC_{nm}_{i}", v, fh)
        fh.write("\n")
    for nm in ("RADII", "SIGMA", "BARRIER", "COLLAPSE", "DVAR", "COV",
               "EXPECT", "EXPECT_BINNED"):
        arrs = ", ".join(f"EP_VIC_{nm}_{i}" for i in range(len(VIC_CASES)))
        fh.write(f"static const double* const EP_VIC_{nm}[] = {{{arrs}}};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ep_emulator.npz")
    ap.add_argument("--mc", default="ep_validation_mc.npz")
    ap.add_argument("-o", "--out", default="../tests/ep_emu_cases.h")
    ap.add_argument("--vic", default="ep_emulator_vic.npz",
                    help="the void-in-cloud network the C header was "
                         "exported from")
    ap.add_argument("--spectra", default="ep_spectra_v2.npz")
    args = ap.parse_args()

    emu = EPEmulator.load(args.weights)
    d = np.load(args.mc)
    tags = sorted({k.split("__")[0] for k in d.files
                   if k.startswith("v")})[:N_CASES]

    wh = hashlib.sha256(open(args.weights, "rb").read()).hexdigest()[:16]
    cases = []

    for t in tags:
        radii = d[f"{t}__radii"].astype(np.float32)
        barrier = d[f"{t}__barrier"].astype(np.float32)
        dvar = d[f"{t}__dvar"].astype(np.float64)
        # sif_real is what the C entry point takes; derive S the way C does.
        sigma = np.sqrt(d[f"{t}__S"]).astype(np.float32)
        S = sigma.astype(np.float64) ** 2

        X, lam, nu_large = compute_features(
            radii.astype(np.float64), S, barrier.astype(np.float64), dvar)
        f, info = emu.multiplicity(
            radii.astype(np.float64), S, barrier.astype(np.float64), dvar)
        fb, _ = emu.multiplicity(
            radii.astype(np.float64), S, barrier.astype(np.float64), dvar,
            check=False, binned=True)
        cases.append((radii, sigma, barrier, dvar, f, fb, nu_large, info))

    n = len(cases[0][0])

    with open(args.out, "w") as fh:
        fh.write(f"""/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_TESTS_EP_EMU_CASES_H
#define SIF_TESTS_EP_EMU_CASES_H

/*
 * GENERATED FILE -- do not edit.
 *
 * Emitted by tools/ep_export_test_cases.py against ep_emulator.npz
 * (sha256 {wh}) on {datetime.date.today().isoformat()}.
 *
 * Reference values from the Python model the network was fitted with. The C
 * emulator must reproduce the multiplicity to the precision sif_real can hold;
 * anything worse means a feature is being computed differently, which no
 * physical invariant would detect. EXPECT is the density at the radii (the
 * default output), EXPECT_BINNED the bin averages (SIF_EP_EMU_BIN_CENTRES).
 */

#define EP_EMU_N_CASES {len(cases)}
#define EP_EMU_CASE_N {n}

""")
        for i, (radii, sigma, barrier, dvar, f, fb, nu_large, info) in \
                enumerate(cases):
            emit(f"EP_EMU_RADII_{i}", radii, fh)
            emit(f"EP_EMU_SIGMA_{i}", sigma, fh)
            emit(f"EP_EMU_BARRIER_{i}", barrier, fh)
            emit(f"EP_EMU_DVAR_{i}", dvar, fh)
            emit(f"EP_EMU_EXPECT_{i}", f, fh)
            emit(f"EP_EMU_EXPECT_BINNED_{i}", fb, fh)
            fh.write(f"static const double EP_EMU_NU_ORIGIN_{i} = "
                     f"{nu_large:.17g};\n")
            fh.write(f"static const int EP_EMU_IN_DOMAIN_{i} = "
                     f"{1 if info['ok'] else 0};\n\n")

        for nm in ("RADII", "SIGMA", "BARRIER", "DVAR", "EXPECT",
                   "EXPECT_BINNED"):
            arrs = ", ".join(f"EP_EMU_{nm}_{i}" for i in range(len(cases)))
            fh.write(f"static const double* const EP_EMU_{nm}[] = "
                     f"{{{arrs}}};\n")
        fh.write("static const double EP_EMU_NU_ORIGIN[] = {"
                 + ", ".join(f"EP_EMU_NU_ORIGIN_{i}"
                             for i in range(len(cases))) + "};\n")
        fh.write("static const int EP_EMU_IN_DOMAIN[] = {"
                 + ", ".join(f"EP_EMU_IN_DOMAIN_{i}"
                             for i in range(len(cases))) + "};\n")
        write_vic_cases(fh, emu, args)
        fh.write("\n#endif /* SIF_TESTS_EP_EMU_CASES_H */\n")

    print(f"{len(cases)} cases of {n} radii -> {args.out}")
    print(f"  weights sha256 {wh}")
    print(f"  in-domain: {[bool(c[7]['ok']) for c in cases]}")


if __name__ == "__main__":
    main()
