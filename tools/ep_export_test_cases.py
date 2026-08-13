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
hazard integration, all eight features, the network, the survival recursion --
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

from ep_model import EPEmulator, compute_features

N_CASES = 3


def emit(name, values, fh):
    v = np.asarray(values, dtype=np.float64).ravel()
    body = ",\n  ".join(", ".join(f"{x:.17g}" for x in v[i:i + 4])
                        for i in range(0, len(v), 4))
    fh.write(f"static const double {name}[{len(v)}] = {{\n  {body}\n}};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ep_emulator.npz")
    ap.add_argument("--mc", default="ep_validation_mc.npz")
    ap.add_argument("-o", "--out", default="../tests/ep_emu_cases.h")
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
        cases.append((radii, sigma, barrier, dvar, f, nu_large, info))

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
 * physical invariant would detect.
 */

#define EP_EMU_N_CASES {len(cases)}
#define EP_EMU_CASE_N {n}

""")
        for i, (radii, sigma, barrier, dvar, f, nu_large, info) in \
                enumerate(cases):
            emit(f"EP_EMU_RADII_{i}", radii, fh)
            emit(f"EP_EMU_SIGMA_{i}", sigma, fh)
            emit(f"EP_EMU_BARRIER_{i}", barrier, fh)
            emit(f"EP_EMU_DVAR_{i}", dvar, fh)
            emit(f"EP_EMU_EXPECT_{i}", f, fh)
            fh.write(f"static const double EP_EMU_NU_ORIGIN_{i} = "
                     f"{nu_large:.17g};\n")
            fh.write(f"static const int EP_EMU_IN_DOMAIN_{i} = "
                     f"{1 if info['ok'] else 0};\n\n")

        for nm in ("RADII", "SIGMA", "BARRIER", "DVAR", "EXPECT"):
            arrs = ", ".join(f"EP_EMU_{nm}_{i}" for i in range(len(cases)))
            fh.write(f"static const double* const EP_EMU_{nm}[] = "
                     f"{{{arrs}}};\n")
        fh.write("static const double EP_EMU_NU_ORIGIN[] = {"
                 + ", ".join(f"EP_EMU_NU_ORIGIN_{i}"
                             for i in range(len(cases))) + "};\n")
        fh.write("static const int EP_EMU_IN_DOMAIN[] = {"
                 + ", ".join(f"EP_EMU_IN_DOMAIN_{i}"
                             for i in range(len(cases))) + "};\n")
        fh.write("\n#endif /* SIF_TESTS_EP_EMU_CASES_H */\n")

    print(f"{len(cases)} cases of {n} radii -> {args.out}")
    print(f"  weights sha256 {wh}")
    print(f"  in-domain: {[bool(c[6]['ok']) for c in cases]}")


if __name__ == "__main__":
    main()
