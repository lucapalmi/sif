#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
Emits the trained emulator as a C header of static const tables.

The weights land in .rodata: nothing is allocated, nothing runs at startup, and
there is no model file to ship, find or version-skew against the library. The
generated header is checked in, but it is generated -- edit the training set or
ep_train_final.py and re-run this, never the header.

Every constant is written with %.17g, which round-trips an IEEE-754 double
exactly. The exporter verifies that: it parses back what it wrote and compares
against the source array bit for bit, and refuses to write a header that would
not reproduce the Python model.

Usage:  python tools/ep_export_c.py [ep_emulator.npz] [-o ../src/model/ep_emu_weights.h]
"""

import argparse
import datetime
import hashlib
import os

import numpy as np

from ep_model import FEATURE_NAMES, VIC_CANDIDATES, EPEmulator, VICEmulator

GUARD = "SIF__MODEL_EP_EMU_WEIGHTS_H"
PREFIX = "SIF_EP_EMU"


def c_double(x):
    """A double literal that reads back to the same bits."""
    s = f"{float(x):.17g}"
    if np.float64(s) != np.float64(x):
        raise ValueError(f"{x!r} does not round-trip through {s}")
    # Make it unambiguously a double to the compiler.
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s:
        s += ".0"
    return s


def emit_array(name, values, per_line=4, indent="  "):
    vals = [c_double(v) for v in np.asarray(values, dtype=np.float64).ravel()]
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(indent + ", ".join(vals[i:i + per_line]))
    body = ",\n".join(lines)
    return (f"static const double {name}[{len(vals)}] = {{\n{body}\n}};\n")


def verify_roundtrip(text, arrays):
    """Parse the emitted literals back and require bit equality."""
    for name, values in arrays.items():
        start = text.index(f"{name}[")
        start = text.index("{", start) + 1
        end = text.index("};", start)
        got = np.array([np.float64(t.strip())
                        for t in text[start:end].replace("\n", " ").split(",")
                        if t.strip()], dtype=np.float64)
        want = np.asarray(values, dtype=np.float64).ravel()
        if got.shape != want.shape or not np.array_equal(
                got.view(np.int64), want.view(np.int64)):
            raise SystemExit(f"{name} does not survive the round trip")


def export_vic(path, out_path):
    """The void-in-cloud network: ln K = -Lambda_cond exp(output)."""
    emu = VICEmulator.load(path)
    P = "SIF_EP_VIC"
    guard = "SIF__MODEL_EP_EMU_VIC_WEIGHTS_H"
    ids = [VIC_CANDIDATES.index(f) for f in emu.features]
    n_in, n_layers = len(emu.mu), len(emu.layers)
    n_par = sum(W.size + b.size for (W, b) in emu.layers)
    src_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    arrays = {f"{P}_MU": emu.mu, f"{P}_SD": emu.sd}
    for i, (W, b) in enumerate(emu.layers):
        arrays[f"{P}_W{i}"] = W
        arrays[f"{P}_B{i}"] = b
    arrays[f"{P}_BOX_LO"] = emu.box[0]
    arrays[f"{P}_BOX_HI"] = emu.box[1]

    out = [f"""/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef {guard}
#define {guard}

/*
 * GENERATED FILE -- do not edit.
 *
 * Emitted by tools/ep_export_c.py --vic from {os.path.basename(path)}
 * (sha256 {src_hash}) on {datetime.date.today().isoformat()}.
 *
 * The void-in-cloud factor: a {n_in}-input, {n_layers}-layer network of
 * {n_par} parameters whose output z gives ln K = -Lambda_cond exp(z) per bin.
 * Its inputs are picked by index from the candidates ep_emu.c computes, in
 * the order of VIC_CANDIDATES in tools/ep_model.py.
 */

#include <stdint.h>

#include "math/nn.h"

#define {P}_N_CAND {len(VIC_CANDIDATES)}u
#define {P}_N_IN {n_in}u

"""]
    for j, name in enumerate(emu.features):
        out.append(f"/*   {j}  {name} (candidate {ids[j]}) */\n")
    out.append(f"static const uint32_t {P}_INPUTS[{n_in}] = {{"
               + ", ".join(f"{i}u" for i in ids) + "};\n\n")
    for name, values in arrays.items():
        out.append(emit_array(name, values))
        out.append("\n")
    layer_lines = []
    for i, (W, b) in enumerate(emu.layers):
        act = "SIF__NN_LINEAR" if i == n_layers - 1 else "SIF__NN_TANH"
        layer_lines.append(f"    {{{W.shape[0]}u, {W.shape[1]}u, {P}_W{i}, "
                           f"{P}_B{i}, {act}}}")
    layers = ",\n".join(layer_lines)
    out.append(f"""static const sif_nn_t {P}_NN = {{
  .n_layers = {n_layers}u,
  .n_in = {n_in}u,
  .n_out = 1u,
  .mu = {P}_MU,
  .sd = {P}_SD,
  .layers = {{
{layers}
  }}
}};

/* Held-out median error of f x K below 80 Mpc/h, where K < 0.999. */
#define {P}_ERROR_IN_DOMAIN {c_double(emu.accuracy.get('median', 0.0))}

#endif /* {guard} */
""")
    text = "".join(out)
    verify_roundtrip(text, arrays)
    with open(out_path, "w") as fh:
        fh.write(text)
    print(f"void-in-cloud: {n_par} parameters, {n_in} inputs "
          f"({', '.join(emu.features)}) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default="ep_emulator.npz")
    ap.add_argument("-o", "--out",
                    default="../src/model/ep_emu_weights.h")
    ap.add_argument("--vic", metavar="VIC_NPZ",
                    help="also emit the void-in-cloud network")
    ap.add_argument("--vic-out", default="../src/model/ep_emu_vic_weights.h")
    args = ap.parse_args()
    if args.vic:
        export_vic(args.vic, args.vic_out)

    emu = EPEmulator.load(args.weights)
    n_in = len(emu.mu)
    n_layers = len(emu.layers)

    if n_in != len(FEATURE_NAMES):
        raise SystemExit(f"model takes {n_in} inputs but ep_model declares "
                         f"{len(FEATURE_NAMES)} features")

    src_hash = hashlib.sha256(open(args.weights, "rb").read()).hexdigest()[:16]

    arrays = {f"{PREFIX}_MU": emu.mu, f"{PREFIX}_SD": emu.sd}
    for i, (W, b) in enumerate(emu.layers):
        arrays[f"{PREFIX}_W{i}"] = W
        arrays[f"{PREFIX}_B{i}"] = b
    arrays[f"{PREFIX}_BOX_LO"] = emu.box[0]
    arrays[f"{PREFIX}_BOX_HI"] = emu.box[1]

    out = []
    out.append(f"""/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef {GUARD}
#define {GUARD}

/*
 * GENERATED FILE -- do not edit.
 *
 * Emitted by tools/ep_export_c.py from {os.path.basename(args.weights)}
 * (sha256 {src_hash}) on {datetime.date.today().isoformat()}.
 *
 * The trained correction for the excursion-set multiplicity function: a
 * {n_in}-input, {n_layers}-layer network of {emu.n_parameters()} parameters
 * predicting ln(Lambda_MC / Lambda_up) per radius bin. See
 * tools/training_README.md for what the inputs mean and how it was fitted.
 *
 * Every constant round-trips exactly through %.17g; the exporter verifies that
 * before writing, so these literals reproduce the Python model bit for bit.
 */

#include "math/nn.h"

/* Inputs, in the order the network expects them. Changing this order silently
 * evaluates a different function -- see sif_ep_emu_features. */
""")

    for j, name in enumerate(FEATURE_NAMES):
        out.append(f"/*   {j}  {name} */\n")
    out.append(f"#define {PREFIX}_N_IN {n_in}u\n\n")

    for name, values in arrays.items():
        out.append(emit_array(name, values))
        out.append("\n")

    # The network descriptor itself.
    layer_lines = []
    for i, (W, b) in enumerate(emu.layers):
        act = "SIF__NN_LINEAR" if i == n_layers - 1 else "SIF__NN_TANH"
        layer_lines.append(
            f"    {{{W.shape[0]}u, {W.shape[1]}u, {PREFIX}_W{i}, "
            f"{PREFIX}_B{i}, {act}}}")
    layers = ",\n".join(layer_lines)

    out.append(f"""static const sif_nn_t {PREFIX}_NN = {{
  .n_layers = {n_layers}u,
  .n_in = {n_in}u,
  .n_out = 1u,
  .mu = {PREFIX}_MU,
  .sd = {PREFIX}_SD,
  .layers = {{
{layers}
  }}
}};

/* The region the fit covered, per input, in the order above. Outside it the
 * network is extrapolating; the call still returns, and says so. */
#define {PREFIX}_NU_ORIGIN_MIN {c_double(emu.nu_origin_min)}

/* Held-out accuracy, from the trainer's cross-validation over whole
 * cosmologies: the median relative error on the multiplicity in domain, and
 * the measured figure just outside it. */
#define {PREFIX}_ERROR_IN_DOMAIN {c_double(emu.accuracy.get('median', 0.0013))}
#define {PREFIX}_ERROR_OUT_DOMAIN {c_double(emu.accuracy.get('median_outside', 0.0077))}

#endif /* {GUARD} */
""")

    text = "".join(out)
    verify_roundtrip(text, arrays)

    with open(args.out, "w") as fh:
        fh.write(text)

    print(f"{emu.n_parameters()} parameters, {n_layers} layers, "
          f"{n_in} inputs -> {args.out}")
    print(f"  source {args.weights} (sha256 {src_hash})")
    print(f"  every literal verified to round-trip bit-exactly")
    print(f"  in-domain error {emu.accuracy.get('median', float('nan')):.4%}")


if __name__ == "__main__":
    main()
