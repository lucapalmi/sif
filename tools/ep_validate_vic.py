# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Void-in-cloud validation: the fresh sets of ep_validate_v2, replayed.

Every curve of val_v2/ (sets V, T, W) is walked again with its own seed and a
collapse barrier, so its void counts pair with the stored one-barrier counts
walk for walk. That scores the void-in-cloud factor K on its own, nearly free
of Monte Carlo noise, and the full emulated f x K against the void-in-cloud
counts directly. None of these curves touched training.

  V, W  a collapse barrier from draw_collapse, under a seed of its own
  T     the thesis barrier twice: delta_c = 1.686, and the ellipsoidal
        1.686 [1 + 0.47 (sigma / 1.686)^1.23] of Sheth, Mo & Tormen (2001)

    python ep_validate_vic.py mc --task 0 --out val_vic     # one per node
    python ep_validate_vic.py analyse --out val_vic --weights ep_emulator.npz
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ep_trainset_design import collapse_values, draw_collapse  # noqa: E402
from ep_validate_v2 import (N_COSMO_V, SEED, SPECTRA,  # noqa: E402
                            load_curves)

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_V2 = os.path.join(HERE, "..", "val_v2")
SEED_COLLAPSE = 77_000          # draw_collapse seed of its own, not training's
T_COLLAPSE = (dict(delta_c=1.686, beta_e=0.0, gamma_e=0.5),
              dict(delta_c=1.686, beta_e=0.47, gamma_e=0.615))
TASKS = ([f"V_{i}" for i in range(N_COSMO_V)] + ["T"]
         + [f"W_{i}" for i in range(N_COSMO_V)])


def stored_seed(tag):
    """The seed ep_validate_v2 walked this curve with."""
    if tag[0] == "V":
        task, j = (int(x) for x in tag[1:].split("x"))
        return SEED + 1000 * task + j
    if tag[0] == "T":
        return SEED + 50000 + int(tag[1:])
    i, j = (int(x) for x in tag[1:].split("x"))
    return SEED + 3000 * i + j


def stage_mc(task, out_dir, val_dir):
    import pysif.model as model
    name = TASKS[task]
    d = np.load(SPECTRA)
    k = np.ascontiguousarray(d["k"])
    curves = [c for c in load_curves(val_dir)
              if c["tag"].startswith(name[0]) and
              (name == "T" or c["tag"][1:].split("x")[0] == name[2:])]
    store = {}
    for n, c in enumerate(curves):
        tag = c["tag"]
        key = "fid" if tag[0] == "T" else f"v{int(c['cosmo'])}"
        pk = np.ascontiguousarray(d[key])
        radii = c["radii"]
        cov, sigma, _, _ = model.delta_covariance_pk(k, pk, radii)
        if n == 0:
            # The pairing rests on reproducing the stored walks; check it.
            again = model.ep_first_crossing_counts(
                radii, cov, c["barrier"], int(c["n_paths"]),
                seed=stored_seed(tag)).astype(np.int64)
            stored = c["counts"].astype(np.int64)
            moved = 0.5 * np.abs(again - stored).sum() / max(stored.sum(), 1)
            print(f"[verify] {tag}: {moved:.2e} of the crossings moved",
                  flush=True)
            if moved > 1e-4:
                sys.exit(f"[verify] {tag}: the replay does not pair with "
                         f"the stored counts ({moved:.2e} moved)")
        if tag[0] == "T":
            variants = [(f"{tag}c{m}", p) for m, p in enumerate(T_COLLAPSE)]
        else:
            i, j = (int(x) for x in tag[1:].split("x"))
            variants = [(tag, draw_collapse({"V": 91, "W": 93}[tag[0]], i, j,
                                            seed=SEED_COLLAPSE))]
        for vtag, p in variants:
            collapse = collapse_values(sigma.astype(np.float64), p).astype(
                np.float32)
            vic, crushed = model.ep_first_crossing_counts(
                radii, cov, c["barrier"], int(c["n_paths"]),
                seed=stored_seed(tag), collapse=collapse,
                return_collapse_counts=True)
            for kk, vv in c.items():
                if kk != "tag":
                    store[f"{vtag}__{kk}"] = vv
            store[f"{vtag}__counts_vic"] = vic
            store[f"{vtag}__counts_crushed"] = crushed
            store[f"{vtag}__collapse"] = collapse
            for kk, vv in p.items():
                store[f"{vtag}__{kk}"] = np.float64(vv)
            print(f"{vtag} dc={p['delta_c']:.3f} be={p['beta_e']:.2f} voids "
                  f"kept {vic.sum() / c['counts'].sum():.4f}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **store)


def stage_analyse(out_dir, weights, weights_vic, rmax):
    import pysif.model as model
    from ep_model import (EPEmulator, VICEmulator, compute_features,
                          mc_hazard, multiplicity_from_hazard, vic_candidates)
    emu, vemu = EPEmulator.load(weights), VICEmulator.load(weights_vic)
    d = np.load(SPECTRA)
    k = np.ascontiguousarray(d["k"])
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.npz"))):
        z = np.load(path)
        for tag in sorted({key.split("__")[0] for key in z.files}):
            c = {key.split("__", 1)[1]: z[key] for key in z.files
                 if key.startswith(tag + "__")}
            radii = c["radii"]
            pk = d["fid"] if tag[0] == "T" else d[f"v{int(c['cosmo'])}"]
            cov, _, _, _ = model.delta_covariance_pk(
                k, np.ascontiguousarray(pk), radii)
            S, B = c["S"].astype(float), c["barrier"].astype(float)
            dv = c["deriv_variance"].astype(float)
            X, lam_up, nul = compute_features(radii.astype(float), S, B, dv)
            f_one = multiplicity_from_hazard(lam_up * np.exp(emu.correction(X)),
                                             radii.astype(float), nul)
            cand, lam_cond = vic_candidates(radii.astype(float), S, B,
                                            c["collapse"].astype(float), dv,
                                            cov)
            ln_k = vemu.ln_k(vemu.inputs(cand), lam_cond)
            n = float(c["n_paths"])
            dr = np.diff(radii.astype(float))
            vic = c["counts_vic"][:-1].astype(float)
            one = c["counts"][:-1].astype(float)
            _, rel, _ = mc_hazard(c["counts_vic"], int(n))
            rm = 0.5 * (radii[1:] + radii[:-1]).astype(float)
            m = (vic > 0) & (rel < 0.05) & (rm < rmax)
            if "r_lo" in c:
                m &= rm >= float(c["r_lo"])
            f_mc = vic / (n * dr)
            with np.errstate(divide="ignore", invalid="ignore"):
                rows.append(dict(
                    tag=tag, set=tag[0],
                    e=(f_one * np.exp(ln_k) / f_mc - 1.0)[m],
                    e_novic=(f_one / f_mc - 1.0)[m],
                    dlnk=(ln_k - np.log(vic / one))[m & (one > 0)],
                    noise=1.0 / np.sqrt(vic[m]),
                    kept=vic.sum() / max(one.sum(), 1.0),
                    delta_nl=float(c.get("delta_nl", np.nan)),
                    beta_e=float(c["beta_e"])))

    def line(label, rs):
        if not rs:
            return
        e = np.abs(np.concatenate([r["e"] for r in rs]))
        e0 = np.abs(np.concatenate([r["e_novic"] for r in rs]))
        dk = np.abs(np.concatenate([r["dlnk"] for r in rs]))
        nz = np.concatenate([r["noise"] for r in rs])
        print(f"  {label:<32}{len(e):>6}{np.median(e):>9.3%}"
              f"{np.percentile(e, 95):>8.2%}{np.median(e0):>10.3%}"
              f"{np.median(dk):>11.2e}{np.percentile(dk, 95):>10.2e}"
              f"{np.median(nz):>8.3%}")
    print(f"  {'set, R < %g' % rmax:<32}{'bins':>6}{'median':>9}{'95th':>8}"
          f"{'no VIC':>10}{'|dlnK|':>11}{'95th':>10}{'noise':>8}")
    line("V generic", [r for r in rows if r["set"] == "V"])
    line("W far origin", [r for r in rows if r["set"] == "W"])
    for m, lab in ((0, "delta_c 1.686"), (1, "ellipsoidal")):
        ts = [r for r in rows if r["set"] == "T" and r["tag"].endswith(f"c{m}")]
        line(f"T thesis, {lab}", ts)
        for r in ts:
            line(f"  dNL {r['delta_nl']:+.2f} (kept {r['kept']:.4f})", [r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("mc", "analyse"))
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--out", default="val_vic")
    ap.add_argument("--val", default=VAL_V2)
    ap.add_argument("--weights", default=os.path.join(HERE, "ep_emulator.npz"))
    ap.add_argument("--weights-vic",
                    default=os.path.join(HERE, "ep_emulator_vic.npz"))
    ap.add_argument("--rmax", type=float, default=80.0)
    args = ap.parse_args()
    if args.stage == "mc":
        stage_mc(args.task, args.out, args.val)
    else:
        stage_analyse(args.out, args.weights, args.weights_vic, args.rmax)


if __name__ == "__main__":
    main()
