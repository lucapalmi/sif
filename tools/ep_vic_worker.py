# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Void-in-cloud training set: replay stored curves with a collapse barrier.

One SLURM task = one stored trainset file (one cosmology of one campaign).
Every curve is walked again with its stored radii, barrier and seed plus a
collapse barrier from draw_collapse, so its void counts are a subset of the
stored ones, walk for walk. --verify first walks a curve WITHOUT the collapse
barrier and measures how many crossings moved against the stored counts: the
pairing rests on that being negligible, and a library change could break it
silently.

    python ep_vic_worker.py --runs runs_v2 --task-id 7 --spectra \\
        tools/ep_spectra_v2.npz --out runs_vic/v2 [--verify]
"""

import argparse
import os
import sys
import time

import numpy as np

import pysif.model as model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ep_trainset_design import collapse_values, draw_collapse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="stored trainset directory")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--spectra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-paths", type=int, default=None,
                    help="default: the stored curves' own, for exact pairing")
    ap.add_argument("--verify", action="store_true",
                    help="replay the first curve without collapse and compare")
    ap.add_argument("--max-curves", type=int, default=None)
    args = ap.parse_args()

    src = os.path.join(args.runs, f"trainset_{args.task_id:03d}.npz")
    d = np.load(src)
    sp = np.load(args.spectra)
    k = np.ascontiguousarray(sp["k"])
    pk = np.ascontiguousarray(sp["pk"][args.task_id])
    campaign = int(d["campaign"]) if "campaign" in d.files else 1
    n_paths = int(args.n_paths or d["n_paths"])
    paired = n_paths == int(d["n_paths"])

    tags = sorted({key.split("__")[0] for key in d.files
                   if key.startswith("b") and "__" in key})
    if args.max_curves:
        tags = tags[:args.max_curves]

    store = {}
    t0 = time.perf_counter()
    for t in tags:
        radii = d[f"{t}__radii"]
        barrier = d[f"{t}__barrier"]
        seed = int(d[f"{t}__seed"])
        cov, sigma, _, _ = model.delta_covariance_pk(k, pk, radii)

        if args.verify and t == tags[0]:
            # Not bit-identity: the covariance is recomputed, and a rounding
            # difference moves the walk by ~1e-16 (1e-7 for campaign 1's older
            # build), which changes the crossing radius only of a walk passing
            # that close to the barrier. What pairing needs is that those are
            # a negligible share: the moved counts, as a fraction of the total.
            stored = d[f"{t}__counts"].astype(np.int64)
            again = model.ep_first_crossing_counts(
                radii, cov, barrier, n_paths, seed=seed).astype(np.int64)
            moved = 0.5 * np.abs(again - stored).sum() / max(stored.sum(), 1)
            print(f"[verify] {src} {t}: replay without collapse, "
                  f"{moved:.2e} of the crossings moved", flush=True)
            if moved > 1e-4:
                sys.exit(f"[verify] {src} {t}: the replay does not pair with "
                         f"the stored counts ({moved:.2e} moved, expected "
                         f"< 1e-4)")

        p = draw_collapse(campaign, args.task_id, int(t[1:]))
        collapse = collapse_values(sigma.astype(np.float64), p).astype(
            np.float32)
        ts = time.perf_counter()
        vic, crushed = model.ep_first_crossing_counts(
            radii, cov, barrier, n_paths, seed=seed, collapse=collapse,
            return_collapse_counts=True)
        dt = time.perf_counter() - ts

        store[f"{t}__counts_vic"] = vic
        store[f"{t}__counts_crushed"] = crushed
        store[f"{t}__collapse"] = collapse
        for key, val in p.items():
            store[f"{t}__{key}"] = np.float64(val)
        one = d[f"{t}__counts"].sum() if paired else np.nan
        print(f"    {t} dc={p['delta_c']:.3f} be={p['beta_e']:.2f} "
              f"ge={p['gamma_e']:.2f} | voids kept {vic.sum() / one:.4f} "
              f"crushed {crushed.sum() / n_paths:.3f} | {dt:.0f}s", flush=True)

    store["n_paths"] = np.uint64(n_paths)
    store["paired"] = np.uint8(paired)
    store["source"] = np.array(os.path.abspath(src))
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"vic_{args.task_id:03d}.npz")
    np.savez_compressed(out, **store)
    print(f"[task {args.task_id:03d}] {len(tags)} curves in "
          f"{(time.perf_counter() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
