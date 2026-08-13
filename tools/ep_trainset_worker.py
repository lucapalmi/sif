#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
One SLURM array task: every barrier of a single cosmology.

The cosmology is fixed per task so that the covariance -- which the walk needs
factorized anyway -- is computed once and reused across all of that
cosmology's barriers, and so that a failed task loses one cosmology rather
than a slice of all of them.

Needs only numpy and pysif on the compute node: the spectra arrive as a table
from ep_make_spectra.py, and the design is regenerated from its seed.

Usage:
  python ep_trainset_worker.py --task-id 0 --spectra ep_spectra.npz --out runs/
  python ep_trainset_worker.py --task-id 0 --spectra ... --benchmark
"""

import argparse
import os
import sys
import time

import numpy as np

import pysif.model as model
from ep_trainset_design import (N_BARRIER, N_PATHS, N_RADII, SEED,
                                barrier_values, build_cosmologies,
                                draw_barriers, radii_for_curve)

# Radii used only to locate the nu and sigma windows; never the walk's grid.
SCAN = np.logspace(-2.5, 2.9, 400).astype(np.float32)


def run_one(k, pk, sigma_scan, params, n_paths, seed):
    """One curve: returns None if its windows do not intersect."""
    radii = radii_for_curve(sigma_scan, SCAN, params)
    if radii is None:
        return None

    cov, sigma, high_k, dvar = model.delta_covariance_pk(k, pk, radii)
    barrier = barrier_values(sigma.astype(np.float64), params).astype(
        np.float32)

    t0 = time.perf_counter()
    counts = model.first_crossing_counts_ep(
        radii, cov, barrier, n_paths=n_paths, seed=seed)
    dt = time.perf_counter() - t0

    n = len(radii)
    idx = np.arange(n)
    S = cov[idx * (idx + 1) // 2 + idx]

    return dict(radii=radii, sigma=sigma, deriv_variance=dvar,
                barrier=barrier, counts=counts, S=S,
                high_k_fraction=high_k, seconds=dt,
                crossed=counts.sum() / n_paths, dropped=counts[-1] / n_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True,
                    help="array index; one cosmology per task")
    ap.add_argument("--spectra", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--n-paths", type=int, default=N_PATHS)
    ap.add_argument("--n-barrier", type=int, default=N_BARRIER)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--benchmark", action="store_true",
                    help="one short curve, report throughput, write nothing")
    args = ap.parse_args()

    d = np.load(args.spectra)
    k = d["k"]
    pk_all = d["pk"]
    n_cosmo = pk_all.shape[0]

    if not 0 <= args.task_id < n_cosmo:
        sys.exit(f"task id {args.task_id} outside 0..{n_cosmo - 1}")

    cosmologies = build_cosmologies(n_cosmo=n_cosmo, seed=args.seed)

    i = args.task_id
    pk = np.ascontiguousarray(pk_all[i])
    params = cosmologies[i]

    _, sigma_scan, _, _ = model.delta_covariance_pk(k, pk, SCAN)

    barriers, n_proposed = draw_barriers(
        sigma_scan, SCAN, i, n_barrier=args.n_barrier, seed=args.seed)

    if len(barriers) < args.n_barrier:
        print(f"[task {i:03d}] only {len(barriers)} of {args.n_barrier} "
              f"barriers were reachable in {n_proposed} draws; this cosmology "
              f"puts most of the barrier box out of reach at physical radii",
              flush=True)

    if args.benchmark:
        n_probe = 2_000_000
        # The most expensive curve of the task, not the first: a walk that
        # rarely crosses steps through every radius, so timing an easy curve
        # would under-predict the task by a factor of two.
        worst = max(barriers, key=lambda b: b["alpha"])
        r = run_one(k, pk, sigma_scan, worst, n_probe, args.seed)
        rate = n_probe / r["seconds"]
        total = n_cosmo * args.n_barrier * args.n_paths
        print(f"throughput {rate:.3e} paths/s at {N_RADII} radii "
              f"({os.environ.get('OMP_NUM_THREADS', '?')} threads), "
              f"worst-case curve (crossed {r['crossed']:.3f})")
        print(f"acceptance {len(barriers)}/{n_proposed} draws reachable")
        print(f"per task   {args.n_barrier * args.n_paths / rate / 60:.1f} min "
              f"(upper bound)")
        print(f"whole set  {total / rate / 3600:.2f} node-hours "
              f"({total:.3e} paths)")
        return

    print(f"[task {i:03d}] omch2={params['omch2']:.4f} h={params['H0'] / 100:.3f} "
          f"ns={params['ns']:.3f} mnu={params['mnu']:.3f} z={params['z']:.3f}",
          flush=True)

    store = {}
    n_dropped = 0
    t_start = time.perf_counter()

    for j, bp in enumerate(barriers):
        # Seeds are a pure function of (cosmology, barrier), so a rerun of one
        # task reproduces its curves bit for bit.
        seed = args.seed + 7_000_000 + i * args.n_barrier + j
        r = run_one(k, pk, sigma_scan, bp, args.n_paths, seed)

        if r is None:
            n_dropped += 1
            print(f"    barrier {j:02d} dropped: nu and sigma windows do not "
                  f"intersect (alpha={bp['alpha']:.3f} beta={bp['beta']:.3f} "
                  f"gamma={bp['gamma']:.3f})", flush=True)
            continue

        tag = f"b{j:02d}"
        for key in ("radii", "sigma", "deriv_variance", "barrier", "counts",
                    "S", "high_k_fraction"):
            store[f"{tag}__{key}"] = r[key]
        for key in ("alpha", "beta", "gamma", "perturb_seed"):
            store[f"{tag}__{key}"] = np.float64(bp[key])
        store[f"{tag}__seed"] = np.uint64(seed)

        print(f"    barrier {j:02d} a={bp['alpha']:.3f} b={bp['beta']:.3f} "
              f"g={bp['gamma']:.3f} | crossed {r['crossed']:.3f} "
              f"dropped {r['dropped']:.1e} maxhighk {r['high_k_fraction'].max():.1e} "
              f"| {r['seconds']:.0f}s", flush=True)

    for key, val in params.items():
        store[f"cosmo__{key}"] = np.float64(val)
    store["n_paths"] = np.uint64(args.n_paths)
    store["n_barrier"] = np.uint32(len(barriers))
    store["n_dropped"] = np.uint32(n_dropped)
    store["n_proposed"] = np.uint32(n_proposed)
    store["design_seed"] = np.uint64(args.seed)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"trainset_{i:03d}.npz")
    np.savez_compressed(path, **store)

    print(f"[task {i:03d}] {len(barriers) - n_dropped} curves in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min -> {path}",
          flush=True)


if __name__ == "__main__":
    main()
