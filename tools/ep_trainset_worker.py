#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
One SLURM array task: every barrier of a single cosmology.

The cosmology is fixed per task so that the covariance -- which the walk needs
factorised anyway -- is computed once and reused across all of that
cosmology's barriers, and so that a failed task loses one cosmology rather
than a slice of all of them.

Needs only numpy and pysif on the compute node: the spectra arrive as a table
from ep_make_spectra.py, and the design is regenerated from its seed.

Usage:
  python ep_trainset_worker.py --task-id 0 --spectra ep_spectra.npz --out runs/
  python ep_trainset_worker.py --task-id 0 --spectra ... --benchmark
  python ep_trainset_worker.py --campaign 2 --task-id 0 \
      --spectra ep_spectra_v2.npz --out runs_v2/
"""

import argparse
import os
import sys
import time

import numpy as np

import pysif.model as model
from ep_trainset_design import (N_BARRIER, N_BARRIER_2, N_BARRIER_3,
                                N_BARRIER_4, N_PATHS, N_RADII, SEED, SEED_2,
                                SEED_3, SEED_4, barrier_values,
                                build_cosmologies, draw_barriers,
                                draw_barriers_2, draw_barriers_3,
                                draw_barriers_4, radii_for_curve,
                                radii_from_origin)

# Radii used only to locate the nu and sigma windows; never the walk's grid.
SCAN = np.logspace(-2.5, 2.9, 400).astype(np.float32)

# Campaign 2 starts walks out to 700 Mpc/h, so its scan reaches 1000.
SCAN_2 = np.logspace(-2.5, 3.0, 500).astype(np.float32)


def run_one(k, pk, sigma_scan, params, n_paths, seed, campaign=1):
    """One curve: returns None if its windows do not intersect."""
    if campaign in (2, 3, 4):
        res = radii_from_origin(sigma_scan, SCAN_2, params,
                                params["r_origin"], params["nu_origin"])
        radii = None if res is None else res[0]
    else:
        radii = radii_for_curve(sigma_scan, SCAN, params)
    if radii is None:
        return None

    cov, sigma, high_k, dvar = model.delta_covariance_pk(k, pk, radii)
    barrier = barrier_values(sigma.astype(np.float64), params).astype(
        np.float32)

    t0 = time.perf_counter()
    counts = model.ep_first_crossing_counts(
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
    ap.add_argument("--n-barrier", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--campaign", type=int, choices=(1, 2, 3, 4), default=1,
                    help="1: nu-window design (runs/); 2: origin-placed "
                         "(runs_v2/); 3: far origins, long walks (runs_v3/); "
                         "4: origins 30-120 (runs_v4/)")
    ap.add_argument("--benchmark", action="store_true",
                    help="one short curve, report throughput, write nothing")
    args = ap.parse_args()

    d = np.load(args.spectra)
    k = d["k"]
    pk_all = d["pk"]
    n_cosmo = pk_all.shape[0]

    if not 0 <= args.task_id < n_cosmo:
        sys.exit(f"task id {args.task_id} outside 0..{n_cosmo - 1}")

    two = args.campaign in (2, 3, 4)
    if args.n_barrier is None:
        args.n_barrier = {1: N_BARRIER, 2: N_BARRIER_2, 3: N_BARRIER_3,
                          4: N_BARRIER_4}[args.campaign]
    if args.seed is None:
        args.seed = {1: SEED, 2: SEED_2, 3: SEED_3,
                     4: SEED_4}[args.campaign]

    # Cosmologies always come from campaign 1's seed: both campaigns share
    # them, so a cosmology can be held out of both at once.
    cosmologies = build_cosmologies(n_cosmo=n_cosmo, seed=SEED)

    i = args.task_id
    pk = np.ascontiguousarray(pk_all[i])
    params = cosmologies[i]

    scan = SCAN_2 if two else SCAN
    _, sigma_scan, _, _ = model.delta_covariance_pk(k, pk, scan)

    if two:
        draw = {2: draw_barriers_2, 3: draw_barriers_3,
                4: draw_barriers_4}[args.campaign]
        barriers = draw(sigma_scan, scan, i, n_barrier=args.n_barrier,
                        seed=args.seed)
        n_proposed = len(barriers)
    else:
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
        r = run_one(k, pk, sigma_scan, worst, n_probe, args.seed,
                    args.campaign)
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
        r = run_one(k, pk, sigma_scan, bp, args.n_paths, seed, args.campaign)

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
        for key in ("alpha", "beta", "gamma", "perturb_seed", "r_origin",
                    "nu_origin"):
            if key in bp:
                store[f"{tag}__{key}"] = np.float64(bp[key])
        store[f"{tag}__seed"] = np.uint64(seed)

        where = (f"R_o={bp['r_origin']:.1f} nu_o={bp['nu_origin']:.2f} "
                 if "r_origin" in bp else "")
        print(f"    barrier {j:02d} {where}a={bp['alpha']:.3g} b={bp['beta']:.3f} "
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
    store["campaign"] = np.uint32(args.campaign)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"trainset_{i:03d}.npz")
    np.savez_compressed(path, **store)

    print(f"[task {i:03d}] {len(barriers) - n_dropped} curves in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min -> {path}",
          flush=True)


if __name__ == "__main__":
    main()
