#!/usr/bin/env python
"""
Quality assurance on the training set, before any fitting.

Five things are checked here, in order of how badly a failure would matter:

1. Integrity -- every curve present, counts consistent with n_paths, no
   first-step mass large enough to distort the bins below it.
2. The noise floor actually achieved, against what 1e9 paths should give. This
   sets the accuracy ceiling for everything downstream, so it is worth knowing
   rather than assuming.
3. Feature coverage, against the design. A gap here is a region the emulator
   will extrapolate into.
4. The high-nu asymptote: does the hazard ratio approach 1? The whole envelope
   architecture rests on this, and it was established at 2e7 paths where the
   tail was noisy. At 1e9 it can be tested properly.
5. Smoothness of the target in nu, curve by curve. A ratio that jitters beyond
   its own error bar is a sign the hazard extraction is wrong, not that the
   physics is rough.

Usage:  python tools/ep_qa.py [runs_dir] [--plot out.png]
"""

import argparse

import numpy as np

from ep_trainset_load import load_runs

MAX_REL_ERR = 0.05


def integrity(curves):
    print(f"\n{'=' * 74}\n1. integrity\n{'=' * 74}")

    bad_counts = bad_finite = 0
    worst_drop = 0.0
    drop_curve = None

    for c in curves:
        total = c["counts"].sum()
        if total > c["n_paths"]:
            bad_counts += 1
        if not np.all(np.isfinite(c["lam_up"])):
            bad_finite += 1
        drop = c["counts"][-1] / c["n_paths"]
        if drop > worst_drop:
            worst_drop, drop_curve = drop, (c["task"], c["barrier_tag"])

    crossed = np.array([c["counts"].sum() / c["n_paths"] for c in curves])

    print(f"  curves                    {len(curves)}")
    print(f"  counts exceeding n_paths  {bad_counts}")
    print(f"  non-finite baseline rate  {bad_finite}")
    print(f"  crossing fraction         min {crossed.min():.3f}, "
          f"median {np.median(crossed):.3f}, max {crossed.max():.3f}")
    print(f"  worst first-step loss     {worst_drop:.2e} "
          f"(task {drop_curve[0]}, {drop_curve[1]})")
    if worst_drop > 0.01:
        print("    WARNING: above 1%, the largest-radius bin is distorted")


def noise_floor(curves):
    print(f"\n{'=' * 74}\n2. noise floor achieved\n{'=' * 74}")

    err = np.concatenate([c["rel_err"][c["mask"]] for c in curves])
    nu = np.concatenate([c["nu"][c["mask"]] for c in curves])
    n_paths = curves[0]["n_paths"]

    print(f"  n_paths/curve             {n_paths:.3e}")
    print(f"  usable bins               {err.size} of "
          f"{sum(c['nu'].size for c in curves)}")
    print(f"  relative error on Lambda  median {np.median(err):.2e}, "
          f"RMS {np.sqrt(np.mean(err ** 2)):.2e}")

    # The pilot ran 2e7 and measured 3.1e-3 median; the floor should fall as
    # 1/sqrt(n_paths), so this is a direct check that the extra paths bought
    # what they were supposed to.
    expected = 3.1e-3 * np.sqrt(2e7 / n_paths)
    print(f"  predicted from the pilot  {expected:.2e} "
          f"(ratio {np.median(err) / expected:.2f})")

    for lo, hi in [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 9.0)]:
        m = (nu >= lo) & (nu < hi)
        if m.sum():
            print(f"    nu {lo:.0f}-{hi:.0f}: {m.sum():>6} bins, "
                  f"median error {np.median(err[m]):.2e}")


def coverage(curves):
    print(f"\n{'=' * 74}\n3. feature coverage\n{'=' * 74}")

    feats = {k: np.concatenate([c[k][c["mask"]] for c in curves])
             for k in ("nu", "gamma2", "y")}

    for k, v in feats.items():
        print(f"  {k:<8} {v.min():>9.4f} - {v.max():<9.4f}  "
              f"(1-99 pct {np.percentile(v, 1):>8.4f} - "
              f"{np.percentile(v, 99):.4f})")

    # Gaps are what matter, not endpoints: a hole in the interior is a region
    # the emulator interpolates blind.
    for k, v in feats.items():
        h, edges = np.histogram(v, bins=40)
        empty = np.where(h == 0)[0]
        interior = [i for i in empty if h[:i].sum() > 0 and h[i + 1:].sum() > 0]
        if interior:
            spans = ", ".join(f"[{edges[i]:.3f},{edges[i + 1]:.3f}]"
                              for i in interior[:5])
            print(f"  GAP in {k}: {len(interior)} empty interior bins {spans}")
        else:
            print(f"  {k:<8} no interior gaps over 40 bins")


def asymptote(curves):
    print(f"\n{'=' * 74}\n4. the high-nu asymptote (the envelope assumption)\n"
          f"{'=' * 74}")
    print("   if the correction does not go to 1, an envelope forced to 1 is "
          "wrong")

    nu = np.concatenate([c["nu"][c["mask"]] for c in curves])
    r = np.concatenate([c["ratio"][c["mask"]] for c in curves])
    e = np.concatenate([c["rel_err"][c["mask"]] for c in curves])

    print(f"\n{'nu bin':>12} {'bins':>7} {'median r':>10} {'r - 1':>10} "
          f"{'err on mean':>12} {'sigma from 1':>13}")
    print("-" * 70)

    for lo, hi in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0),
                   (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 9.0)]:
        m = (nu >= lo) & (nu < hi)
        if m.sum() < 20:
            continue
        med = np.median(r[m])
        # Error on the median of n independent noisy bins, roughly.
        sem = np.median(e[m]) / np.sqrt(m.sum()) * 1.253
        print(f"  {lo:.1f} - {hi:<4.1f} {m.sum():>7} {med:>10.5f} "
              f"{med - 1:>+10.5f} {sem:>12.2e} {(med - 1) / sem:>+13.1f}")

    print("\n   a large sigma at high nu is not necessarily a problem: it can")
    print("   mean the approach to 1 is real but slower than the envelope")
    print("   assumed. The size of r-1 is what matters, not its significance.")


def smoothness(curves):
    print(f"\n{'=' * 74}\n5. smoothness of the target\n{'=' * 74}")
    print("   second difference of ln(ratio) along each curve, in units of its")
    print("   own error: ~1 means the scatter is Monte Carlo noise and nothing")
    print("   else; >>1 means structure the extraction is inventing")

    pulls = []
    for c in curves:
        m = c["mask"]
        if m.sum() < 12:
            continue
        lr = np.log(c["ratio"][m])
        er = c["rel_err"][m]
        d2 = lr[2:] - 2 * lr[1:-1] + lr[:-2]
        sd = np.sqrt(er[2:] ** 2 + 4 * er[1:-1] ** 2 + er[:-2] ** 2)
        pulls.append(d2 / sd)

    p = np.concatenate(pulls)
    print(f"\n  bins            {p.size}")
    print(f"  RMS pull        {np.sqrt(np.mean(p ** 2)):.3f}")
    print(f"  |pull| > 5      {100 * np.mean(np.abs(p) > 5):.2f}%")
    if np.sqrt(np.mean(p ** 2)) > 3:
        print("    the target carries real curvature beyond its noise, so a "
              "smoothing prior would bias it")


def make_plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nu = np.concatenate([c["nu"][c["mask"]] for c in curves])
    r = np.concatenate([c["ratio"][c["mask"]] for c in curves])
    g2 = np.concatenate([c["gamma2"][c["mask"]] for c in curves])
    y = np.concatenate([c["y"][c["mask"]] for c in curves])
    e = np.concatenate([c["rel_err"][c["mask"]] for c in curves])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    s = np.random.default_rng(0).choice(nu.size, min(60000, nu.size),
                                        replace=False)
    sc = ax[0].scatter(nu[s], r[s], c=y[s], s=2, alpha=0.3, cmap="viridis")
    ax[0].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[0].set_xlabel(r"$\nu$")
    ax[0].set_ylabel(r"$\Lambda_{\rm MC}/\Lambda_{\rm up}$")
    ax[0].set_title(f"target, {nu.size} usable bins")
    fig.colorbar(sc, ax=ax[0], label="y")
    ax[0].grid(alpha=0.3)

    edges = np.linspace(nu.min(), min(nu.max(), 5.0), 40)
    cen, med, q16, q84 = [], [], [], []
    for i in range(len(edges) - 1):
        m = (nu >= edges[i]) & (nu < edges[i + 1])
        if m.sum() > 20:
            cen.append(0.5 * (edges[i] + edges[i + 1]))
            med.append(np.median(r[m]))
            q16.append(np.percentile(r[m], 16))
            q84.append(np.percentile(r[m], 84))
    ax[1].fill_between(cen, q16, q84, alpha=0.3, color="steelblue",
                       label="16-84 pct")
    ax[1].plot(cen, med, "-", color="steelblue", lw=1.5, label="median")
    ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel(r"$\nu$")
    ax[1].set_ylabel("hazard ratio")
    ax[1].set_title("approach to the asymptote")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].semilogy(nu[s], e[s], ".", ms=1, alpha=0.2, color="crimson")
    ax[2].set_xlabel(r"$\nu$")
    ax[2].set_ylabel("relative error on $\\Lambda_{\\rm MC}$")
    ax[2].set_title("noise floor vs barrier height")
    ax[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nplot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--plot")
    args = ap.parse_args()

    curves = load_runs(args.runs, max_rel_err=MAX_REL_ERR)
    if not curves:
        raise SystemExit(f"no trainset_*.npz under {args.runs}")

    integrity(curves)
    noise_floor(curves)
    coverage(curves)
    asymptote(curves)
    smoothness(curves)

    if args.plot:
        make_plot(curves, args.plot)


if __name__ == "__main__":
    main()
