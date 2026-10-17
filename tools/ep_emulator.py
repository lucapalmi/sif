#!/usr/bin/env python
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""
The reference emulator: a small weighted MLP predicting the log hazard ratio.

Written out rather than taken from sklearn for two reasons. The fit needs
per-bin inverse-variance weights -- the Monte Carlo error varies by a factor of
twenty-five across the nu range, and treating a 0.05% bin and a 1.3% bin alike
would throw away most of what the long runs bought. And this is the definition
the C implementation has to reproduce bit for bit, so it is worth having it in
one readable place with no library between the equations and the weights.

Architecture: standardise, two tanh layers, linear output. Small on purpose --
three smooth inputs and a target that is nearly one-dimensional in nu do not
need capacity, and every parameter here becomes a constant in the C source.

The target is log(Lambda_MC / Lambda_up), so the correction is exp(prediction)
and is positive by construction. It is applied to the HAZARD, not to the
multiplicity, so the survival recursion keeps the result in [0, 1] whatever the
network does.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import minimize

# Rows per block in the loss. The cost is the elementwise tanh and its
# derivative over n_rows x width arrays, which numpy runs on one thread and
# from main memory; in blocks of this size they stay in cache and run on every
# core at once (numpy releases the GIL in ufuncs and matmul). Blocks are summed
# in a fixed order, so the result does not depend on the thread count. Run it
# with the BLAS single-threaded (ep_train_final.py does): a threaded BLAS under
# these threads oversubscribes the cores and was measured to stall.
BLOCK_ROWS = 8192


# The negexp link's output is capped before the exponential: K = e^-e^10 is
# already zero to any precision, and the cap keeps a wild early step finite.
LINK_Z_MAX = 10.0


class EmulatorMLP:

    def __init__(self, hidden=(16, 16), seed=0, l2=1e-6, link="identity"):
        self.hidden = tuple(hidden)
        self.seed = seed
        self.l2 = l2
        # "identity": the output is the target. "negexp": the target is
        # -scale exp(output), for a quantity that is never positive and
        # reaches zero -- ln K of void-in-cloud, where K <= 1 and is 1
        # wherever nothing collapses -- with `scale` a per-row baseline that
        # the network corrects multiplicatively (1 without one). The loss
        # stays on the target either way.
        if link not in ("identity", "negexp"):
            raise ValueError(f"link: {link!r} (expected 'identity' or "
                             f"'negexp')")
        self.link = link
        self.mu_ = None
        self.sd_ = None
        self.shapes_ = None
        self.theta_ = None

    # --- parameter packing ---------------------------------------------------

    def _shapes(self, n_in):
        dims = (n_in,) + self.hidden + (1,)
        return [((dims[i], dims[i + 1]), (dims[i + 1],))
                for i in range(len(dims) - 1)]

    def _unpack(self, theta):
        out = []
        at = 0
        for (w_shape, b_shape) in self.shapes_:
            n_w = w_shape[0] * w_shape[1]
            n_b = b_shape[0]
            W = theta[at:at + n_w].reshape(w_shape)
            at += n_w
            b = theta[at:at + n_b]
            at += n_b
            out.append((W, b))
        return out

    def _n_params(self):
        return sum(w[0] * w[1] + b[0] for (w, b) in self.shapes_)

    # --- forward and gradient ------------------------------------------------

    def _forward(self, layers, X, scale=None):
        """Returns the output and the per-layer activations for the backward."""
        acts = [X]
        h = X
        for i, (W, b) in enumerate(layers):
            z = h @ W + b
            h = np.tanh(z) if i < len(layers) - 1 else z
            acts.append(h)
        out = h.ravel()
        if self.link == "negexp":
            out = -np.exp(np.minimum(out, LINK_Z_MAX))
            if scale is not None:
                out = out * scale
        return out, acts

    def _block_loss_grad(self, layers, X, y, w, w_total, scale=None):
        """Unregularised loss and flat gradient over one block of rows."""
        pred, acts = self._forward(layers, X, scale)

        resid = pred - y
        loss = 0.5 * np.sum(w * resid ** 2) / w_total

        # dL/dpred, then through the link: d(-e^z)/dz = -e^z = pred
        g = w * resid / w_total
        if self.link == "negexp":
            # zero past the cap, where the output no longer moves with z: a
            # gradient that ignored it disagreed with the loss and stopped the
            # line search within ~100 iterations
            g = np.where(acts[-1].ravel() < LINK_Z_MAX, g * pred, 0.0)
        g = g[:, None]

        grads = [None] * len(layers)
        for i in range(len(layers) - 1, -1, -1):
            W, b = layers[i]
            h_in = acts[i]
            grads[i] = (h_in.T @ g, g.sum(axis=0))
            if i > 0:
                g = (g @ W.T) * (1.0 - acts[i] ** 2)

        return loss, np.concatenate([np.concatenate([gw.ravel(), gb.ravel()])
                                     for (gw, gb) in grads])

    def _loss_grad(self, theta, X, y, w, scale=None):
        layers = self._unpack(theta)
        w_total = np.sum(w)
        blocks = [slice(i, i + BLOCK_ROWS)
                  for i in range(0, len(y), BLOCK_ROWS)]
        parts = list(self._pool.map(
            lambda sl: self._block_loss_grad(
                layers, X[sl], y[sl], w[sl], w_total,
                None if scale is None else scale[sl]), blocks))

        loss = sum(p[0] for p in parts)
        flat = np.sum([p[1] for p in parts], axis=0)
        loss += 0.5 * self.l2 * np.dot(theta, theta)
        flat += self.l2 * theta
        return loss, flat

    # --- fitting -------------------------------------------------------------

    def fit(self, X, y, w=None, maxiter=600, restarts=1, scale=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
        if scale is not None:
            scale = np.asarray(scale, dtype=float)

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_

        self.shapes_ = self._shapes(X.shape[1])
        n = self._n_params()

        best = None
        self._pool = ThreadPoolExecutor(
            int(os.environ.get("EP_FIT_THREADS", os.cpu_count() or 1)))
        for r in range(restarts):
            rng = np.random.default_rng(self.seed + r)
            # Modest init: tanh saturates and the gradient dies otherwise.
            theta0 = rng.normal(0.0, 0.5, n) / np.sqrt(max(X.shape[1], 1))
            # ftol = gtol = 0: stop on maxiter alone. SciPy's default relative
            # tolerance (~2e-9) ends the fit after a few hundred iterations
            # here, because the loss is ~1e-5 and each step's gain looks
            # negligible relative to it -- measured: every network stopped
            # near iteration 660 whatever maxiter said, at twice the training
            # loss a full run reaches.
            res = minimize(self._loss_grad, theta0, args=(Xs, y, w, scale),
                           jac=True, method="L-BFGS-B",
                           options=dict(maxiter=maxiter, maxfun=maxiter * 2,
                                        ftol=0.0, gtol=0.0))
            if best is None or res.fun < best.fun:
                best = res
        self._pool.shutdown()
        del self._pool

        self.theta_ = best.x
        self.loss_ = best.fun
        # Kept so a fit that stopped early says so: a line search that fails
        # ends L-BFGS long before maxiter, and the scores look merely worse.
        self.nit_ = int(best.nit)
        self.message_ = str(best.message)
        return self

    def predict(self, X, scale=None):
        Xs = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        pred, _ = self._forward(self._unpack(self.theta_), Xs, scale)
        return pred

    # --- export --------------------------------------------------------------

    def weights(self):
        """(mu, sd, [(W, b), ...]) -- everything the C side needs."""
        return self.mu_.copy(), self.sd_.copy(), [
            (W.copy(), b.copy()) for (W, b) in self._unpack(self.theta_)]

    def n_parameters(self):
        return len(self.theta_)


def build_matrix(curves, feats, mask_key="mask"):
    """Feature matrix, target and weights, pooled over curves."""
    X, y, w, meta = [], [], [], []
    for c in curves:
        m = c[mask_key]
        if m.sum() == 0:
            continue
        X.append(np.column_stack([c[f][m] for f in feats]))
        y.append(np.log(c["ratio"][m]))
        # sigma on log(Lambda_MC) is the relative error on Lambda_MC itself.
        w.append(1.0 / np.maximum(c["rel_err"][m], 1e-6) ** 2)
        meta.append(np.column_stack([c["nu"][m],
                                     np.full(m.sum(), c["task"])]))
    return (np.vstack(X), np.concatenate(y), np.concatenate(w),
            np.vstack(meta))
