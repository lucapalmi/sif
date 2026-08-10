#!/usr/bin/env python
"""
The reference emulator: a small weighted MLP predicting the log hazard ratio.

Written out rather than taken from sklearn for two reasons. The fit needs
per-bin inverse-variance weights -- the Monte Carlo error varies by a factor of
twenty-five across the nu range, and treating a 0.05% bin and a 1.3% bin alike
would throw away most of what the long runs bought. And this is the definition
the C implementation has to reproduce bit for bit, so it is worth having it in
one readable place with no library between the equations and the weights.

Architecture: standardize, two tanh layers, linear output. Small on purpose --
three smooth inputs and a target that is nearly one-dimensional in nu do not
need capacity, and every parameter here becomes a constant in the C source.

The target is log(Lambda_MC / Lambda_up), so the correction is exp(prediction)
and is positive by construction. It is applied to the HAZARD, not to the
multiplicity, so the survival recursion keeps the result in [0, 1] whatever the
network does.
"""

import numpy as np
from scipy.optimize import minimize


class EmulatorMLP:

    def __init__(self, hidden=(16, 16), seed=0, l2=1e-6):
        self.hidden = tuple(hidden)
        self.seed = seed
        self.l2 = l2
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

    def _forward(self, layers, X):
        """Returns the output and the per-layer activations for the backward."""
        acts = [X]
        h = X
        for i, (W, b) in enumerate(layers):
            z = h @ W + b
            h = np.tanh(z) if i < len(layers) - 1 else z
            acts.append(h)
        return h.ravel(), acts

    def _loss_grad(self, theta, X, y, w):
        layers = self._unpack(theta)
        pred, acts = self._forward(layers, X)

        resid = pred - y
        loss = 0.5 * np.sum(w * resid ** 2) / np.sum(w)
        loss += 0.5 * self.l2 * np.dot(theta, theta)

        # dL/dpred
        g = (w * resid / np.sum(w))[:, None]

        grads = [None] * len(layers)
        for i in range(len(layers) - 1, -1, -1):
            W, b = layers[i]
            h_in = acts[i]
            grads[i] = (h_in.T @ g, g.sum(axis=0))
            if i > 0:
                g = (g @ W.T) * (1.0 - acts[i] ** 2)

        flat = np.concatenate([np.concatenate([gw.ravel(), gb.ravel()])
                               for (gw, gb) in grads])
        flat += self.l2 * theta
        return loss, flat

    # --- fitting -------------------------------------------------------------

    def fit(self, X, y, w=None, maxiter=600, restarts=1):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_

        self.shapes_ = self._shapes(X.shape[1])
        n = self._n_params()

        best = None
        for r in range(restarts):
            rng = np.random.default_rng(self.seed + r)
            # Modest init: tanh saturates and the gradient dies otherwise.
            theta0 = rng.normal(0.0, 0.5, n) / np.sqrt(max(X.shape[1], 1))
            res = minimize(self._loss_grad, theta0, args=(Xs, y, w),
                           jac=True, method="L-BFGS-B",
                           options=dict(maxiter=maxiter, maxfun=maxiter * 2))
            if best is None or res.fun < best.fun:
                best = res

        self.theta_ = best.x
        self.loss_ = best.fun
        return self

    def predict(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        pred, _ = self._forward(self._unpack(self.theta_), Xs)
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
