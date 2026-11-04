/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file variance_kernel.h
 * @brief The variance of the smoothed field, and of its slope, as a linear map
 * from a power spectrum: built once, applied to every P(k) for microseconds.
 *
 * sigma^2(R), dS/dR and <(d delta / dR)^2> are each an integral over log k of
 * a window term W(kR) or W'(kR) against P(k). The window terms carry every
 * transcendental and do not depend on the spectrum; P(k) enters once and
 * linearly. So with the k nodes and the radii fixed, each integral is a matrix
 * applied to the vector P(k_m), and the matrix can be built once with a
 * quadrature that resolves the window's oscillation at every radius.
 *
 * That makes it more accurate than quadraturing the table directly, as
 * sif_delta_covariance_pk does, as well as faster. The sin^2(kR) behind the
 * derivative variance turns over every pi / R in k, which a log-spaced table
 * samples at an effectively random phase at large k and R; here the window is
 * integrated exactly between the nodes, and P(k) only has to be smooth between
 * them. On a CDM spectrum, 600 nodes from 1e-4 to 100 h/Mpc reach 4e-6 in the
 * derivative variance, where an 8000-point table quadratured directly is off
 * by 1.5e-4 at R = 150 Mpc/h.
 *
 * The radii form a lattice uniform in ln R. sif_ep_multiplicity_function_emu
 * takes a contiguous slice of it: as far out as the barrier needs and as far
 * in as the caller's radii, wherever those fall.
 */

#ifndef SIF_MODEL_VARIANCE_KERNEL_H
#define SIF_MODEL_VARIANCE_KERNEL_H

#include <stdint.h>

#include "sif/core/macros.h"

/**
 * @brief The linear map from P(k) at fixed nodes to the variance and its
 * derivatives on a fixed radius lattice.
 *
 * Read-only once built. Every member is owned by the kernel and released by
 * sif_variance_kernel_free.
 */
typedef struct {
  /* The nodes P(k) is supplied at, ascending: the caller's array, copied. */
  uint32_t n_k;
  sif_real* k;

  /* The radius lattice, r_min e^{j dln_r}, ascending. */
  uint32_t n_radii;
  sif_real* radii;
  double dln_r;

  /* SIF_DELTA_FILTER_TOP_HAT or SIF_DELTA_FILTER_GAUSSIAN. */
  sif_option window;

  /* 3 n_radii + 1 rows of n_k weights: sigma^2, dS/dR, <(d delta / dR)^2>
   * per radius, then the last over the top half of k at the smallest. */
  double* _weights;
} sif_variance_kernel_t;

/**
 * @brief The variance of the smoothed field and of its slope, on a kernel's
 * radius lattice, for one power spectrum.
 *
 * What sif_ep_multiplicity_function_emu takes. Carries the spectrum at the
 * kernel's nodes as well, because void-in-cloud needs the covariance between
 * scales, and that is built from it on the slice the emulator uses. Every
 * member is owned by the walk and released by sif_walk_free.
 */
typedef struct {
  /* The kernel's lattice, copied. */
  uint32_t n_radii;
  sif_real* radii;

  /* sigma(R), and <(d delta / dS)^2> with S = sigma^2, per lattice radius. */
  sif_real* sigma;
  double* deriv_variance;

  /* The spectrum the walk was built from, at the kernel's nodes. */
  uint32_t n_k;
  sif_real* k;
  sif_real* pk;
  sif_option window;
} sif_walk_t;

/**
 * @brief Builds the kernel for P(k) given at `k`, over a radius lattice from
 * r_min to r_max.
 *
 * Costs a few hundred milliseconds, independent of any spectrum: the window
 * terms are integrated with 4-point Gauss-Legendre on sub-intervals of each
 * node interval, small enough in ln k and in phase kR to resolve the
 * oscillation at every lattice radius. Between nodes P(k) is the cubic in
 * ln k through the four nearest, which is what makes the map linear.
 *
 * @param k Nodes, strictly positive and strictly increasing, at least 4. The
 * integrals stop at the first and last. Six hundred log-spaced from 1e-4 to
 * 100 h/Mpc reach 4e-6 in the derivative variance on a CDM spectrum, 1200
 * reach 3e-7; an array of a Boltzmann code or an emulator of P(k) can be
 * passed as it is.
 * @param n_k Number of nodes
 * @param r_min Smallest lattice radius, strictly positive
 * @param r_max Largest lattice radius; the lattice stops at the last point
 * not beyond it. Must leave at least 4 radii and at most SIF_COV_MAX_RADII.
 * @param dln_r Lattice spacing in ln R, strictly positive. 0.02 holds the
 * emulated multiplicity function within 2e-4 of a lattice twice as fine.
 * @param opt SIF_DELTA_FILTER_TOP_HAT (default) or SIF_DELTA_FILTER_GAUSSIAN
 *
 * @return A new kernel, released with sif_variance_kernel_free, or NULL on
 * invalid input or failed allocation.
 */
SIF_NODISCARD sif_variance_kernel_t* sif_variance_kernel_alloc(
  const sif_real* k, uint32_t n_k, sif_real r_min, sif_real r_max,
  sif_real dln_r, sif_option opt);

/**
 * @brief Releases a kernel. Accepts NULL.
 */
void sif_variance_kernel_free(sif_variance_kernel_t* kernel);

/**
 * @brief Applies a kernel to a power spectrum: sigma and the derivative
 * variance at every lattice radius.
 *
 * Three matrix-vector products, n_radii x n_k each, and no transcendentals;
 * tens of microseconds for 360 radii and 600 nodes.
 *
 * @param kernel A kernel from sif_variance_kernel_alloc; not modified.
 * @param pk kernel->n_k values, P(k) at kernel->k, non-negative.
 *
 * @return A new walk, released with sif_walk_free, or NULL on invalid input,
 * failed allocation, or a spectrum that gives no variance at some lattice
 * radius.
 */
SIF_NODISCARD sif_walk_t* sif_walk_pk(
  const sif_variance_kernel_t* kernel, const sif_real* pk);

/**
 * @brief Releases a walk. Accepts NULL.
 */
void sif_walk_free(sif_walk_t* walk);

#endif /* SIF_MODEL_VARIANCE_KERNEL_H */
