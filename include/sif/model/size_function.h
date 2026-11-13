/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file size_function.h
 * @brief Model void size functions, and the spherical-evolution mapping they
 * are expressed in.
 *
 * The models predict a number density per radius bin in the same container the
 * measurement fills, so a prediction and a catalogue can be plotted and
 * compared directly.
 */

#ifndef SIF_MODEL_SIZE_FUNCTION_H
#define SIF_MODEL_SIZE_FUNCTION_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/model/variance_kernel.h"
#include "sif/structures/size_function.h"

/* Bernardeau (1994) fit constant linking the linear and non-linear void
 * density contrasts. */
#define SIF_SPHERICAL_EXPANSION_C 1.594

/**
 * @brief Non-linear density contrast a void of a given linear contrast
 * evolves to.
 *
 * Under SIF_SPHERICAL_B94, the Bernardeau (1994) fit quoted by Jennings, Li &
 * Hu (2013) eq. (A4): delta_NL = (1 - delta_L/c)^-c - 1. Under
 * SIF_SPHERICAL_EDS, the Einstein-de Sitter expansion solution, by
 * root-find. The two agree to better than 0.1% on the expansion factor.
 * Under SIF_SPHERICAL_LCDM, the spherical top-hat integrated in flat
 * LambdaCDM without radiation, from the growing mode in matter domination to
 * the epoch of the voids; it differs from EDS by up to 0.2% on delta_L
 * between delta_NL = -0.3 and -0.8 at Omega_m = 0.32, and equals it at
 * Omega_m = 1.
 *
 * @note Only the expanding branch is implemented; a non-negative linear
 * contrast collapses rather than expands.
 *
 * @param delta_linear Linear density contrast, strictly negative, at the
 * epoch of the voids
 * @param omega_m Omega_m at the epoch of the voids, Omega_m(z), in (0, 1];
 * read only under SIF_SPHERICAL_LCDM. The flat LambdaCDM solution depends on
 * the cosmology and the epoch only through it.
 * @param opt SIF_SPHERICAL_B94 (default), SIF_SPHERICAL_EDS or
 * SIF_SPHERICAL_LCDM
 *
 * @return The non-linear contrast, strictly between -1 and 0, or 0 for input
 * outside that branch. A valid result is never 0, so the two are
 * distinguishable.
 */
sif_real sif_spherical_map_nonlinear(
  sif_real delta_linear, sif_real omega_m, sif_option opt);

/**
 * @brief Linear density contrast that evolves into a given non-linear one,
 * the inverse of sif_spherical_map_nonlinear.
 *
 * Wanted when a barrier is quoted as an observed underdensity rather than as a
 * linear threshold.
 *
 * @param delta_nonlinear Non-linear density contrast, strictly between -1 and 0
 * @param omega_m As sif_spherical_map_nonlinear
 * @param opt SIF_SPHERICAL_B94 (default), SIF_SPHERICAL_EDS or
 * SIF_SPHERICAL_LCDM
 *
 * @return The linear contrast, strictly negative, or 0 for input outside that
 * range.
 */
sif_real sif_spherical_map_linear(
  sif_real delta_nonlinear, sif_real omega_m, sif_option opt);

/**
 * @brief Sheth & van de Weygaert void size function.
 *
 * Jennings, Li & Hu (2013) eqs. (9) and (10): the number-conserving mapping,
 * in which the linear abundance is evaluated at r_L = r / F and reported
 * against r without rescaling its amplitude.
 *
 * @note Number conservation makes this model exceed a void volume fraction of
 * one at large radii, which is why Vdn exists. It is reproduced here as the
 * reference, not as a recommendation.
 *
 * @param k Wavenumbers, strictly positive and strictly increasing
 * @param pk Linear power spectrum sampled at k
 * @param n_points Length of k and pk, at least 2
 * @param radii Eulerian (observed) void radii, strictly positive
 * @param n_radii Number of radii
 * @param delta_nl Non-linear density contrast the voids are defined by (the
 * threshold of the finder), strictly between -1 and 0. Its linear contrast, the
 * barrier the field must cross, is sif_spherical_map_linear of it, with omega_m
 * and the mapping of opt
 * @param delta_c Collapse barrier, strictly positive
 * @param omega_m Omega_m at the epoch of the voids, read only under
 * SIF_SPHERICAL_LCDM; see sif_spherical_map_nonlinear
 * @param opt SIF_VSF_BIN_LN (default) or SIF_VSF_BIN_LINEAR for the units,
 * combined with SIF_SPHERICAL_B94 (default), SIF_SPHERICAL_EDS or
 * SIF_SPHERICAL_LCDM for the mapping used to expand the radii. The window is always a top-hat, which is
 * what the barriers are calibrated against.
 *
 * @note The Lagrangian-to-Eulerian expansion factor is not an argument: it is
 * (1 + delta_NL)^(-1/3), which delta_nl already determines.
 *
 * @return Newly allocated size function with n_bins = n_radii, released with
 * sif_size_function_free, or NULL on invalid input.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_svdw(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real delta_nl, sif_real delta_c, sif_real omega_m,
  sif_option opt);

/**
 * @brief Volume-conserving (Vdn) void size function.
 *
 * Jennings, Li & Hu (2013) eq. (12). Identical to SvdW except that the
 * abundance is divided by the Eulerian volume rather than the Lagrangian one,
 * which conserves volume instead of number and keeps the void volume fraction
 * below unity.
 *
 * Parameters are as sif_size_function_svdw.
 *
 * @return Newly allocated size function with n_bins = n_radii, released with
 * sif_size_function_free, or NULL on invalid input.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_vdn(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real delta_nl, sif_real delta_c, sif_real omega_m,
  sif_option opt);

/**
 * @brief The EVM void size function: a calibrated moving barrier, its emulated
 * first crossing, and a fitted Lagrangian-to-Eulerian mapping.
 *
 * sigma(R) comes from the spectrum, and with sigma12 = sigma(12 Mpc) it fixes
 * the Sheth-Mo-Tormen barrier calibrated at z = 99. The emulator
 * (sif_ep_multiplicity_function_emu, single barrier) gives the Lagrangian
 * abundance n_L, and
 *
 *     dn/dln R_E = (1 + delta_NL) n_L(R_L) T,  R_L = R_E (1 + delta_NL)^(1/3)
 *
 * where T is a five-constant correction to the volume-conserving mapping,
 * a function of B / sigma and sigma at R_E, delta_NL and sigma12. Nothing is
 * free: all nine constants are the maximum a posteriori of their calibration.
 *
 * The epoch and the cosmology are seen only through the spectrum, which is the
 * linear P(k) at the redshift wanted; no redshift is passed.
 *
 * @note Calibrated on one cosmology, for sigma12 between 0.39 and 1, a
 * magnitude of its linear contrast at z = 99 (scaled by sigma12(z = 99) / sigma12) between
 * 0.003 and 0.045, and radii above nu ~ 0.5. Outside the first two a warning is
 * logged and the answer is still returned. The correction is not a derived law,
 * so do not expect it to extrapolate.
 *
 * @note Builds a variance kernel on every call, which takes a good part of a
 * second. To evaluate many spectra, build the kernel once and call
 * sif_size_function_evm_kernel.
 *
 * @param k Wavenumbers, strictly positive and strictly increasing, at least 4.
 * Sampled well enough for sif_variance_kernel_alloc, whose integrals stop at
 * the first and last.
 * @param pk Linear power spectrum sampled at k
 * @param n_points Length of k and pk
 * @param radii Eulerian (observed) void radii, strictly positive and strictly
 * increasing
 * @param n_radii Number of radii
 * @param delta_nl Non-linear density contrast the voids are defined by (the
 * threshold of the finder), strictly between -1 and 0; converted to the linear
 * barrier inside, with omega_m and the mapping of opt.
 * @param h Hubble parameter in units of 100 km/s/Mpc, strictly positive: it
 * places 12 Mpc on radii measured in Mpc/h
 * @param omega_m Omega_m at the epoch of the voids, read only under
 * SIF_SPHERICAL_LCDM; see sif_spherical_map_nonlinear
 * @param opt SIF_VSF_BIN_LN (default) or SIF_VSF_BIN_LINEAR for the units,
 * combined with SIF_SPHERICAL_B94 (default, what the model was calibrated
 * with), SIF_SPHERICAL_EDS or SIF_SPHERICAL_LCDM for delta_NL. The window is always a top-hat.
 *
 * @return Newly allocated size function with n_bins = n_radii, released with
 * sif_size_function_free, or NULL on invalid input. A radius where the
 * emulated multiplicity underflows gives exactly 0.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_evm(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real delta_nl, sif_real h, sif_real omega_m,
  sif_option opt);

/**
 * @brief sif_size_function_evm on a variance kernel built beforehand: what a
 * chain should call, since only the spectrum changes between evaluations.
 *
 * @param kernel From sif_variance_kernel_alloc, with a top-hat window; not
 * modified. Its lattice must hold 12 Mpc and `radii`, reach in to the
 * Lagrangian radii and out to where B / sigma is ~3.5 or more. 0.3 to 1000
 * Mpc/h at a spacing of 0.02 in ln R is what the model was calibrated with.
 * @param pk kernel->n_k values, P(k) at kernel->k
 *
 * Remaining parameters, and the result, are as sif_size_function_evm.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_evm_kernel(
  const sif_variance_kernel_t* kernel, const sif_real* pk,
  const sif_real* radii, uint32_t n_radii, sif_real delta_nl, sif_real h,
  sif_real omega_m, sif_option opt);

/**
 * @brief sif_size_function_evm_kernel on a walk built beforehand: for several
 * thresholds of one spectrum, which then share the walk.
 *
 * @param walk From sif_walk_pk, of a kernel as sif_size_function_evm_kernel
 * requires; not modified.
 *
 * Remaining parameters, and the result, are as sif_size_function_evm.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_evm_walk(
  const sif_walk_t* walk, const sif_real* radii, uint32_t n_radii,
  sif_real delta_nl, sif_real h, sif_real omega_m, sif_option opt);

#endif /* SIF_MODEL_SIZE_FUNCTION_H */
