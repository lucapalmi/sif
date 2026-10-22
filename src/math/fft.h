/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file fft.h
 * @brief FFTW wrapper: forward and backward transforms of a grid, and the
 * spectral operations done between them. Private to the library.
 *
 * Two objects, with different lifetimes. The manager owns the process-global
 * FFTW state -- the plan cache and its wisdom file -- and lives from
 * sif_system_init() to sif_system_finalise(). A workspace owns the buffers and
 * plans for one grid size and is created and destroyed around a piece of work.
 *
 * The transforms are real-to-complex, so the Fourier-space array holds only
 * half the modes; the rest follow by Hermitian symmetry and are never stored.
 */

#ifndef SIF__MATH_FFT_H
#define SIF__MATH_FFT_H

#include <fftw3.h>
#include <stdbool.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/grid.h"

/* --- FFTW Precision Wrapper --- */
#ifdef SIF_USE_DOUBLE
typedef fftw_complex sif_real_complex;
typedef fftw_plan sif_real_fftw_plan;

#  define real_fftw_init_threads                fftw_init_threads
#  define real_fftw_plan_with_nthreads          fftw_plan_with_nthreads
#  define real_fftw_cleanup_threads             fftw_cleanup_threads
#  define real_fftw_import_wisdom_from_filename fftw_import_wisdom_from_filename
#  define real_fftw_export_wisdom_to_filename   fftw_export_wisdom_to_filename
#  define real_fftw_cleanup                     fftw_cleanup
#  define real_fftw_execute                     fftw_execute
#  define real_fftw_destroy_plan                fftw_destroy_plan
#  define real_fftw_plan_dft_r2c_3d             fftw_plan_dft_r2c_3d
#  define real_fftw_plan_dft_c2r_3d             fftw_plan_dft_c2r_3d
#  define real_fftw_plan_dft_r2c_2d             fftw_plan_dft_r2c_2d
#  define real_fftw_plan_dft_c2r_2d             fftw_plan_dft_c2r_2d
#  define real_fftw_plan_many_dft               fftw_plan_many_dft
#  define real_fftw_execute_dft                 fftw_execute_dft
#  define real_fftw_export_wisdom_to_string     fftw_export_wisdom_to_string
#  define real_fftw_malloc                      fftw_malloc
#  define real_fftw_free                        fftw_free
#  define real_fftw_execute_dft_r2c             fftw_execute_dft_r2c
#  define real_fftw_execute_dft_c2r             fftw_execute_dft_c2r
#  define real_fftw_alignment_of                fftw_alignment_of
#  define real_fftw_forget_wisdom               fftw_forget_wisdom

/**
 * @brief Basename of the wisdom file.
 *
 * Tagged with the precision because FFTW keeps two entirely separate wisdom
 * namespaces: `fftw_import_wisdom_from_filename` rejects a file written by
 * `fftwf_export_wisdom_to_filename` outright. Sharing one name between a float
 * build and a double build would have each reject the other's file and then
 * overwrite it, so the two would take turns destroying accumulated wisdom.
 */
#  define SIF__FFT_WISDOM_FILE "sif-f64.wisdom"
#else
typedef fftwf_complex sif_real_complex;
typedef fftwf_plan sif_real_fftw_plan;

#  define real_fftw_init_threads       fftwf_init_threads
#  define real_fftw_plan_with_nthreads fftwf_plan_with_nthreads
#  define real_fftw_cleanup_threads    fftwf_cleanup_threads
#  define real_fftw_import_wisdom_from_filename                                \
    fftwf_import_wisdom_from_filename
#  define real_fftw_export_wisdom_to_filename fftwf_export_wisdom_to_filename
#  define real_fftw_cleanup                   fftwf_cleanup
#  define real_fftw_execute                   fftwf_execute
#  define real_fftw_destroy_plan              fftwf_destroy_plan
#  define real_fftw_plan_dft_r2c_3d           fftwf_plan_dft_r2c_3d
#  define real_fftw_plan_dft_c2r_3d           fftwf_plan_dft_c2r_3d
#  define real_fftw_plan_dft_r2c_2d           fftwf_plan_dft_r2c_2d
#  define real_fftw_plan_dft_c2r_2d           fftwf_plan_dft_c2r_2d
#  define real_fftw_plan_many_dft             fftwf_plan_many_dft
#  define real_fftw_execute_dft               fftwf_execute_dft
#  define real_fftw_export_wisdom_to_string   fftwf_export_wisdom_to_string
#  define real_fftw_malloc                    fftwf_malloc
#  define real_fftw_free                      fftwf_free
#  define real_fftw_execute_dft_r2c           fftwf_execute_dft_r2c
#  define real_fftw_execute_dft_c2r           fftwf_execute_dft_c2r
#  define real_fftw_alignment_of              fftwf_alignment_of
#  define real_fftw_forget_wisdom             fftwf_forget_wisdom

/** @brief Basename of the wisdom file; see the double-precision branch. */
#  define SIF__FFT_WISDOM_FILE                "sif-f32.wisdom"
#endif

/**
 * @brief Global manager for FFTW state, flags, and wisdom.
 */
typedef struct {
  unsigned int flags;
  char* wisdom_dir;
} sif_fft_manager_t;

/**
 * @brief Ephemeral workspace for specific grid operations.
 *
 * The 3D transforms are not FFTW's own. sif runs them as a 2D transform of
 * every x-plane and a 1D transform along x of every (y, kz) pencil, each a
 * single-threaded plan, and parallelises over the planes and pencils itself.
 * FFTW's threaded 3D plans are not bit-identical across thread counts -- the
 * planner picks a different decomposition for a different count -- and
 * everything downstream of the smoothed field inherits the last bits. Here
 * every plane and every pencil goes through the same plan whatever the thread
 * count, so the output does not depend on it. It is also faster: 10-35% on
 * 256^3 to 512^3, tuned or not.
 *
 * @note delta_k_cpy doubles as the real-space output buffer: the backward
 * transform is executed in place on it. It is owned by the workspace until
 * sif__fft_workspace_take_real_buffer() hands it over.
 */
typedef struct {
  sif_fft_manager_t* mgr;
  uint32_t n_cells;
  sif_real_complex* delta_k;
  sif_real_complex* delta_k_cpy;

  /* Forward: plane_r2c over every x-plane, then pencil_forward along x.
   * Backward: pencil_backward, then plane_c2r. NULL until planned. */
  sif_real_fftw_plan plane_r2c;
  sif_real_fftw_plan pencil_forward;
  sif_real_fftw_plan pencil_backward;
  sif_real_fftw_plan plane_c2r;

  /**
   * Planner quality, resolved once from the manager and used for **both**
   * transforms.
   *
   * One field rather than a decision at each plan site, because the two used
   * to disagree: the backward plan measured while the forward silently settled
   * for FFTW_ESTIMATE, so `skip_tuning` described only half of what the
   * workspace did. Whatever this says, both transforms are planned to it or
   * neither is -- including when it is downgraded mid-flight, which is why it
   * is mutable. See fft_plan_stage().
   */
  unsigned int plan_flags;

  /**
   * The assignment-window correction every filter carries, or NULL when none
   * is wanted. Set by sif__fft_set_window_correction().
   *
   * It corrects the *filtered* copy rather than delta_k, so the spectrum the
   * workspace was given stays untouched and an unfiltered backward transform
   * still reproduces the caller's input exactly.
   *
   * window_axis is the per-axis factor when the correction is separable --
   * n_cells entries cover all three axes -- and window_wsa holds W, S and A
   * per axis (3 n_cells) for the one case that is not, the aliasing-aware
   * correction of an interlaced grid; see sif__fft_set_window_correction().
   */
  double* window_axis;
  double* window_wsa;
} sif_fft_workspace_t;

/**
 * @brief The order of the window a grid was deposited with: the
 * sif_assignment_t, or CIC for a grid that does not say -- what every grid
 * was before the scheme was recorded.
 */
static inline int sif__fft_grid_window_order(const sif_grid_t* grid) {
  return grid->assignment == SIF_ASSIGNMENT_UNKNOWN ? (int)SIF_ASSIGNMENT_CIC
                                                    : (int)grid->assignment;
}

/** @brief How an assignment window is divided out of a spectrum. */
typedef enum {
  /** Divide by the window: exact for the field the grid would hold without
   * aliasing, and amplifies whatever aliased power sits near Nyquist. */
  SIF__WINDOW_FULL = 0,
  /** The least-squares estimate under white power near Nyquist: multiply by
   * W over the sum of W^2 over the images that reach the grid. */
  SIF__WINDOW_ALIASING = 1
} sif_window_mode_t;

typedef enum {
  SIF__FILTER_NONE,
  SIF__FILTER_TOP_HAT,
  SIF__FILTER_GAUSSIAN
} sif_filter_type_t;

/**
 * @brief Creates the fft manager
 *
 * @param skip_tuning Use FFTW_ESTIMATE instead of FFTW_MEASURE
 * @param save_wisdom Save the plans on the disk
 * @param wisdom_file The path to the wisdom file
 *
 * @return The initialised fft manager
 */
SIF_NODISCARD sif_fft_manager_t* sif__fft_manager_init(
  bool skip_tuning, const char* wisdom_dir);

/**
 * @brief Closes an fft manager
 *
 *@param mgr The fft manager to finalise
 */
void sif__fft_manager_finalise(sif_fft_manager_t* mgr);

/**
 * @brief Allocates a fft workspace
 *
 * Allocates the spectrum buffer only; the forward plan is created lazily by
 * the first sif__fft_grid_forward().
 *
 * Also fixes the planner quality for both transforms, from @p mgr and the
 * `fft_tuning_max_gib` ceiling, which is what makes `skip_tuning` mean the
 * same thing in both directions. Tuning measures on scratch buffers of one
 * plane, never on the caller's grid; see sif_fft_workspace_t::plan_flags and
 * #SIF__FFT_TUNING_MAX_GIB_DEFAULT.
 *
 * @param mgr The global fft manager
 * @param n_cells the number of cells for the fft
 *
 * @return The initialised fft workspace
 */
SIF_NODISCARD sif_fft_workspace_t* sif__fft_workspace_alloc(
  sif_fft_manager_t* mgr, uint32_t n_cells);

/**
 * @brief Allocates the real-space buffer and plans the backward transform
 *
 * @return SIF_OK on success, SIF_ERR_ALLOC on failure. Must succeed before
 * sif__fft_apply_filter or sif__fft_grid_backward are called.
 */
int sif__fft_workspace_init_backward(
  sif_fft_workspace_t* ws, sif_fft_manager_t* mgr);

/**
 * @brief Detaches the real-space buffer and transfers ownership to the caller
 *
 * The workspace can no longer run a backward transform afterwards. Use this
 * when the real-space data has to outlive the workspace; the returned pointer
 * must eventually be released with sif_free_aligned.
 *
 * @return The buffer, or NULL if the workspace never allocated one
 */
SIF_NODISCARD sif_real* sif__fft_workspace_take_real_buffer(
  sif_fft_workspace_t* ws);

/**
 * @brief Frees an fft workspace, including the real-space buffer if it has not
 * been taken
 *
 *@param ws The workspace to free
 */
void sif__fft_workspace_free(sif_fft_workspace_t* ws);

/**
 * @brief Apply a filter to the fourier-space density field
 *
 * Reads delta_k and writes the filtered result into the real-space buffer,
 * which sif__fft_grid_backward then transforms in place.
 *
 * @param ws The fft workspace
 * @param filter The filter to apply
 * @param r The smoothing radius
 * @param box_length The physical length of the simulation box
 *
 * @return SIF_OK on success, SIF_ERR_INVALID if the backward stage was never
 * initialised or the filter is unsupported, SIF_ERR_ALLOC on failure
 */
/**
 * @brief sif__fft_apply_filter() with a Gaussian of width `sigma` on top of
 * the filter; 0 for none, which is sif__fft_apply_filter().
 */
int sif__fft_apply_filter_smoothed(sif_fft_workspace_t* ws,
  sif_filter_type_t filter, sif_real r, sif_real sigma, sif_real box_length);

int sif__fft_apply_filter(sif_fft_workspace_t* ws, sif_filter_type_t filter,
  sif_real r, sif_real box_length);

/**
 * @brief Execute the real-to-complex fft
 *
 * Creates the forward plan on the first call. The plan is then reused, so
 * every later call must pass a buffer with the same alignment (everything sif
 * allocates is cache-line aligned).
 *
 * @param ws The fft workspace
 * @param grid The real-space cubic grid to transform. Not modified: an r2c
 * transform preserves its input by default, and a tuned plan measures against
 * a scratch buffer rather than against this one.
 *
 * @return SIF_OK on success, SIF_ERR_INVALID on bad arguments, SIF_ERR_ALLOC
 * if the plan could not be created
 */
int sif__fft_grid_forward(sif_fft_workspace_t* ws, const sif_grid_t* grid);

/**
 * @brief The factor that undoes an assignment window of order @p order at the
 * Nyquist corner (k_x = k_y = k_z = N/2): the largest a correction ever
 * multiplies a mode by, for the warnings that say how much a small filter
 * radius lets through.
 */
double sif__fft_window_corner_gain(
  int order, sif_window_mode_t mode, int interlaced);

/**
 * @brief Divide an assignment window out of delta_k, in place.
 *
 * Mass assignment of order p (1 NGP, 2 CIC, 3 TSC, 4 PCS) convolves the field
 * with a kernel whose transform is W = prod_i sinc^p(pi k_i / N), and any
 * filter applied afterwards sees a field that kernel already smoothed. In
 * #SIF__WINDOW_FULL mode the correction is 1 / W. It diverges towards the
 * Nyquist corner -- ~15 there for CIC, ~230 for PCS -- and what it amplifies
 * is mostly aliased power, the images above Nyquist that the grid folds back.
 *
 * #SIF__WINDOW_ALIASING instead multiplies by W / sum_n W^2(k + 2 pi n / l),
 * the least-squares estimate of the band-limited field when the power near
 * Nyquist is white, as shot noise is. The image sum is separable, per axis
 *
 *   S_p(x) = sum_n sinc^(2p)(x + pi n),   x = pi k_i / N, s = sin x, c = cos x
 *     S_1 = 1                  S_2 = 1 - 2 s^2 / 3
 *     S_3 = 1 - s^2 + 2 s^4 / 15
 *     S_4 = 1 - 4 s^2 / 3 + 2 s^4 / 5 - 4 s^6 / 315
 *
 * An interlaced grid keeps only the images with an even index sum, whose
 * weight is (prod S + prod A) / 2 with the alternating sums
 *
 *   A_p(x) = sum_n (-1)^n sinc^(2p)(x + pi n)
 *     A_1 = c                  A_2 = c (c^2 + 5) / 6
 *     A_3 = c (c^4 + 58 c^2 + 61) / 120
 *     A_4 = c (c^6 + 543 c^4 + 3111 c^2 + 1385) / 5040
 *
 * which is not separable. Both sums come from differentiating
 * sum_n 1 / (x + pi n) = cot x and sum_n (-1)^n / (x + pi n) = csc x
 * 2p - 1 times. The two modes agree at large scales and part at the Nyquist
 * corner: for CIC 1.22 per axis instead of 2.47.
 *
 * @param ws The fft workspace, after a forward transform
 * @param order The window's order p, 1 to 4
 * @param mode How to divide it out
 * @param interlaced Non-zero for an interlaced grid
 *
 * @return SIF_OK, SIF_ERR_INVALID if the workspace has no spectrum or the
 * order is out of range, or SIF_ERR_ALLOC
 */
int sif__fft_deconvolve_window(sif_fft_workspace_t* ws, int order,
  sif_window_mode_t mode, int interlaced);

/**
 * @brief Divide an assignment window out of every filtered field this
 * workspace produces from here on, instead of out of delta_k itself.
 *
 * The same correction as sif__fft_deconvolve_window(), for callers that
 * filter repeatedly and must also hand the input field back: correcting inside
 * the filter leaves delta_k alone, so sif__fft_apply_filter() with
 * #SIF__FILTER_NONE still restores the original field bit for bit.
 *
 * Setting it again replaces the previous correction.
 *
 * @param ws The workspace, which must know its grid size
 * @param order The window's order p, 1 to 4; 0 to stop correcting
 * @param mode How to divide it out
 * @param interlaced Non-zero for an interlaced grid
 *
 * @return SIF_OK, or SIF_ERR_* if the tables could not be built
 */
int sif__fft_set_window_correction(sif_fft_workspace_t* ws, int order,
  sif_window_mode_t mode, int interlaced);

/**
 * @brief Replace the phases of delta_k with random ones, in place
 *
 * Produces a surrogate field with the same power spectrum and no higher-order
 * correlations: the Gaussian counterpart of the input. Hermitian symmetry is
 * enforced explicitly on the two self-redundant planes of the r2c layout
 * (k_z = 0 and, for even N, k_z = N/2), and the self-conjugate modes there are
 * kept real, so the c2r transform has no imaginary part to silently discard.
 * The k = 0 mode is left untouched, preserving the mean of the field.
 *
 * Phases are drawn from a per-mode hash of the seed rather than from a
 * sequential stream, so the result depends only on (seed, n_cells) and not on
 * the thread count.
 *
 * @param ws The fft workspace, after a forward transform
 * @param seed Random seed
 * @param resample_amplitudes If false, every |delta_k| is preserved exactly and
 * only the phase changes. If true, amplitudes are additionally redrawn from the
 * Rayleigh distribution with the same mean square, giving a true Gaussian
 * random field realisation rather than a fixed-amplitude one.
 *
 * @return SIF_OK, or SIF_ERR_INVALID if the workspace has no spectrum
 */
int sif__fft_randomise_phases(
  sif_fft_workspace_t* ws, uint64_t seed, bool resample_amplitudes);

/** @brief Highest order sif__fft_spectral_moments will accept. Past this the
 * k^(2j) weighting is so steep that the sum is entirely a statement about the
 * smallest scale on the grid, whatever that happens to be.
 *
 * @note Must not be below SIF_MAX_MOMENT_ORDER in
 * `sif/structures/delta_moments.h`, which is what the public API advertises and
 * what the result containers are sized for. This header is not the place to
 * include that one, so the two are kept equal by hand. */
#define SIF__FFT_MAX_MOMENT_ORDER 4

/**
 * @brief Spectral moments of the filtered field, evaluated in Fourier space
 *
 * Computes sigma_j^2(R) = sum_{k != 0} k^(2j) |delta_k|^2 W^2(kR) for every
 * j from 0 to max_order in a single pass, normalised so that j = 0 reproduces
 * the variance of the real-space smoothed field. Pinning the convention on
 * that identity is what makes the whole family unambiguous: it is the discrete
 * Parseval relation, exact rather than a continuum approximation, so
 * real-space measurements of <delta_R^2> and of the gradient and Laplacian
 * variances must reproduce j = 0, 1, 2 to round-off.
 *
 * sigma_1 and sigma_2 are the RMS of |grad delta_R| and of the Laplacian, so
 * they carry units of 1/length and 1/length^2 respectively. The k = 0 mode is
 * excluded, which is exactly what turns the j = 0 sum into a variance rather
 * than a mean square.
 *
 * @note The k^(2j) weighting puts all the numerical risk at high k. With a
 * top-hat window W^2 falls only as k^-4, so the j = 2 sum is dominated by the
 * modes just below Nyquist and its value is set by the grid resolution rather
 * than by the field. Use SIF__FILTER_GAUSSIAN when sigma_2 has to mean
 * something, and read high_k_fraction before trusting any moment.
 *
 * @param ws The fft workspace, after a forward transform
 * @param filter The filter to evaluate the moments under
 * @param r The smoothing radius
 * @param box_length The physical side length of the box
 * @param max_order The highest moment order, at most SIF__FFT_MAX_MOMENT_ORDER
 * @param n_tracers Number of tracers behind the density field. The Poisson
 * term is then subtracted mode by mode, which matters for j >= 1 because the
 * flat shot-noise floor is amplified by k^(2j). Pass 0 to leave it in.
 * @param sigma_sq Output, max_order + 1 entries. A shot-noise subtraction can
 * drive an entry negative, which is reported as-is rather than clamped: it
 * means that moment is noise-dominated.
 * @param high_k_fraction Output, max_order + 1 entries, or NULL. The fraction
 * of each sum contributed by modes above half the Nyquist frequency. Anything
 * that is not small means the moment is a statement about the grid rather than
 * about the field.
 *
 * @return SIF_OK, or SIF_ERR_INVALID / SIF_ERR_ALLOC
 */
int sif__fft_spectral_moments(const sif_fft_workspace_t* ws,
  sif_filter_type_t filter, sif_real r, sif_real box_length, uint8_t max_order,
  uint64_t n_tracers, double* sigma_sq, double* high_k_fraction);

/**
 * @brief Execute the complex-to-real fft, normalised and compacted
 *
 * The transform runs in place on the workspace's real-space buffer, which is
 * then stripped of its FFTW row padding so the result is a contiguous
 * n_cells^3 array.
 *
 * @param ws The fft workspace
 *
 * @return The buffer, still owned by the workspace (see
 * sif__fft_workspace_take_real_buffer), or NULL if the backward stage is
 * missing
 */
sif_real* sif__fft_grid_backward(sif_fft_workspace_t* ws);

#endif /* SIF__MATH_FFT_H */
