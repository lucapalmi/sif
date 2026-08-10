#ifndef __SIF_FFT_H__
#define __SIF_FFT_H__

#include <fftw3.h>
#include <stdbool.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/grid.h"

/* --- FFTW Precision Wrapper --- */
#ifdef __SIF_USE_DOUBLE
typedef fftw_complex sif_real_complex_t;
typedef fftw_plan sif_real_fftw_plan_t;

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
#  define real_fftw_malloc                      fftw_malloc
#  define real_fftw_free                        fftw_free
#  define real_fftw_execute_dft_r2c             fftw_execute_dft_r2c
#  define real_fftw_execute_dft_c2r             fftw_execute_dft_c2r
#else
typedef fftwf_complex sif_real_complex_t;
typedef fftwf_plan sif_real_fftw_plan_t;

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
#  define real_fftw_malloc                    fftwf_malloc
#  define real_fftw_free                      fftwf_free
#  define real_fftw_execute_dft_r2c           fftwf_execute_dft_r2c
#  define real_fftw_execute_dft_c2r           fftwf_execute_dft_c2r
#endif

/* * @brief Global manager for FFTW state, flags, and wisdom.
 */
typedef struct {
  unsigned int flags;
  char* wisdom_dir;
} sif_fft_manager_t;

/* * @brief Ephemeral workspace for specific grid operations.
 *
 * @note delta_k_cpy doubles as the real-space output buffer: the backward
 * transform is executed in place on it. It is owned by the workspace until
 * sif_fft_workspace_take_real_buffer() hands it over.
 */
typedef struct {
  sif_fft_manager_t* mgr;
  uint32_t n_cells;
  sif_real_complex_t* delta_k;
  sif_real_complex_t* delta_k_cpy;
  sif_real_fftw_plan_t forward_plan;
  sif_real_fftw_plan_t backward_plan;
} sif_fft_workspace_t;

typedef enum { FILTER_NONE, FILTER_TOP_HAT, FILTER_GAUSSIAN } sif_filter_type_t;

/*
 * @brief Creates the fft manager
 *
 * @param skip_tuning Use FFTW_ESTIMATE instead of FFTW_MEASURE
 * @param save_wisdom Save the plans on the disk
 * @param wisdom_file The path to the wisdom file
 *
 * @return The initialized fft manager
 */
NODISCARD sif_fft_manager_t* sif_fft_manager_init(
  bool skip_tuning, const char* wisdom_dir);

/*
 * @brief Closes an fft manager
 *
 *@param mgr The fft manager to finalize
 */
void sif_fft_manager_finalize(sif_fft_manager_t* mgr);

/*
 * @brief Allocates a fft workspace
 *
 * @param mgr The global fft manager
 * @param n_cells the number of cells for the fft
 *
 * @return The initialized fft workspace
 */
NODISCARD sif_fft_workspace_t* sif_fft_workspace_alloc(
  sif_fft_manager_t* mgr, uint32_t n_cells);

/*
 * @brief Allocates the real-space buffer and plans the backward transform
 *
 * @return SIF_OK on success, SIF_ERR_ALLOC on failure. Must succeed before
 * sif_fft_apply_filter or sif_fft_grid_backward are called.
 */
int sif_fft_workspace_init_backward(sif_fft_workspace_t* ws, sif_fft_manager_t* mgr);

/*
 * @brief Detaches the real-space buffer and transfers ownership to the caller
 *
 * The workspace can no longer run a backward transform afterwards. Use this
 * when the real-space data has to outlive the workspace; the returned pointer
 * must eventually be released with sif_free_aligned.
 *
 * @return The buffer, or NULL if the workspace never allocated one
 */
NODISCARD real_t* sif_fft_workspace_take_real_buffer(sif_fft_workspace_t* ws);

/*
 * @brief Frees an fft workspace, including the real-space buffer if it has not
 * been taken
 *
 *@param ws The workspace to free
 */
void sif_fft_workspace_free(sif_fft_workspace_t* ws);

/*
 * @brief Apply a filter to the fourier-space density field
 *
 * Reads delta_k and writes the filtered result into the real-space buffer,
 * which sif_fft_grid_backward then transforms in place.
 *
 * @param ws The fft workspace
 * @param filter The filter to apply
 * @param r The smoothing radius
 * @param box_length The physical length of the simulation box
 *
 * @return SIF_OK on success, SIF_ERR_INVALID if the backward stage was never
 * initialized or the filter is unsupported, SIF_ERR_ALLOC on failure
 */
int sif_fft_apply_filter(
  sif_fft_workspace_t* ws, sif_filter_type_t filter, real_t r, real_t box_length);

/*
 * @brief Execute the real-to-complex fft
 *
 * @param ws The fft workspace
 * @param grid The real-space cubic grid to transform
 */
void sif_fft_grid_forward(sif_fft_workspace_t* ws, const sif_grid_t* grid);

/*
 * @brief Divide out the Cloud-In-Cell assignment window from delta_k
 *
 * CIC deposition convolves the field with a triangular kernel, whose transform
 * is prod_i sinc^2(pi k_i / N). Any subsequent filtering sees a field that has
 * already been smoothed by it, so it has to be removed first. Operates in
 * place on delta_k and is independent of any filter radius, so it belongs
 * between sif_fft_grid_forward and the first sif_fft_apply_filter.
 *
 * @note The correction diverges towards the Nyquist corner, where it reaches a
 * factor of ~15. That is harmless as long as the filter applied afterwards
 * suppresses those modes, which a top-hat of radius >= 2 cells does.
 *
 * @param ws The fft workspace, after a forward transform
 *
 * @return SIF_OK, or SIF_ERR_INVALID if the workspace has no spectrum
 */
int sif_fft_deconvolve_cic(sif_fft_workspace_t* ws);

/*
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
 * random field realization rather than a fixed-amplitude one.
 *
 * @return SIF_OK, or SIF_ERR_INVALID if the workspace has no spectrum
 */
int sif_fft_randomize_phases(
  sif_fft_workspace_t* ws, uint64_t seed, bool resample_amplitudes);

/* Highest order sif_fft_spectral_moments will accept. Past this the k^(2j)
 * weighting is so steep that the sum is entirely a statement about the
 * smallest scale on the grid, whatever that happens to be. */
#define FFT_MAX_MOMENT_ORDER 4

/*
 * @brief Spectral moments of the filtered field, evaluated in Fourier space
 *
 * Computes sigma_j^2(R) = sum_{k != 0} k^(2j) |delta_k|^2 W^2(kR) for every
 * j from 0 to max_order in a single pass, normalized so that j = 0 reproduces
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
 * than by the field. Use FILTER_GAUSSIAN when sigma_2 has to mean something,
 * and read high_k_fraction before trusting any moment.
 *
 * @param ws The fft workspace, after a forward transform
 * @param filter The filter to evaluate the moments under
 * @param r The smoothing radius
 * @param box_length The physical side length of the box
 * @param max_order The highest moment order, at most FFT_MAX_MOMENT_ORDER
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
int sif_fft_spectral_moments(const sif_fft_workspace_t* ws, sif_filter_type_t filter,
  real_t r, real_t box_length, uint8_t max_order, uint64_t n_tracers,
  double* sigma_sq, double* high_k_fraction);

/*
 * @brief Execute the complex-to-real fft, normalized and compacted
 *
 * The transform runs in place on the workspace's real-space buffer, which is
 * then stripped of its FFTW row padding so the result is a contiguous
 * n_cells^3 array.
 *
 * @param ws The fft workspace
 *
 * @return The buffer, still owned by the workspace (see
 * sif_fft_workspace_take_real_buffer), or NULL if the backward stage is missing
 */
real_t* sif_fft_grid_backward(sif_fft_workspace_t* ws);

#endif /* __SIF_FFT_H__ */
