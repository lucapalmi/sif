#ifndef __SIF_DELTA_COMMON_H__
#define __SIF_DELTA_COMMON_H__

/*
 * Internals shared by the delta distribution and the delta moment estimators.
 * Not a public header: both estimators start from the same prepared spectrum,
 * and duplicating that preparation is how the two would drift apart.
 */

#include "math/fft.h"
#include "sif/core/macros.h"
#include "sif/structures/grid.h"

#define __SIF_DELTA_TAG "delta"

/* Below this many cells per radius the k-space window is aliased badly enough
 * that the result is grid artefacts rather than field. */
#define __SIF_DELTA_MIN_CELLS_PER_RADIUS 2.0

/*
 * @brief Validates radii against the grid resolution and the box size.
 *
 * A radius below a couple of cells is not a smaller measurement, it is a
 * different (wrong) one, so this fails rather than returning a plausible
 * looking answer.
 *
 * @return SIF_OK or SIF_ERR_INVALID
 */
int __sif_delta_validate_radii(
  const sif_grid_t* grid, const real_t* radii, uint32_t n_radii);

/*
 * @brief Resolves the smoothing window selected in the options bitmask.
 */
filter_type_t __sif_delta_filter(sif_option_t opt);

/*
 * @brief Validates the parts of the options bitmask both estimators share.
 *
 * @return SIF_OK or SIF_ERR_INVALID
 */
int __sif_delta_validate_options(sif_option_t opt);

/*
 * @brief Builds the spectrum both estimators work from.
 *
 * Allocates a workspace, runs the forward transform, deconvolves the CIC
 * assignment window unless asked not to, and applies the phase shuffle. The
 * shuffle is a pure function of (seed, n_cells), so calling this twice with
 * the same seed -- once from each estimator -- yields the identical surrogate,
 * which is what lets the PDF and the moments describe the same field.
 *
 * The backward stage is *not* initialized: a caller that only needs the
 * moments never pays for it.
 *
 * @return The workspace, owned by the caller, or NULL on failure.
 */
NODISCARD fft_workspace_t* __sif_delta_prepare_spectrum(
  const sif_grid_t* grid, uint64_t seed, sif_option_t opt);

#endif /* __SIF_DELTA_COMMON_H__ */
