#ifndef __SIF_SIZEFUNCTION_H__
#define __SIF_SIZEFUNCTION_H__

#include <stdbool.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"
#include "sif/structures/sizefunction.h"

/*
 * @brief Computes the void size function for a given catalog
 *
 * @param cat The void catalog
 * @param box_length The physical size of the simulation box
 * @param n_bins Number of radial bins (must be >= 1)
 * @param options Bitmask for the binning (SIF_VSF_BIN_LN / _LINEAR)
 * @param r_min_in Minimum radius bound (<= 0 to auto-detect)
 * @param r_max_in Maximum radius bound (<= 0 to auto-detect)
 *
 * @return A newly allocated sif_size_function_t, or NULL on failure
 */
NODISCARD sif_size_function_t* sif_size_function_catalog(
  const sif_catalog_t* cat, real_t box_length, uint32_t n_bins,
  sif_option_t options, real_t r_min_in, real_t r_max_in);

/*
 * @brief Combines multiple Void Size Functions into a single master VSF.
 *
 * @param vsfs Array of pointers to the individual size functions
 * @param n_vsfs Number of size functions in the array
 * @param master_bins The requested number of radial bins for the final output
 * @param domains Array of length n_vsfs with valid intervals. If NULL, uses
 * native VSF limits.
 * @param options Bitmask defining the merge strategy
 *
 * @return A newly allocated sif_size_function_t containing the merged result
 */
NODISCARD sif_size_function_t* sif_size_function_combine(
  const sif_size_function_t** vsfs, uint32_t n_vsfs, uint32_t master_bins,
  const sif_interval_t* domains, sif_option_t options);

#endif /* __SIF_SIZEFUNCTION_H__ */
