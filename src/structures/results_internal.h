#ifndef __SIF_RESULTS_INTERNAL_H__
#define __SIF_RESULTS_INTERNAL_H__

/*
 * Allocators for the result containers in structures/.
 *
 * Not a public header. The containers are only meaningful once an estimator
 * has filled them in, so the public API exposes the struct and its free and
 * nothing else. But several of them have more than one producer -- a moment
 * set can come from a grid or from a model spectrum -- sitting in different
 * modules, and duplicating the allocator in each is how the two drift apart
 * when a field is added. They live here instead, next to the free that has to
 * match them.
 */

#include "sif/structures/deltadistribution.h"
#include "sif/structures/deltamoments.h"
#include "sif/structures/sizefunction.h"

NODISCARD sif_delta_moments_t* __sif_delta_moments_alloc(
  uint32_t n_radii, uint8_t order);

NODISCARD sif_delta_distribution_t* __sif_delta_distribution_alloc(
  uint32_t n_radii, uint32_t n_bins);

NODISCARD sif_size_function_t* __sif_size_function_alloc(uint32_t n_bins);

/*
 * @brief Reconstructs bin edges from the radii a model was evaluated at.
 *
 * A measured size function is a histogram and its edges are the grid it was
 * binned on. A model is evaluated pointwise, so there is no such grid: the
 * radii are centres and the edges are geometric midpoints, present so the
 * container is complete and plottable rather than because the model
 * integrated over them.
 *
 * @param centers n entries, strictly positive
 * @param n Number of centres, at least 2
 * @param edges Output, n + 1 entries
 */
void __sif_edges_from_centers(const real_t* centers, uint32_t n, real_t* edges);

#endif /* __SIF_RESULTS_INTERNAL_H__ */
