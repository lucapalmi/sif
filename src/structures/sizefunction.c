#include "sif/structures/sizefunction.h"

#include "results_internal.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>

sif_size_function_t* sif_size_function_alloc(uint32_t n_bins) {

  sif_size_function_t* vsf = calloc(1, sizeof(sif_size_function_t));
  if (!vsf) {
    SIF_LOG_ERROR("size_function", "failed to allocate the VSF struct");
    return NULL;
  }

  vsf->n_bins = n_bins;

  vsf->r_edges = malloc(((size_t)n_bins + 1) * sizeof(real_t));
  vsf->r_centers = malloc((size_t)n_bins * sizeof(real_t));
  vsf->counts = calloc(n_bins, sizeof(uint64_t));
  vsf->vsf = calloc(n_bins, sizeof(real_t));
  vsf->err = calloc(n_bins, sizeof(real_t));

  if (!vsf->r_edges || !vsf->r_centers || !vsf->counts || !vsf->vsf ||
      !vsf->err) {
    SIF_LOG_ERROR("size_function", "failed to allocate the VSF arrays");
    sif_size_function_free(vsf);
    return NULL;
  }

  return vsf;
}

void sif_size_function_free(sif_size_function_t* vsf) {
  if (!vsf)
    return;

  free(vsf->r_edges);
  free(vsf->r_centers);
  free(vsf->counts);
  free(vsf->vsf);
  free(vsf->err);
  free(vsf);
}

void sif_edges_from_centers(
  const real_t* centers, uint32_t n, real_t* edges) {

  for (uint32_t i = 1; i < n; i++)
    edges[i] = (real_t)sqrt((double)centers[i - 1] * (double)centers[i]);

  /* Extrapolate the outer two, which reproduces the exact midpoints when the
   * radii are geometrically spaced. */
  edges[0] = (real_t)((double)centers[0] * centers[0] / (double)edges[1]);
  edges[n] =
    (real_t)((double)centers[n - 1] * centers[n - 1] / (double)edges[n - 1]);
}
