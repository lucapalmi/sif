#ifndef __SIF_STRUCT_SIZEFUNCTION_H__
#define __SIF_STRUCT_SIZEFUNCTION_H__

#include <stdint.h>

#include "sif/core/macros.h"

/*
 * @brief Container for the Void Size Function.
 *
 * Produced either by binning a catalogue (sif_size_function_catalog) or by a
 * model; released with sif_size_function_free. `counts` and `err` carry the
 * Poisson bookkeeping of a measurement and are left at zero by a model.
 */
typedef struct {
  uint32_t n_bins;
  sif_option_t options;

  real_t r_min;
  real_t r_max;

  real_t* r_edges;   /* n_bins + 1 entries */
  real_t* r_centers; /* n_bins entries */
  uint64_t* counts;  /* raw voids per bin */
  real_t* vsf;       /* normalized number density per bin */
  real_t* err;       /* Poisson error on vsf */
} sif_size_function_t;

/*
 * @brief A physical radial domain interval.
 */
typedef struct {
  real_t min;
  real_t max;
} sif_interval_t;

void sif_size_function_free(sif_size_function_t* vsf);

/*
 * @brief The normalized number density in one bin.
 */
static inline real_t sif_size_function_get(
  const sif_size_function_t* vsf, uint32_t bin_idx) {
  return vsf->vsf[bin_idx];
}

#endif /* __SIF_STRUCT_SIZEFUNCTION_H__ */
