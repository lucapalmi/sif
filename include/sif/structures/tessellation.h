#ifndef __SIF_TESSELLATION_H__
#define __SIF_TESSELLATION_H__

#include "sif/core/macros.h"
#include "sif/structures/field.h"

#include <stdint.h>

/*
 * @brief Represents a generalized volume and topology tessellation
 * using a Compressed Sparse Row (CSR) graph format.
 */
typedef struct {
  uint64_t num_particles;
  uint64_t num_edges;

  real_t* volumes;
  uint64_t* neighbor_offsets;
  uint64_t* neighbor_indices;

} sif_tessellation_t;

/*
 * @brief Builds a tessellation approximation.
 *
 * @param field              The input particle field.
 * @param supersample_factor How many tracers (fake particles) to generate per
 * real particle.
 * @param opt                Options
 */
NODISCARD sif_tessellation_t* sif_tessellation_build_approx(
  const sif_field_t* field, uint32_t supersample_factor, sif_option_t opt);

/*
 * @brief Safely frees a tessellation object and all its internal memory.
 *
 * @param tess The tessellation to free.
 */
void sif_tessellation_free(sif_tessellation_t* tess);

#endif /* __SIF_TESSELLATION_H__ */

