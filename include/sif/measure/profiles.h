#ifndef __SIF_PROFILES_H__
#define __SIF_PROFILES_H__

#include <stdint.h>

#include "sif/structures/catalog.h"
#include "sif/structures/field.h"
#include "sif/structures/tessellation.h"

/*
 * @brief Container for the 1D radial density profiles of voids
 */
typedef struct {
  uint64_t n_voids;
  uint32_t n_bins;
  real_t ext;
  real_t* r_edges;
  real_t* profiles;
} sif_density_profiles_t;

/*
 * @brief Container for the 1D radial velocity profiles of voids
 */
typedef struct {
  uint64_t n_voids;
  uint32_t n_bins;
  real_t ext;
  real_t* r_edges;
  real_t* v_rad;
} sif_velocity_profiles_t;

/* Allocation is internal: both containers are only meaningful once an
 * estimator has filled them in, so they are obtained from sif_profiles_mesh
 * or sif_profiles_voronoi and released here. */

void sif_density_profiles_free(sif_density_profiles_t* profs);
void sif_velocity_profiles_free(sif_velocity_profiles_t* profs);

/* --- Inline Getters --- */

static inline real_t* sif_density_profiles_get(
  const sif_density_profiles_t* profs, uint64_t void_idx) {
  return &profs->profiles[void_idx * profs->n_bins];
}

static inline real_t* sif_velocity_profiles_get(
  const sif_velocity_profiles_t* profs, uint64_t void_idx) {
  return &profs->v_rad[void_idx * profs->n_bins];
}

/*
 * @brief Computes volume-weighted profiles using a basic Grid/Mesh approach.
 */
void sif_profiles_mesh(const sif_catalog_t* cat,
  const sif_field_t* field, real_t box_length, real_t ext, uint32_t n_bins,
  sif_option_t opt, sif_density_profiles_t** out_dens,
  sif_velocity_profiles_t** out_vel);

/*
 * @brief Computes volume-weighted profiles natively from the Voronoi
 * tessellation.
 *
 * @warning EXPERIMENTAL. This path sweeps a 100^3 voxel grid per void and runs
 * a nearest-neighbour query at every voxel, so it costs ~1e6 queries per void
 * and does not scale to production catalogs. It is kept for future work and is
 * not exercised by the test suite. Use sif_profiles_mesh instead.
 */
void sif_profiles_voronoi(const sif_catalog_t* cat,
  const sif_field_t* field, const sif_tessellation_t* tess, real_t box_length,
  real_t ext, uint32_t n_bins, sif_option_t opt,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel);

#endif /* __SIF_PROFILES_H__ */
