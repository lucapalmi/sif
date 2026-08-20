/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file tessellation.h
 * @brief Voronoi cell volumes, measured by sampling rather than constructed.
 *
 * Every point of space belongs to the tracer nearest it, so throwing uniform
 * random points into the box and asking each one which tracer owns it measures
 * the Voronoi partition directly: the fraction of the samples a tracer catches
 * is the fraction of the box its cell occupies. No cell is ever built, and the
 * estimate is unbiased -- E[n_i]/n_samples is exactly V_i/V_box -- with the
 * total conserved identically, since every sample lands in exactly one cell.
 *
 * This buys two different things.
 *
 * The first is #sif_tessellation_t::volumes, a density per tracer that adapts
 * to the local sampling instead of to a fixed grid, which is what a sparse
 * field wants: a grid cell in a void holds no tracers and says nothing, while
 * a Voronoi cell there is simply large.
 *
 * The second is #sif_tessellation_t::samples, and it is the more useful one.
 * Each sample carries a share of its owner's weight, so the samples *are* a
 * field whose weights sum to the tracer mass exactly, and whose density is the
 * tessellation's. Measuring that field measures the tessellation: binning the
 * samples that fall inside a sphere counts the fraction of every cell that the
 * sphere covers, which is the geometric intersection that constructing the
 * cells would otherwise be needed for. sif_profiles() run on a mesh of these
 * samples is the volume-weighted profile, with no changes to the estimator.
 *
 * @note This is for fields too sparse to profile by counting tracers. The
 * sample set is @p samples_per_tracer times the field, in both memory and in
 * the cost of everything measured from it, and a field dense enough to count
 * has nothing to gain here. Construction peaks about half again above the
 * result, for the map from each sample to its owner, which is released once
 * the weights are worked out.
 */

#ifndef SIF_STRUCTURES_TESSELLATION_H
#define SIF_STRUCTURES_TESSELLATION_H

#include <math.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

/**
 * @brief A sampled Voronoi tessellation of a tracer field.
 */
typedef struct {
  /** Tracers in the mesh this was built from. */
  uint64_t n_tracers;
  /** Samples thrown over the box. */
  uint64_t n_samples;
  /**
   * Tracers no sample landed in, whose #volumes entry is therefore zero and
   * whose weight is absent from #samples.
   *
   * Zero at any reasonable sampling rate -- it takes a cell smaller than one
   * sample's worth of space -- and the two profiles it can distort are the
   * ones that would have been dominated by that tracer anyway. Raise the
   * sampling rate if it is not zero.
   */
  uint64_t n_empty;
  /** Cells per side of the mesh this was built from, which is also the order
   * #volumes is in. */
  uint32_t n_cells;
  sif_real box_length;
  /** Volume one sample stands for: box_length^3 / #n_samples. */
  sif_real sample_volume;

  /**
   * Volume of each tracer's cell, indexed by mesh slot rather than by field
   * index -- the same order as sif_chain_mesh_t::x and the rest of the mesh's
   * payload, so it reads alongside them.
   */
  sif_real* volumes;

  /**
   * The samples, as a field: positions, a weight each, and velocities when the
   * mesh carried them.
   *
   * A sample's weight is its owner's weight divided between that owner's
   * samples, so the total is the tracer mass exactly and the mean density of
   * this field is the mean density of the tracers. Velocities are copied from
   * the owner unchanged, which makes a mean over samples a volume-weighted
   * mean rather than a tracer-weighted one.
   *
   * Emptied by sif_chain_mesh_alloc_tessellation_consume(), which is the only
   * thing that changes it.
   */
  sif_field_t* samples;
} sif_tessellation_t;

/**
 * @brief Sampling rate that costs nothing to defend.
 *
 * A cell's volume is measured to about 1/sqrt(rate), so this is a per-cell
 * error near 18%. That sounds loose and is not, for the use this exists for: a
 * cell wholly inside a sphere contributes its whole weight whatever the rate,
 * so the sampling error lives only in the cells the sphere's surface cuts,
 * where it replaces the whole-tracer granularity that counting is stuck with.
 */
#define SIF_TESSELLATION_DEFAULT_SAMPLES 32u

/**
 * @brief Samples per tracer needed for a given per-cell volume accuracy.
 *
 * A cell's volume comes from a count, so its relative error is one over the
 * root of that count: the rate needed is the inverse square of the error
 * asked for. Note what this does *not* describe -- the accuracy of anything
 * integrated over many cells, which is better than this by the root of how
 * many, and the accuracy of a profile, which is set by the tracers themselves
 * long before it is set by this.
 *
 * @param relative_volume_error Wanted error on a single cell's volume, as a
 * fraction. Values outside (0, 1] are read as
 * #SIF_TESSELLATION_DEFAULT_SAMPLES.
 * @return Samples per tracer, at least 1.
 */
static inline uint32_t sif_tessellation_suggest_samples(
  sif_real relative_volume_error) {

  if (!(relative_volume_error > (sif_real)0.0) ||
      relative_volume_error > (sif_real)1.0)
    return SIF_TESSELLATION_DEFAULT_SAMPLES;

  const double n =
    ceil(1.0 / ((double)relative_volume_error * (double)relative_volume_error));

  return (n < 1.0) ? 1u : (uint32_t)n;
}

/**
 * @brief Sample a field's Voronoi tessellation.
 *
 * Samples are stratified over the mesh's cells -- every cell receives the same
 * number, to within the one that cannot be divided evenly -- which both
 * removes the fluctuation in how many samples a region happens to receive and
 * keeps each nearest-neighbour walk inside a cell the previous sample just
 * touched.
 *
 * Reproducible: each cell's samples come from a generator seeded by @p seed
 * and the cell index, so the result does not depend on the thread count.
 *
 * @param mesh Tracers to tessellate, and the box they live in. Weights and
 * velocities ride along into the samples if the mesh carries them.
 * #SIF_MESH_DROP_INDICES is fine; nothing here names a tracer in field order.
 * @param samples_per_tracer Samples thrown per tracer, which sets both the
 * accuracy and the size of the result. See
 * sif_tessellation_suggest_samples() and #SIF_TESSELLATION_DEFAULT_SAMPLES.
 * Must be non-zero, and must give at least as many samples as the mesh has
 * cells -- the stratification puts at least one sample in every cell, and a
 * mesh too fine for that is refused rather than sampled unevenly. Any mesh
 * sized for its own tracers satisfies this at any rate.
 * @param seed Any 64-bit value.
 * @param opt Honours SIF_PBC_PERIODIC / SIF_PBC_OPEN, which decides whether a
 * sample near a face may be owned by a tracer across it.
 * @return The tessellation, owned by the caller and released with
 * sif_tessellation_free(). NULL on invalid input or allocation failure.
 */
SIF_NODISCARD sif_tessellation_t* sif_tessellation_alloc(
  const sif_chain_mesh_t* mesh, uint32_t samples_per_tracer, uint64_t seed,
  sif_option opt);

/**
 * @brief Release a tessellation and everything it owns.
 * @param tess Tessellation to free. NULL is accepted and ignored.
 */
void sif_tessellation_free(sif_tessellation_t* tess);

/**
 * @brief Bin the samples into a chain mesh, ready to measure.
 *
 * What turns a tessellation into a profile: hand the result to sif_profiles()
 * and the densities come out volume-weighted, since a sample stands for a
 * piece of space rather than for a tracer.
 *
 * @param tess Tessellation to bin. Its samples are copied and left intact.
 * @param n_cells Cells per side. Resolution only costs speed and memory; size
 * it from sif_tessellation_t::n_samples, not from the tracer count, since it
 * is the samples being binned.
 * @param opt As sif_chain_mesh_alloc().
 * @return The mesh, owned by the caller and released with
 * sif_chain_mesh_free(). NULL on invalid input or allocation failure.
 */
SIF_NODISCARD sif_chain_mesh_t* sif_chain_mesh_alloc_tessellation(
  const sif_tessellation_t* tess, uint32_t n_cells, sif_option opt);

/**
 * @brief Bin the samples into a chain mesh, taking their storage over.
 *
 * Identical to sif_chain_mesh_alloc_tessellation() in every observable way
 * except that the samples are moved rather than copied, which matters here more
 * than it does for a field: the sample set is the largest thing in the process
 * by a factor of the sampling rate, and copying it means holding two.
 *
 * @param tess Tessellation to bin, whose sif_tessellation_t::samples is
 * **consumed**. On return it is a valid but empty field, and the volumes and
 * the counts are untouched -- so the tessellation is still good for everything
 * except handing out its samples a second time.
 * @param n_cells Cells per side, as sif_chain_mesh_alloc_tessellation().
 * @param opt As sif_chain_mesh_alloc().
 * @return The mesh, owned by the caller and released with
 * sif_chain_mesh_free(). NULL on invalid input or allocation failure.
 *
 * @warning The samples are emptied whether or not this succeeds, exactly as in
 * sif_chain_mesh_alloc_consume().
 */
SIF_NODISCARD sif_chain_mesh_t* sif_chain_mesh_alloc_tessellation_consume(
  sif_tessellation_t* tess, uint32_t n_cells, sif_option opt);

/**
 * @brief Deposit a tessellation onto a grid by Cloud-In-Cell assignment.
 *
 * The tessellation's density rather than the tracers': every sample carries a
 * share of its owner's weight and lands where the owner's cell actually
 * reaches, so a region with few tracers is filled by whatever large cells
 * cover it instead of being left empty. That is the difference this exists
 * for -- depositing sparse tracers directly leaves most cells holding nothing
 * and the rest holding shot noise, and no amount of smoothing afterwards puts
 * back a field that was never sampled.
 *
 * The result is a density in the same units and with the same mean as
 * sif_grid_assign_cic() of the tracers themselves, since the sample weights
 * sum to the tracer weight exactly. Everything downstream --
 * sif_grid_to_density_contrast(), the finders, the delta statistics -- treats
 * it identically, including the deconvolution of the CIC window.
 *
 * Declared here rather than in grid.h so that a grid needs to know nothing
 * about tessellations.
 *
 * @param grid Destination grid, overwritten. Must span the same box as the
 * tessellation.
 * @param tess Tessellation to deposit. Its samples must still be present, so
 * not one whose samples sif_chain_mesh_alloc_tessellation_consume() has taken.
 *
 * @note Costs what depositing the samples costs, which is the sampling rate
 * times what depositing the tracers would. See the note on sif_tessellation_t
 * about which fields this is worth it for.
 *
 * @note Inherits the grid cache described on sif_grid_assign_cic(), keyed on
 * the samples rather than the tracers -- so a different seed or sampling rate
 * is a different key.
 */
void sif_grid_assign_cic_tessellation(
  sif_grid_t* grid, const sif_tessellation_t* tess);

#endif /* SIF_STRUCTURES_TESSELLATION_H */
