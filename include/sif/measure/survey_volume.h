/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file survey_volume.h
 * @brief The comoving volume a survey covers, from its random catalogue.
 */

#ifndef SIF_MEASURE_SURVEY_VOLUME_H
#define SIF_MEASURE_SURVEY_VOLUME_H

#include "sif/core/macros.h"
#include "sif/structures/field.h"

/**
 * @brief Measure the comoving volume of a survey's footprint from its
 * randoms.
 *
 * The footprint is taken to be **separable**: one angular mask, seen out to
 * the same range of distances in every direction. That is how survey randoms
 * are made -- directions drawn over the mask, distances from n(z),
 * independently -- and it is what makes the volume measurable to the
 * accuracy below. The volume is then `V = Omega (r_max^3 - r_min^3) / 3`,
 * with r_min and r_max the randoms' nearest and furthest distances from the
 * observer, and Omega the solid angle of the mask.
 *
 * Omega comes from the randoms' directions alone, so every random counts
 * towards it whatever its distance: a dense, uniform sample of the mask even
 * where n(z) is sparse. Each random stands for 1 / sigma of solid angle,
 * where sigma is the randoms' density on the sky measured only over cells
 * the mask covers entirely. An edge or a hole therefore never dilutes it, and
 * a cell the edge cuts through counts by the randoms it holds rather than as
 * a whole cell, which is what keeps the edges from biasing the result. Holes
 * are resolved down to the scale of a few randoms' separation on the sky.
 *
 * Against shapes of known volume -- spheres, shells and wedges with a steep
 * n(z), wedges with star-mask holes -- it is unbiased to within its scatter,
 * which is about 0.3% at 10^4 randoms, 0.1% at 10^5 and 0.02% at 10^6.
 *
 * A catalogue whose radial range changes across the sky -- two regions with
 * different redshift cuts, say -- breaks the assumption. That is detected:
 * a sky patch whose randoms stop significantly short of r_min or r_max is
 * reported in a warning, and the volume is still returned. Measure such
 * regions one call each.
 *
 * Weights are ignored: this is the geometric volume.
 *
 * @param randoms The random catalogue, in comoving Cartesian coordinates.
 * Only read.
 * @param observer Where the observer is, in the same frame, or NULL for the
 * origin -- where sif_field_convert_sky_coordinates() puts it. Pass it if the
 * randoms have been moved since, e.g. by the offset of a survey box.
 * @param[out] volume The volume, in the cube of the positions' unit.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument, a field without
 * positions or still in sky coordinates, a position that is not finite, all
 * randoms at the observer, or too few randoms to find any part of the sky
 * the mask covers entirely; SIF_ERR_ALLOC.
 */
SIF_NODISCARD int sif_survey_volume(
  const sif_field_t* randoms, const sif_real observer[3], sif_real* volume);

#endif /* SIF_MEASURE_SURVEY_VOLUME_H */
