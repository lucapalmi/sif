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
 * towards it whatever its distance: a dense sample of the mask even where
 * n(z) is sparse. Their density on the sky need not be uniform -- randoms
 * drawn with an angular completeness, denser where it is higher, are
 * measured as they are. The sky is cut into cells, finer where the randoms
 * are denser; a cell whose neighbours all hold randoms lies wholly inside
 * the mask and counts whole, and a cell an edge or a hole cuts through
 * counts by the randoms it holds against the density of the whole cells
 * nearest it. The edges are therefore resolved statistically, not to the
 * nearest cell, and holes down to the scale of a few randoms' separation.
 *
 * Against shapes of known volume -- spheres, shells and wedges with a steep
 * n(z), wedges with star-mask holes, randoms whose density steps or varies
 * pixel by pixel across the sky -- it is unbiased to within about its
 * scatter, which is about 0.5% at 10^4 randoms, 0.05-0.1% at 10^5 and 0.015%
 * at 10^6. Only where the density changes sharply right at the mask's edge,
 * by several times over a few randoms' separation, is it biased, low by up
 * to a few tenths of a per cent: no count tells there how much of a cell
 * the mask covers.
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
