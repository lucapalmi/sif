/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file catalog.h
 * @brief Growable list of voids: centre and radius, one entry each.
 *
 * This is what a finder produces and what the measurement code consumes. It
 * grows by appending, since a finder does not know how many voids it will find
 * until it has finished looking.
 */

#ifndef SIF_STRUCTURES_CATALOG_H
#define SIF_STRUCTURES_CATALOG_H

#include "sif/core/macros.h"
#include <stdint.h>

/**
 * @brief A void catalogue.
 *
 * @note cx/cy/cz/radii are views into a single cache-line aligned arena
 * (#_block). They move whenever the capacity changes, so never cache them
 * across an append or a trim.
 */
typedef struct {
  /** Backing arena holding all four views. Owned; not for callers. */
  sif_real* _block;

  /** Void centres, x axis. */
  sif_real* cx;
  /** Void centres, y axis. */
  sif_real* cy;
  /** Void centres, z axis. */
  sif_real* cz;
  /** Void radii. */
  sif_real* radii;

  /** Backing arena for the two footprint views, or NULL when the catalogue
   * carries none. Owned; not for callers. */
  sif_real* _footprint_block;

  /**
   * @brief Fraction of each void's sphere that lies inside the survey
   * footprint, in [0, 1], or NULL for a catalogue that has no footprint -- a
   * periodic box, where every void is whole by construction.
   *
   * Written by sif_finder_exodus_survey(). A void added afterwards with
   * sif_catalog_append() reads #SIF_CATALOG_FOOTPRINT_UNKNOWN here until
   * something measures it.
   */
  sif_real* footprint;
  /**
   * @brief The same fraction over the shell between one and two radii, which
   * is what says whether the void's surroundings were observed -- the part a
   * stacked profile reaches into. NULL exactly when #footprint is.
   */
  sif_real* footprint_shell;

  /** Entries in use. */
  uint64_t n_voids;
  /** Entries the arena can hold before it must grow. */
  uint64_t capacity;
} sif_catalog_t;

/**
 * @brief What a footprint column holds for a void nobody has measured it for.
 * Negative, so it cannot be mistaken for a fraction.
 */
#define SIF_CATALOG_FOOTPRINT_UNKNOWN ((sif_real)(-1.0))

/**
 * @brief Allocate an empty catalogue.
 *
 * @param initial_capacity Entries to make room for up front. Clamped up to 1,
 * so a successfully returned catalogue is always usable. Sizing it near the
 * expected void count avoids the copies that growth costs.
 * @return The catalogue, owned by the caller and released with
 * sif_catalog_free(). NULL on allocation failure.
 */
SIF_NODISCARD sif_catalog_t* sif_catalog_alloc(uint64_t initial_capacity);

/**
 * @brief Release a catalogue and its arena.
 * @param catalog Catalogue to free. NULL is accepted and ignored.
 */
void sif_catalog_free(sif_catalog_t* catalog);

/**
 * @brief Append one void, doubling the capacity if it is full.
 *
 * Doubling rather than growing by a fixed step keeps the total copying linear
 * in the number of appends, which matters because a finder appends one void at
 * a time and may find millions.
 *
 * @param catalog Catalogue to append to.
 * @param x Void centre, x axis.
 * @param y Void centre, y axis.
 * @param z Void centre, z axis.
 * @param r Void radius.
 * @return SIF_OK on success. SIF_ERR_ALLOC if the catalogue could not grow, in
 * which case it is left untouched and the void is NOT stored. SIF_ERR_INVALID
 * on a NULL catalogue.
 *
 * @warning Invalidates cx/cy/cz/radii whenever it grows.
 */
int sif_catalog_append(
  sif_catalog_t* catalog, sif_real x, sif_real y, sif_real z, sif_real r);

/**
 * @brief Release the capacity a catalogue is not using.
 *
 * Worth calling once a finder has finished, since doubling leaves up to half
 * the arena unused and a catalogue is usually kept for the rest of the run.
 *
 * @param catalog Catalogue to trim.
 * @return SIF_OK on success, including when there is nothing to trim.
 * SIF_ERR_ALLOC if the smaller buffer could not be allocated, in which case the
 * catalogue keeps its current larger allocation and stays fully valid.
 * SIF_ERR_INVALID on a NULL catalogue.
 *
 * @warning Invalidates cx/cy/cz/radii.
 */
int sif_catalog_trim(sif_catalog_t* catalog);

/**
 * @brief Shift every void centre by @p offset.
 *
 * The way back out of the box a survey was searched in: pass the negated
 * offset that sif_field_translate() moved the survey in by, and the centres
 * return to the caller's own frame. Radii and footprints are unchanged.
 *
 * @param catalog The catalogue, modified in place.
 * @param offset Added to cx, cy and cz respectively.
 * @return SIF_OK, or SIF_ERR_INVALID on a NULL argument.
 */
int sif_catalog_translate(sif_catalog_t* catalog, const sif_real offset[3]);

/**
 * @brief Give the catalogue its footprint columns, sif_catalog_t::footprint
 * and sif_catalog_t::footprint_shell.
 *
 * Optional because most catalogues have no use for them: a void found in a
 * periodic box is whole by construction. Once reserved they follow the
 * catalogue through every append and trim. A no-op if they already exist.
 *
 * @param catalog The catalogue.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL catalogue, or SIF_ERR_ALLOC, in
 * which case the catalogue is left without them.
 *
 * @note Every void already in the catalogue starts at
 * #SIF_CATALOG_FOOTPRINT_UNKNOWN.
 */
int sif_catalog_reserve_footprint(sif_catalog_t* catalog);

#endif /* SIF_STRUCTURES_CATALOG_H */
