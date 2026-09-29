/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file catalogue.h
 * @brief Growable list of voids: centre and radius, one entry each.
 *
 * This is what a finder produces and what the measurement code consumes. It
 * grows by appending, since a finder does not know how many voids it will find
 * until it has finished looking.
 *
 * **Centres are Cartesian, or on the sky.** A finder produces comoving
 * Cartesian centres; for a survey, sif_catalogue_to_sky() turns them into right
 * ascension, declination and redshift, for a catalogue to hand on in the
 * survey's own terms. Every writer writes what the catalogue holds, and says
 * which; everything that measures around the centres refuses a sky
 * catalogue.
 */

#ifndef SIF_STRUCTURES_CATALOGUE_H
#define SIF_STRUCTURES_CATALOGUE_H

#include "sif/core/macros.h"
#include "sif/model/cosmology.h"

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
   * sif_catalogue_append() reads #SIF_CATALOGUE_FOOTPRINT_UNKNOWN here until
   * something measures it.
   */
  sif_real* footprint;
  /**
   * @brief The same fraction over the shell between one and two radii, which
   * is what says whether the void's surroundings were observed -- the part a
   * stacked profile reaches into. NULL exactly when #footprint is.
   */
  sif_real* footprint_shell;

  /**
   * What cx, cy and cz hold: #SIF_COORDINATES_CARTESIAN from
   * sif_catalogue_alloc() and from every finder, #SIF_COORDINATES_SKY after
   * sif_catalogue_to_sky() -- right ascension and declination in degrees, and
   * redshift. The radii are comoving lengths either way.
   */
  sif_coordinates_t units;

  /** Named values describing the catalogue; see the catalogue_meta group.
   * Owned; not for callers. */
  struct sif_catalogue_meta_entry* _meta;
  uint32_t _n_meta;

  /** Entries in use. */
  uint64_t n_voids;
  /** Entries the arena can hold before it must grow. */
  uint64_t capacity;
} sif_catalogue_t;

/**
 * @brief What a footprint column holds for a void nobody has measured it for.
 * Negative, so it cannot be mistaken for a fraction.
 */
#define SIF_CATALOGUE_FOOTPRINT_UNKNOWN ((sif_real)(-1.0))

/**
 * @brief Allocate an empty catalogue.
 *
 * @param initial_capacity Entries to make room for up front. Clamped up to 1,
 * so a successfully returned catalogue is always usable. Sizing it near the
 * expected void count avoids the copies that growth costs.
 * @return The catalogue, owned by the caller and released with
 * sif_catalogue_free(). NULL on allocation failure.
 */
SIF_NODISCARD sif_catalogue_t* sif_catalogue_alloc(uint64_t initial_capacity);

/**
 * @brief Release a catalogue and its arena.
 * @param catalogue Catalogue to free. NULL is accepted and ignored.
 */
void sif_catalogue_free(sif_catalogue_t* catalogue);

/**
 * @brief Append one void, doubling the capacity if it is full.
 *
 * Doubling rather than growing by a fixed step keeps the total copying linear
 * in the number of appends, which matters because a finder appends one void at
 * a time and may find millions.
 *
 * @param catalogue Catalogue to append to.
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
int sif_catalogue_append(
  sif_catalogue_t* catalogue, sif_real x, sif_real y, sif_real z, sif_real r);

/**
 * @brief Release the capacity a catalogue is not using.
 *
 * Worth calling once a finder has finished, since doubling leaves up to half
 * the arena unused and a catalogue is usually kept for the rest of the run.
 *
 * @param catalogue Catalogue to trim.
 * @return SIF_OK on success, including when there is nothing to trim.
 * SIF_ERR_ALLOC if the smaller buffer could not be allocated, in which case the
 * catalogue keeps its current larger allocation and stays fully valid.
 * SIF_ERR_INVALID on a NULL catalogue.
 *
 * @warning Invalidates cx/cy/cz/radii.
 */
int sif_catalogue_trim(sif_catalogue_t* catalogue);

/**
 * @brief Shift every void centre by @p offset.
 *
 * The way back out of the box a survey was searched in: pass the negated
 * offset that sif_field_translate() moved the survey in by, and the centres
 * return to the caller's own frame. Radii and footprints are unchanged.
 *
 * @param catalogue The catalogue, modified in place.
 * @param offset Added to cx, cy and cz respectively.
 * @return SIF_OK, or SIF_ERR_INVALID on a NULL argument or a sky catalogue.
 */
int sif_catalogue_translate(sif_catalogue_t* catalogue, const sif_real offset[3]);

/**
 * @brief Turn Cartesian void centres into sky coordinates, in place.
 *
 * The inverse of sif_field_convert_sky_coordinates(), for the voids a survey
 * gave: each centre becomes the right ascension and declination of its
 * direction from the origin, in degrees, and the redshift at which the
 * line-of-sight comoving distance in @p cosmo is its distance from the origin.
 * Right ascensions come out in [0, 360). The catalogue is then
 * #SIF_COORDINATES_SKY.
 *
 * The observer has to be at the origin, as sif_field_convert_sky_coordinates()
 * puts it: a catalogue found in a survey box is moved back first, with
 * sif_catalogue_translate() and the negated offset. Converted with the cosmology
 * the tracers were, a centre comes back to the sky within single precision.
 *
 * Radii are left as they are, comoving lengths in Mpc/h: a void's size is not
 * an angle. Footprints are untouched.
 *
 * @param catalogue The catalogue, with Cartesian centres; modified in place.
 * @param cosmo The cosmology the distances are converted in.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument, a catalogue already on
 * the sky, a centre that is not finite, or parameters
 * sif_cosmology_comoving_distance() refuses; SIF_ERR_RANGE for a centre
 * farther than the model reaches; SIF_ERR_ALLOC. On failure the catalogue is
 * left as it was.
 */
SIF_NODISCARD int sif_catalogue_to_sky(
  sif_catalogue_t* catalogue, const sif_cosmology_t* cosmo);

/**
 * @brief Give the catalogue its footprint columns, sif_catalogue_t::footprint
 * and sif_catalogue_t::footprint_shell.
 *
 * Optional because most catalogues have no use for them: a void found in a
 * periodic box is whole by construction. Once reserved they follow the
 * catalogue through every append and trim. A no-op if they already exist.
 *
 * @param catalogue The catalogue.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL catalogue, or SIF_ERR_ALLOC, in
 * which case the catalogue is left without them.
 *
 * @note Every void already in the catalogue starts at
 * #SIF_CATALOGUE_FOOTPRINT_UNKNOWN.
 */
int sif_catalogue_reserve_footprint(sif_catalogue_t* catalogue);

/**
 * @defgroup catalogue_meta Metadata
 * @brief Named values that describe a catalogue -- the finder and its
 * settings, the cosmology, where the tracers came from -- and travel with it.
 *
 * Every writer records them and every reader gives them back: the ASCII
 * header as `#key=value` lines, the HDF5 file as attributes of `/catalogue`,
 * the FITS file as keywords of the VOIDS table. So a catalogue says how it
 * was made whatever format it went through.
 *
 * Keys are identifiers -- a letter or '_', then letters, digits and '_', at
 * most 64 characters -- matched without regard to case and kept in lower
 * case, since FITS does not tell cases apart. Names the formats use for
 * themselves (`n`, `n_voids`, `coordinates`, and the FITS structural
 * keywords) are refused. Values are 64-bit integers, doubles or strings; a
 * string may not hold a newline or a double quote.
 *
 * Setting a key that exists replaces it, whatever its kind; the order keys
 * were first set in is kept, and is the order the writers use.
 * @{
 */

/** @brief What kind of value a key holds. */
typedef enum {
  SIF_CATALOGUE_META_MISSING = 0,
  SIF_CATALOGUE_META_INT,
  SIF_CATALOGUE_META_REAL,
  SIF_CATALOGUE_META_STRING
} sif_catalogue_meta_kind_t;

/**
 * @brief Set an integer value.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument or a key that is not
 * allowed; SIF_ERR_ALLOC.
 */
int sif_catalogue_meta_int_set(
  sif_catalogue_t* catalogue, const char* key, int64_t value);

/** @brief Set a real value; see sif_catalogue_meta_int_set(). Not finite is
 * refused, as no format but HDF5 could keep it. */
int sif_catalogue_meta_real_set(
  sif_catalogue_t* catalogue, const char* key, double value);

/** @brief Set a string value; see sif_catalogue_meta_int_set(). */
int sif_catalogue_meta_string_set(
  sif_catalogue_t* catalogue, const char* key, const char* value);

/** @brief Remove a key. SIF_OK whether or not it was there. */
int sif_catalogue_meta_remove(sif_catalogue_t* catalogue, const char* key);

/** @brief What kind of value a key holds, or #SIF_CATALOGUE_META_MISSING. */
sif_catalogue_meta_kind_t sif_catalogue_meta_kind(
  const sif_catalogue_t* catalogue, const char* key);

/** @brief An integer value; 0 for a key that is missing or not an integer. */
int64_t sif_catalogue_meta_int_get(const sif_catalogue_t* catalogue, const char* key);

/** @brief A numeric value, integer or real; 0 for a key that is missing or
 * holds a string. */
double sif_catalogue_meta_real_get(const sif_catalogue_t* catalogue, const char* key);

/**
 * @brief A string value, or NULL for a key that is missing or not a string.
 * The string belongs to the catalogue, and holds until the key is set again
 * or removed, or the catalogue freed.
 */
const char* sif_catalogue_meta_string_get(
  const sif_catalogue_t* catalogue, const char* key);

/** @brief How many keys the catalogue carries. */
uint32_t sif_catalogue_meta_count(const sif_catalogue_t* catalogue);

/** @brief The name of key @p index, in the order keys were first set; NULL
 * past the end. Owned by the catalogue, as for sif_catalogue_meta_string_get().
 */
const char* sif_catalogue_meta_name(const sif_catalogue_t* catalogue, uint32_t index);

/** @} */

#endif /* SIF_STRUCTURES_CATALOGUE_H */
