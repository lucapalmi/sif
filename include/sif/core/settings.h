/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file settings.h
 * @brief Runtime settings: a fixed set of keys, each read from the first of
 * these that gives it a value:
 *
 * 1. a value set for this process: with sif_setting_override(), or in
 *    sif_config_t::settings at sif_system_init()
 * 2. the environment variable `SIF_<KEY>`, e.g. `SIF_FFT_TUNING_MAX_GIB`
 * 3. the settings file, `$SIF_HOME/config`, or `$HOME/.sif/config` when
 *    SIF_HOME is unset; sif_setting_set() writes to it
 * 4. the library's default
 *
 * The keys, with their defaults:
 *
 * `cache_directory`, `$SIF_HOME/grid_cache`
 *   Where cached grids go.
 *
 * `grid_cache_enabled`, `0`
 *   Cache CIC grids on disk.
 *
 * `fft_wisdom_dir`, `$SIF_HOME/wisdoms`
 *   Where FFTW's tuned plans are kept.
 *
 * `fft_tuning_max_gib`, `16`
 *   See #SIF__FFT_TUNING_MAX_GIB_DEFAULT.
 *
 * `cic_tile_particles`, `67108864`
 *   See #SIF__CIC_TILE_PARTICLES_DEFAULT.
 *
 * `scratch_directory`, `$SIF_HOME/scratch`
 *   Where `save_memory` runs put the grid; see sif_finder_exodus_extras_t.
 *
 * The file and the environment are read by sif_system_init(), and the file is
 * written back by sif_system_finalise() if sif_setting_set() changed it --
 * with every key listed and documented, those at their default commented out.
 * The first run writes one for the user to edit.
 *
 * Values may reference environment variables (`$HOME`, `${USER}`), expanded
 * when the value is loaded or set; `$SIF_HOME` always expands to the directory
 * holding the file. So a file shared by every job on a cluster can still say
 * `cache_directory = $TMPDIR/sif`, and a job that wants something else for
 * itself exports `SIF_CACHE_DIRECTORY` rather than editing the file. A path
 * naming a variable the process does not set is refused where it is used, not
 * read as if the variable were empty.
 *
 * A value that cannot be read as what its key holds -- a word for a number --
 * reads as the default, with a warning naming the file and line, or the
 * variable, it came from.
 *
 * @warning The table is not synchronised: sif_setting_set() and
 * sif_setting_override() race with any other access. Reads alone are safe.
 */

#ifndef SIF_CORE_SETTINGS_H
#define SIF_CORE_SETTINGS_H

#include <stddef.h>

#include "sif/core/macros.h"

/**
 * @brief Default for the `fft_tuning_max_gib` setting.
 *
 * FFTW_MEASURE picks its plan by *running* the transform, repeatedly, so the
 * cost of tuning grows with the transform while the benefit does not: one
 * better plan, however large the grid. Past some size the planning outlasts
 * the run it was meant to speed up -- measured on a 2250^3 grid it does not
 * finish in any useful time at all -- so above this many GiB of spectrum the
 * library plans both transforms with FFTW_ESTIMATE and says so.
 *
 * 16 GiB sits just above a 1500^3 single-precision spectrum (12.6 GiB), which
 * tunes in reasonable time, and below 2048^3 (32 GiB), which does not. Raise it
 * if a grid between those is worth the wait; 0 turns tuning off outright.
 *
 * This is a ceiling, not a request: `skip_tuning` still decides whether tuning
 * is wanted at all, and the cap only ever takes it away.
 */
#define SIF__FFT_TUNING_MAX_GIB_DEFAULT "16"

/**
 * @brief Default for the `cic_tile_particles` setting.
 *
 * The CIC deposit sorts particles by destination slab through an index array
 * of 8 bytes each, and does it a tile at a time so that array does not scale
 * with the field: 512 MiB here, against 27 GB for a 3.4e9-tracer field sorted
 * in one go and 69 GB at 8.6e9.
 *
 * Tiling is also the faster arrangement -- the deposit's indirect reads stay
 * inside a slice of the coordinates instead of ranging over all of them -- and
 * the gain is broad enough that this does not want tuning. Lower it if the
 * scratch is still too much; the cost is one extra pass over the slabs per
 * tile, which only starts to show when the tiles get small.
 */
#define SIF__CIC_TILE_PARTICLES_DEFAULT "67108864"

/**
 * @brief Set a setting in the settings file, from now on and for later runs.
 *
 * Takes effect in this process unless the environment or
 * sif_setting_override() gives the key a value of its own, which is warned
 * about. Environment references in @p value are expanded now for this process,
 * and stored as written for the next.
 *
 * Values longer than 255 characters are refused. One that only grows past that
 * when its variables are expanded is refused where it is read.
 *
 * @param key One of the keys listed in settings.h.
 * @param value Raw value, possibly containing environment references.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL key or value, a key that is not a
 * setting, a value its key cannot hold (a word for a number), or a value
 * containing a newline, which the file cannot hold; SIF_ERR_UNINITIALISED
 * before sif_system_init(). The table is unchanged on failure.
 *
 * @note Requires sif_system_init().
 */
SIF_NODISCARD int sif_setting_set(const char* key, const char* value);

/**
 * @brief Set a setting for this process only, over the environment and the
 * file; nothing is saved.
 *
 * @param key One of the keys listed in settings.h.
 * @param value Raw value, as for sif_setting_set(); NULL removes the
 * override, and the key reads as it would without it.
 * @return As sif_setting_set().
 *
 * @note Requires sif_system_init().
 */
SIF_NODISCARD int sif_setting_override(const char* key, const char* value);

/**
 * @brief Whether @p value is one @p key can be set to, without setting it.
 *
 * The checks sif_setting_set() and sif_setting_override() make, for a caller
 * that wants to report a bad value its own way -- a configuration file that
 * names the line, say. A path is only checked for being there, since what its
 * variables expand to is known when it is read.
 *
 * @param reason Receives why the value is refused, when it is; may be NULL.
 * @return SIF_OK, or SIF_ERR_INVALID. Logs nothing.
 *
 * @note Does not require sif_system_init().
 */
SIF_NODISCARD int sif_setting_check(
  const char* key, const char* value, char* reason, size_t reason_size);

/**
 * @brief Look up a setting.
 *
 * Nothing is stored: a key that is not a setting reads as @p fallback, and a
 * key that is reads as its value, which is never NULL.
 *
 * @param key Setting name.
 * @param fallback Returned when @p key is not a setting, or before
 * sif_system_init(). May be NULL.
 * @return The expanded value, owned by the settings table -- the caller must
 * neither free nor modify it -- or @p fallback.
 *
 * @warning The returned string is overwritten by the next sif_setting_set() or
 * sif_setting_override() of the same key. Copy it if it has to outlive that.
 *
 * @note Before sif_system_init() the call returns @p fallback.
 */
const char* sif_setting_get(const char* key, const char* fallback);

#endif /* SIF_CORE_SETTINGS_H */
