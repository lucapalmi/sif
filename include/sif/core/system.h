/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file system.h
 * @brief The system state: FFTW, the settings file, threads and logging.
 *
 * sif_system_init() brings up the process-global resources a few parts of the
 * library share, and sif_system_finalise() releases them:
 *
 * - FFTW: its threading, the planner quality and the wisdom file
 * - the settings table, read from the environment and from
 *   `$SIF_HOME/config` (`$HOME/.sif/config` when SIF_HOME is unset), and
 *   saved back there; see settings.h
 * - the thread ceiling
 * - the log level
 *
 * It is not a library-wide initialisation. Most of sif works without it:
 * the containers, I/O, the analytic models and everything else that does not
 * transform a grid. The functions that do need it say so in their
 * documentation, with the note "Requires sif_system_init()":
 *
 * - FFTW: sif_finder_spherical(), sif_finder_exodus(),
 *   sif_finder_exodus_survey(), sif_delta_moments_grid() and
 *   sif_delta_distribution_grid()
 * - the settings table: sif_setting_set(), sif_setting_override() and
 *   sif_setting_get()
 *
 * Called without it, they fail with SIF_ERR_UNINITIALISED -- reported through
 * sif_error_status() by the ones that return a pointer -- except
 * sif_setting_get(), which returns its fallback. The rest of the library uses
 * the OpenMP runtime's own thread count and logs at INFO until it is called.
 */

#ifndef SIF_CORE_SYSTEM_H
#define SIF_CORE_SYSTEM_H

#include <stdbool.h>
#include <stddef.h> /* NULL, in the SIF_CONFIG_* presets */
#include <stdint.h>

#include "sif/core/macros.h"

/**
 * @brief FFTW3 configuration.
 */
typedef struct {
  /** Skip plan tuning and take FFTW's estimated plan instead. Faster to set
   * up, slower to transform; worth setting for short runs, or for runs with
   * many distinct transform sizes.
   */
  bool skip_tuning;
} sif_fft_config_t;

/**
 * @brief OpenMP configuration.
 */
typedef struct {
  /** Thread ceiling. 0 leaves the OpenMP runtime's own default in place. */
  uint32_t n_threads;
} sif_omp_config_t;

/**
 * @brief General library configuration.
 */
typedef struct {
  /** FFTW settings, or NULL for the defaults. */
  sif_fft_config_t* fft_config;
  /** OpenMP settings, or NULL for the defaults. */
  sif_omp_config_t* omp_config;
  /** Minimum level a message must reach to be printed; one of the
   * SIF_LOG_LEVEL_* constants. SIF_LOG_LEVEL_TRACE is the verbose mode.
   *
   * @warning The field is always honoured, and SIF_LOG_LEVEL_TRACE is 0 --
   * so a zero-initialised sif_config_t asks for trace logging, not for the
   * default. Set it explicitly, or use one of the presets below.
   */
  uint8_t log_level;
} sif_config_t;

/**
 * @brief Bring up the system state.
 *
 * Applies the log level and thread ceiling, loads the settings table and
 * initialises FFTW. Calling it a second time logs a warning, keeps the first
 * configuration, and returns SIF_OK.
 *
 * @param config Configuration to apply, or NULL for the defaults. It is read
 * once and copied, so neither the struct nor the sub-structs it points at need
 * to outlive the call, and the library never frees them.
 * @return SIF_OK, or SIF_ERR_ALLOC if the state or FFTW could not be brought
 * up -- in which case nothing is left initialised, and the call may be
 * retried.
 *
 * @note Failing to create the cache or wisdom directories is only a warning:
 * the run still does everything that does not need them.
 */
SIF_NODISCARD int sif_system_init(sif_config_t* config);

/**
 * @brief Release everything sif_system_init() acquired.
 *
 * Saves the settings table if it changed and tears down the FFTW plan cache.
 * The functions that require sif_system_init() may not be called afterwards
 * without initialising again.
 */
void sif_system_finalise(void);

/**
 * @brief Deprecated name of sif_system_init(); logs a warning and forwards.
 */
SIF_DEPRECATED("use sif_system_init()")
SIF_NODISCARD int sif_init(sif_config_t* config);

/**
 * @brief Deprecated name of sif_system_finalise(); logs a warning and
 * forwards.
 */
SIF_DEPRECATED("use sif_system_finalise()")
void sif_finalise(void);

/**
 * @defgroup config_presets Configuration presets
 * @brief Ready-made sif_config_t pointers for the common cases.
 *
 * @code
 * if (sif_system_init(SIF_CONFIG_STANDARD) != SIF_OK)
 *   return 1;
 * @endcode
 *
 * @warning These expand to compound literals, whose lifetime ends with the
 * enclosing block. Pass one straight to sif_system_init(); do not store the
 * pointer.
 * @{
 */

/** @brief Trace-level logging, timings reported. */
#define SIF_CONFIG_VERBOSE                                                     \
  (&(sif_config_t){                                                            \
    .fft_config = NULL, .omp_config = NULL, .log_level = SIF_LOG_LEVEL_TRACE})

/** @brief Warnings and errors only. */
#define SIF_CONFIG_QUIET                                                       \
  (&(sif_config_t){.fft_config = NULL,                                         \
    .omp_config = NULL,                                                        \
    .log_level = SIF_LOG_LEVEL_WARNING})

/** @brief The default: informational messages and above. */
#define SIF_CONFIG_STANDARD                                                    \
  (&(sif_config_t){                                                            \
    .fft_config = NULL, .omp_config = NULL, .log_level = SIF_LOG_LEVEL_INFO})

/** @} */

#endif /* SIF_CORE_SYSTEM_H */
