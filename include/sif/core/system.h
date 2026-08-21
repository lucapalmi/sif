/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file system.h
 * @brief Library lifetime: initialization, configuration and teardown.
 *
 * sif_init() must be called before any other entry point and sif_finalize()
 * after the last one. Between them the library holds process-global state: the
 * logger, the thread ceiling, the FFTW plan cache and the settings table.
 */

#ifndef SIF_CORE_SYSTEM_H
#define SIF_CORE_SYSTEM_H

#include <stdbool.h>
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
   * so a zero-initialized sif_config_t asks for trace logging, not for the
   * default. Set it explicitly, or use one of the presets below.
   */
  uint8_t log_level;
} sif_config_t;

/**
 * @brief Initialize the library.
 *
 * Brings up the logger, the thread ceiling, the settings table and the FFTW
 * plan cache. Calling it a second time logs a warning and does nothing.
 *
 * @param config Configuration to apply, or NULL for the defaults. It is read
 * once and copied, so neither the struct nor the sub-structs it points at need
 * to outlive the call, and the library never frees them.
 *
 * @note Terminates the process if the system state cannot be allocated, on the
 * grounds that a caller has no useful way to continue from that. Every other
 * failure is reported and survived: a run that cannot bring up FFTW, or cannot
 * create its cache directories, still does everything that does not need them.
 */
void sif_init(sif_config_t* config);

/**
 * @brief Shut the library down, releasing everything sif_init() acquired.
 *
 * Saves the settings table if it changed and tears down the FFTW plan cache.
 * No sif entry point may be called afterwards without initializing again.
 */
void sif_finalize(void);

/**
 * @defgroup config_presets Configuration presets
 * @brief Ready-made sif_config_t pointers for the common cases.
 *
 * @code
 * sif_init(SIF_CONFIG_STANDARD);
 * @endcode
 *
 * @warning These expand to compound literals, whose lifetime ends with the
 * enclosing block. Pass one straight to sif_init(); do not store the pointer.
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
