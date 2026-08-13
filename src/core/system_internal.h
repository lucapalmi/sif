/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file system_internal.h
 * @brief Access to the process-global system state. Private to the library.
 *
 * The state itself lives in system.c and is created by sif_init(). Everything
 * here assumes the library has been initialized.
 */

#ifndef SIF__CORE_SYSTEM_INTERNAL_H
#define SIF__CORE_SYSTEM_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>

#include "math/fft.h"
#include "sif/utils/timer.h"

/** @brief Logger configuration, resolved from sif_config_t at init. */
typedef struct {
  /** Minimum level a message must reach to be printed. */
  uint8_t level;
  /** Report timings and trace-level detail. */
  bool verbose;
} sif_logger_state_t;

/** @brief Everything the library holds between init and finalize. */
typedef struct {
  sif_logger_state_t logger;

  uint8_t save_memory;
  /** Thread ceiling actually in force, as reported by the OpenMP runtime. */
  uint32_t max_threads;
  sif_fft_manager_t* fft_mgr;

  sif_timer_t total_runtime_timer;
  sif_timer_t scratch_timer;
} sif_system_state_t;

/**
 * @brief The system state, aborting if the library is not initialized.
 *
 * For call sites that cannot do anything useful without it, which is most of
 * them.
 */
sif_system_state_t* sif__system_state(void);

/**
 * @brief The system state, or NULL if the library is not initialized.
 *
 * For the few call sites that must be able to run either way -- the logger
 * itself, and teardown paths.
 */
sif_system_state_t* sif__system_state_safe(void);

/**
 * @defgroup omp_wrappers OpenMP wrappers
 * @brief Thread-count queries that also work in a build without OpenMP, where
 * they report a single thread.
 * @{
 */
int sif__system_max_threads(void);
int sif__system_num_threads(void);
int sif__system_thread_num(void);
/** @} */

/**
 * @defgroup settings_lifetime Settings table lifetime
 * @brief Called by sif_init() and sif_finalize() only.
 * @{
 */

/**
 * @brief Load the settings table, creating @p default_dir if necessary.
 * @param default_dir Directory holding the settings file.
 */
void sif__settings_init(const char* default_dir);

/** @brief Write the table back if it changed, then release it. */
void sif__settings_finalize(void);

/** @} */

#endif /* SIF__CORE_SYSTEM_INTERNAL_H */
