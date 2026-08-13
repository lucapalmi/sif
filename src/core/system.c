/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/core/system.h"
#include "system_internal.h"

#include "sif/core/settings.h"
#include "sif/utils/logger.h"
#include "sif/utils/timer.h"

#include "predicates/predicates.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static void ensure_dir(const char* path) {
  char tmp[512];
  snprintf(tmp, sizeof(tmp), "%s", path);
  size_t len = strlen(tmp);
  if (tmp[len - 1] == '/')
    tmp[len - 1] = 0;
  for (char* p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      mkdir(tmp, 0777);
      *p = '/';
    }
  }
  mkdir(tmp, 0777);
}

static sif_system_state_t* system_state = NULL;

void sif_init(sif_config_t* config) {

  if (system_state) {
    SIF_LOG_WARNING("system", "sif library already initialized");
    return;
  }

  system_state = malloc(sizeof(sif_system_state_t));
  if (!system_state) {
    SIF_LOG_ERROR("system", "failed to allocate sif system state (%zu bytes)",
      sizeof(sif_system_state_t));
    exit(EXIT_FAILURE);
    return;
  }

  bool req_verbose = false;
  uint8_t req_log_level = SIF_LOG_LEVEL_INFO;
  uint32_t req_threads = 0; /* default all cores */

  /* FFTW defaults */
  bool req_skip_tuning = false;
  const char* req_wisdom_dir = NULL;

  if (config) {
    req_verbose = config->verbose;

    if (req_verbose) {
      req_log_level = SIF_LOG_LEVEL_TRACE;
    } else if (config->log_level != 0 || !req_verbose) {
      req_log_level = config->log_level;
    }

    if (config->omp_config) {
      req_threads = config->omp_config->n_threads;
    }
  }

  system_state->logger.verbose = req_verbose;
  system_state->logger.level = req_log_level;

  sif_timer_start(&system_state->total_runtime_timer);

  /* --- OpenMP Configuration --- */
  system_state->max_threads = 1;

#ifdef _OPENMP
  if (req_threads > 0) {
    omp_set_num_threads(req_threads); // Set the ceiling if explicitly requested
  }
  system_state->max_threads = omp_get_max_threads();
#else
  if (req_threads > 1) {
    SIF_LOG_WARNING("system",
      "requested %u threads, but compiled without OpenMP", req_threads);
  }
#endif

  /* --- Setup Root Directory & Settings Engine --- */
  const char* home = getenv("HOME");
  char base_dir[512];
  if (home) {
    snprintf(base_dir, sizeof(base_dir), "%s/.sif", home);
  } else {
    snprintf(base_dir, sizeof(base_dir), "/tmp/.sif");
    SIF_LOG_WARNING(
      "system", "HOME env variable not set, defaulting cache to %s", base_dir);
  }

  ensure_dir(base_dir);

  /* Boot the configuration engine FIRST from the base .sif directory */
  sif__settings_init(base_dir);

  /* Parse directories from settings */
  const char* cache_dir = sif_setting_get("cache_directory", NULL);
  if (cache_dir) {
    ensure_dir(cache_dir);
  }

  /* --- FFTW Configuration --- */
  req_wisdom_dir = sif_setting_get("fft_wisdom_dir", NULL);

  if (config && config->fft_config) {
    req_skip_tuning = config->fft_config->skip_tuning;
  }

  if (req_wisdom_dir) {
    ensure_dir(req_wisdom_dir);
  }

  system_state->fft_mgr =
    sif__fft_manager_init(req_skip_tuning, req_wisdom_dir);

  real_fftw_plan_with_nthreads(sif__system_max_threads());

  /* --- Exact Math Init --- */
  exactinit();

  SIF_LOG_INFO("system", "sif library initialized (threads: %d)",
    system_state->max_threads);

  SIF_LOG_FLUSH();
}

void sif_finalize() {
  if (!system_state) {
    SIF_LOG_WARNING("system", "sif library not currently initalized");
    return;
  }

  sif_timer_stop(&system_state->total_runtime_timer);
  double total_time = sif_timer_elapsed_ms(&system_state->total_runtime_timer);
  SIF_LOG_INFO("system", "sif library finalized. Total execution time: %.2f s",
    total_time / 1000);

  SIF_LOG_FLUSH();

  /* FFTW saves wisdom during its finalize step */
  sif__fft_manager_finalize(system_state->fft_mgr);

  /* Shut down settings engine and trigger auto-save if modified */
  sif__settings_finalize();

  free(system_state);
  system_state = NULL;
}

sif_system_state_t* sif__system_state() {
  if (!system_state) {
    SIF_LOG_ERROR("system", "system not initialized. aborting");
    exit(EXIT_FAILURE);
  }

  return system_state;
}

sif_system_state_t* sif__system_state_safe() { return system_state; }

int sif__system_max_threads(void) {
  if (system_state)
    return (int)system_state->max_threads;
  return 1;
}

int sif__system_num_threads(void) {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

int sif__system_thread_num(void) {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}
