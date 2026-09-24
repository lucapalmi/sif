/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/core/system.h"

#include "sif/core/settings.h"
#include "sif/utils/logger.h"
#include "sif/utils/timer.h"

#include "core/system_internal.h"

#include "predicates/predicates.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

/* Enough for the paths sif builds itself ($HOME plus two components). Anything
 * longer is refused rather than truncated: a truncated path names a real but
 * different directory, and the run would silently write somewhere else. */
#define PATH_CAP 512

/* --- filesystem --- */

/*
 * Create @p path and every missing component above it, as `mkdir -p` does.
 *
 * The paths involved come from the environment or from the settings file, so
 * the whole chain may be missing on a fresh machine. Permissions are left to
 * the process umask, matching what the shell would have produced.
 *
 * Returns false, having logged, if the directory does not exist afterwards.
 */
static bool ensure_dir(const char* path) {
  if (!path || path[0] == '\0')
    return false;

  char tmp[PATH_CAP];
  int written = snprintf(tmp, sizeof(tmp), "%s", path);
  if (written < 0 || (size_t)written >= sizeof(tmp)) {
    SIF_LOG_WARNING("system", "path too long, not created: %s", path);
    return false;
  }

  size_t len = strlen(tmp);
  while (len > 1 && tmp[len - 1] == '/')
    tmp[--len] = '\0';

  /* Walk the separators, creating each ancestor in turn: mkdir() creates one
   * level only. EEXIST is the ordinary case here, not a failure -- the whole
   * point is that we do not know how much of the chain is already there. */
  for (char* p = tmp + 1; *p; p++) {
    if (*p != '/')
      continue;

    *p = '\0';
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
      SIF_LOG_WARNING(
        "system", "failed to create %s: %s", tmp, strerror(errno));
      return false;
    }
    *p = '/';
  }

  if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
    SIF_LOG_WARNING("system", "failed to create %s: %s", tmp, strerror(errno));
    return false;
  }

  return true;
}

/* --- lifetime --- */

/* The one piece of process-global state in the library. NULL between
 * sif_finalize() and the next sif_init(), which is how both accessors below
 * and the logger tell whether the library is up. */
static sif_system_state_t* system_state = NULL;

int sif_init(sif_config_t* config) {
  /* Idempotent rather than an error: the library is up, which is what the
   * caller asked for. The configuration of the first call stays in force. */
  if (system_state) {
    SIF_LOG_WARNING("system", "sif library already initialized");
    return SIF_OK;
  }

  /* calloc, not malloc: every field is then readable before the step that
   * owns it has run, which matters on the failure paths below. A library does
   * not get to end its caller's process -- least of all a Python
   * interpreter's -- so running out of memory here is a status like any
   * other. */
  system_state = calloc(1, sizeof(sif_system_state_t));
  if (!system_state) {
    SIF_LOG_ERROR("system", "failed to allocate sif system state (%zu bytes)",
      sizeof(sif_system_state_t));
    return SIF_ERR_ALLOC;
  }

  /* Resolve the logger first: everything below here logs, and until the state
   * exists sif__log_impl() has to guess a level. */
  system_state->logger.level = config ? config->log_level : SIF_LOG_LEVEL_INFO;

  sif_timer_start(&system_state->total_runtime_timer);

  /* --- threads --- */

  const uint32_t req_threads =
    (config && config->omp_config) ? config->omp_config->n_threads : 0;

#ifdef _OPENMP
  if (req_threads > 0) {
    omp_set_num_threads((int)req_threads);
  }
  /* Read back rather than trusting the request: OMP_THREAD_LIMIT, a cgroup or
   * the batch scheduler may cap us below it, and every work-splitting decision
   * in the library sizes its buffers from this number. */
  system_state->max_threads = (uint32_t)omp_get_max_threads();
#else
  system_state->max_threads = 1;
  if (req_threads > 1) {
    SIF_LOG_WARNING("system",
      "requested %u threads, but compiled without OpenMP", req_threads);
  }
#endif

  /* --- settings --- */

  /* Everything sif persists hangs off one root directory, and the settings
   * file inside it names the rest. HOME is absent on some batch nodes; /tmp
   * keeps the run working, at the price of a table that does not survive it. */
  const char* home = getenv("HOME");
  char base_dir[PATH_CAP];
  if (home) {
    snprintf(base_dir, sizeof(base_dir), "%s/.sif", home);
  } else {
    snprintf(base_dir, sizeof(base_dir), "/tmp/.sif");
    SIF_LOG_WARNING(
      "system", "HOME env variable not set, defaulting cache to %s", base_dir);
  }

  ensure_dir(base_dir);
  sif__settings_init(base_dir);

  /* The directories the settings point at are created now rather than on
   * first use, where a missing one would surface deep inside a run as an
   * unhelpful IO error from the cache or the wisdom loader. */
  const char* cache_dir = sif_setting_get("cache_directory", NULL);
  if (cache_dir)
    ensure_dir(cache_dir);

  /* --- FFTW --- */

  const bool skip_tuning =
    (config && config->fft_config) ? config->fft_config->skip_tuning : false;

  const char* wisdom_dir = sif_setting_get("fft_wisdom_dir", NULL);
  if (wisdom_dir)
    ensure_dir(wisdom_dir);

  /* Without FFTW there are no finders, no smoothing and no delta statistics,
   * so a library that came up anyway would only fail later, somewhere less
   * obvious. It fails here instead, and leaves nothing behind: the caller gets
   * a status and a library that is simply not initialized. */
  system_state->fft_mgr = sif__fft_manager_init(skip_tuning, wisdom_dir);
  if (!system_state->fft_mgr) {
    SIF_LOG_ERROR("system", "failed to initialize FFTW");
    sif__settings_finalize();
    free(system_state);
    system_state = NULL;
    return SIF_ERR_ALLOC;
  }

  /* FFTW's thread count is global and read at plan time, so setting it once
   * here fixes it for every plan the library makes afterwards. It must come
   * after the manager, which is what initializes FFTW's threading. */
  real_fftw_plan_with_nthreads(sif__system_max_threads());

  /* Shewchuk's exact predicates derive their error bounds from the running
   * machine's floating-point behaviour, once, before any predicate is
   * evaluated. Nothing evaluates one at the moment -- the tessellation that
   * did was removed -- but the initialization has to happen here rather than
   * at the first call, so it stays. */
  exactinit();

  SIF_LOG_INFO("system", "sif library initialized (threads: %u)",
    system_state->max_threads);

  SIF_LOG_FLUSH();
  return SIF_OK;
}

void sif_finalize(void) {
  if (!system_state) {
    SIF_LOG_WARNING("system", "sif library not currently initialized");
    return;
  }

  sif_timer_stop(&system_state->total_runtime_timer);
  double total_time = sif_timer_elapsed_ms(&system_state->total_runtime_timer);
  SIF_LOG_INFO("system", "sif library finalized. Total execution time: %.2f s",
    total_time / 1000);

  /* Flush before tearing anything down: the teardown below can abort the
   * process on a broken FFTW state, and a buffered log line would be lost. */
  SIF_LOG_FLUSH();

  /* FFTW writes its accumulated wisdom out from here. */
  sif__fft_manager_finalize(system_state->fft_mgr);

  /* Saves the table if anything changed, then releases it. */
  sif__settings_finalize();

  free(system_state);
  system_state = NULL;
}

/* --- accessors --- */

sif_system_state_t* sif__system_state(void) {
  if (!system_state)
    SIF_LOG_ERROR("system", "the library is not initialized; call sif_init()");

  return system_state;
}

sif_system_state_t* sif__system_state_safe(void) { return system_state; }

int sif__system_max_threads(void) {
  /* Answering 1 rather than aborting lets a caller size a per-thread buffer
   * before the library is up, which the logger and the tests both do. */
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

#undef PATH_CAP
