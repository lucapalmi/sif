/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/core/system.h"

#include "sif/core/settings.h"
#include "sif/utils/logger.h"
#include "sif/utils/timer.h"

#include "core/settings_internal.h"
#include "core/system_internal.h"
#include "io/fits_internal.h"
#include "io/hdf5_internal.h"
#include "utils/logger_internal.h"

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

/* --- lifetime --- */

/* The one piece of process-global state in the library. NULL between
 * sif_system_finalise() and the next sif_system_init(), which is how both
 * accessors below and the logger tell whether it is up. */
static sif_system_state_t* system_state = NULL;

int sif_system_init(sif_config_t* config) {
  /* Idempotent rather than an error: the library is up, which is what the
   * caller asked for. The configuration of the first call stays in force. */
  if (system_state) {
    SIF_LOG_WARNING("system", "the system state is already initialised");
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

  /* omp_get_num_procs() counts the processors this process may run on -- the
   * affinity mask it started with, not the machine. Fewer of them than
   * threads means the threads share cores, and nothing else would say so:
   * the log shows the full thread count while the run crawls. The usual cause
   * is a parent that was pinned before it exec'd us -- a launcher that loaded
   * an OpenMP runtime with OMP_PROC_BIND set binds itself to one core, and
   * the mask is inherited -- or a cpu-bind or cgroup narrower than intended.
   * Running this way is not an error, so it is a warning. */
  const int n_procs = omp_get_num_procs();
  if (n_procs > 0 && system_state->max_threads > (uint32_t)n_procs) {
    SIF_LOG_WARNING("system",
      "%u threads but %d processor%s available to this process: the threads "
      "will share them. Check the CPU affinity this process inherited "
      "(taskset -p, or Cpus_allowed_list in /proc/self/status)",
      system_state->max_threads, n_procs, n_procs == 1 ? "" : "s");
  }
#else
  system_state->max_threads = 1;
  if (req_threads > 1) {
    SIF_LOG_WARNING("system",
      "requested %u threads, but compiled without OpenMP", req_threads);
  }
#endif

  /* --- settings --- */

  /* Everything sif persists hangs off one root directory, and the settings
   * file inside it names the rest: SIF_HOME, or $HOME/.sif. HOME is absent on
   * some batch nodes; /tmp keeps the run working, at the price of a table that
   * does not survive it. */
  const char* sif_home = getenv("SIF_HOME");
  const char* home = getenv("HOME");
  char base_dir[PATH_CAP];
  int written;
  if (sif_home && sif_home[0]) {
    written = snprintf(base_dir, sizeof(base_dir), "%s", sif_home);
  } else if (home) {
    written = snprintf(base_dir, sizeof(base_dir), "%s/.sif", home);
  } else {
    written = snprintf(base_dir, sizeof(base_dir), "/tmp/.sif");
    SIF_LOG_WARNING("system", "HOME: not set; settings, cache and wisdom go "
      "to %s, which may not outlive the job", base_dir);
  }
  if (written < 0 || (size_t)written >= sizeof(base_dir)) {
    SIF_LOG_ERROR("system", "SIF_HOME: longer than %d characters",
      PATH_CAP - 1);
    free(system_state);
    system_state = NULL;
    return SIF_ERR_INVALID;
  }

  (void)sif__ensure_dir(base_dir);
  sif__settings_init(base_dir);

  /* --- FFTW --- */

  const bool skip_tuning =
    (config && config->fft_config) ? config->fft_config->skip_tuning : false;

  /* Without its directory the run simply plans from scratch, so a bad path
   * costs the saved plans, not the run. The cache directory is left until a
   * grid is actually cached: it is off by default. */
  const char* wisdom_dir = NULL;
  if (sif__setting_directory("fft_wisdom_dir", &wisdom_dir) != SIF_OK)
    wisdom_dir = NULL;

  /* Without FFTW there are no finders, no smoothing and no delta statistics,
   * so a library that came up anyway would only fail later, somewhere less
   * obvious. It fails here instead, and leaves nothing behind: the caller gets
   * a status and a library that is simply not initialised. */
  system_state->fft_mgr = sif__fft_manager_init(skip_tuning, wisdom_dir);
  if (!system_state->fft_mgr) {
    SIF_LOG_ERROR("system", "failed to initialise FFTW");
    sif__settings_finalise();
    free(system_state);
    system_state = NULL;
    return SIF_ERR_ALLOC;
  }

  /* Said at start-up so that a build that came out without HDF5 or cfitsio
   * -- an AUTO configure on a machine where it was not loaded -- is visible
   * in the log of the first run, not discovered at the first call. */
  char hdf5[64];
  sif__hdf5_describe(hdf5, sizeof(hdf5));
  char fits[64];
  sif__fits_describe(fits, sizeof(fits));

  SIF_LOG_INFO("system", "sif system state initialised (threads: %u, %s, %s)",
    system_state->max_threads, hdf5, fits);

  SIF_LOG_FLUSH();
  return SIF_OK;
}

void sif_system_finalise(void) {
  if (!system_state) {
    SIF_LOG_WARNING("system", "the system state is not initialised");
    return;
  }

  sif_timer_stop(&system_state->total_runtime_timer);
  double total_time = sif_timer_elapsed_ms(&system_state->total_runtime_timer);
  SIF_LOG_INFO("system",
    "sif system state finalised. Total execution time: %.2f s",
    total_time / 1000);

  /* Flush before tearing anything down: the teardown below can abort the
   * process on a broken FFTW state, and a buffered log line would be lost. */
  SIF_LOG_FLUSH();

  /* FFTW writes its accumulated wisdom out from here. */
  sif__fft_manager_finalise(system_state->fft_mgr);

  /* Saves the table if anything changed, then releases it. */
  sif__settings_finalise();

  free(system_state);
  system_state = NULL;
}

/* --- deprecated names --- */

/* Kept for one release so existing callers keep building. The warning is
 * logged as well as declared: a caller that builds with deprecation warnings
 * off, or through the Python bindings, still hears about it. */
int sif_init(sif_config_t* config) {
  SIF_LOG_WARNING("system",
    "sif_init(): deprecated, removed in the next release: call "
    "sif_system_init()");
  return sif_system_init(config);
}

void sif_finalise(void) {
  SIF_LOG_WARNING("system",
    "sif_finalise(): deprecated, removed in the next release: call "
    "sif_system_finalise()");
  sif_system_finalise();
}

/* --- accessors --- */

sif_system_state_t* sif__system_state(void) {
  /* Logged and given its status here, the innermost place that knows, so
   * every caller only has to pass SIF_ERR_UNINITIALISED up -- and a function
   * that returns a pointer still reports it through sif_error_status(). */
  if (!system_state) {
    SIF_LOG_ERROR("system",
      "system state not initialised: call sif_system_init() first "
      "(pysif.system_init() in Python)");
    sif__error_status_set(SIF_ERR_UNINITIALISED);
  }

  return system_state;
}

sif_system_state_t* sif__system_state_safe(void) { return system_state; }

int sif__system_max_threads(void) {
  if (system_state)
    return (int)system_state->max_threads;

  /* Most of the library runs without the system state, and sizes its
   * per-thread scratch from this. Answering 1 here used to run all of it on
   * one thread without a word; the runtime's own ceiling is what a caller that
   * never asked for anything else expects. */
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
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
