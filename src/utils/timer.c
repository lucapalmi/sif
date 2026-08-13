/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/utils/timer.h"
#include "sif/utils/logger.h"

void sif_timer_start(sif_timer_t* timer) {
  if (!timer) {
    SIF_LOG_WARNING("timer", "start called on NULL timer");
    return;
  }
  clock_gettime(CLOCK_MONOTONIC, &timer->start_time);
}

void sif_timer_stop(sif_timer_t* timer) {
  if (!timer) {
    SIF_LOG_WARNING("timer", "stop called on NULL timer");
    return;
  }
  clock_gettime(CLOCK_MONOTONIC, &timer->stop_time);
}

double sif_timer_elapsed_ms(const sif_timer_t* timer) {
  if (!timer) {
    SIF_LOG_WARNING("timer", "elapsed_ms called on NULL timer");
    return -1.0;
  }

  double elapsed =
    (timer->stop_time.tv_sec - timer->start_time.tv_sec) * 1e3 +
    (timer->stop_time.tv_nsec - timer->start_time.tv_nsec) * 1e-6;

  return elapsed;
}
