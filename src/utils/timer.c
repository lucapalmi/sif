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

  /* CLOCK_MONOTONIC, not CLOCK_REALTIME: a long run can outlive an NTP step
   * or a daylight-saving change, either of which would make a wall-clock
   * interval come out negative. */
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

  /* The two fields are differenced separately and converted afterwards.
   * Folding each timestamp into a single double first would lose nanoseconds
   * outright: seconds since boot needs more mantissa than a double has left
   * over for a 1e-9 resolution. The nanosecond difference may be negative,
   * which is correct -- it borrows from the whole second above it. */
  const double seconds =
    (double)(timer->stop_time.tv_sec - timer->start_time.tv_sec);
  const double nanoseconds =
    (double)(timer->stop_time.tv_nsec - timer->start_time.tv_nsec);

  return seconds * 1e3 + nanoseconds * 1e-6;
}
