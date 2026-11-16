/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file timer.h
 * @brief Monotonic wall-clock timer used for the library's own timings.
 */

#ifndef SIF_UTILS_TIMER_H
#define SIF_UTILS_TIMER_H

#include <time.h>

#include "sif/core/macros.h"

/**
 * @brief A start/stop pair of monotonic timestamps.
 *
 * Monotonic rather than calendar time, so a clock adjustment mid-run cannot
 * produce a negative interval.
 */
typedef struct {
  struct timespec start_time;
  struct timespec stop_time;
} sif_timer_t;

/**
 * @brief Start, or restart, the timer.
 * @param timer Timer to stamp.
 * @return SIF_OK, SIF_ERR_INVALID for a NULL timer, or SIF_ERR_UNSUPPORTED
 * if the system has no monotonic clock.
 *
 * @note Not SIF_NODISCARD, unlike the library's other status returns: a
 * timing is a diagnostic, and the failures above are a NULL that a timer on
 * the stack cannot be and a clock POSIX requires. The status is there for the
 * caller that does care, and the reason is logged either way.
 */
int sif_timer_start(sif_timer_t* timer);

/**
 * @brief Stop the timer, fixing the interval.
 * @param timer Timer previously passed to sif_timer_start().
 * @return As sif_timer_start().
 */
int sif_timer_stop(sif_timer_t* timer);

/**
 * @brief Elapsed time between the last start and the last stop.
 *
 * Returns `double` rather than sif_real deliberately: a duration is not a
 * physical field quantity, and milliseconds at float precision would quantise
 * visibly over a long run.
 *
 * @param timer Timer to read.
 * @return Elapsed time in milliseconds, or -1.0 if @p timer is NULL.
 *
 * @warning Only meaningful once sif_timer_stop() has been called. Reading a
 * timer that was started but never stopped returns whatever the uninitialised
 * stop timestamp held.
 */
double sif_timer_elapsed_ms(const sif_timer_t* timer);

#endif /* SIF_UTILS_TIMER_H */
