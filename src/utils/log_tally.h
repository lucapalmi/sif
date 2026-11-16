/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file log_tally.h
 * @brief A condition counted over radii, for one warning in place of one per
 * radius. Private to the library.
 *
 * A diagnostic that can fire at every radius of a table would otherwise warn
 * once per radius -- hundreds of lines saying one thing. The tally keeps how
 * many radii it hit, their range, and the worst value seen, and the caller
 * logs that once, after the loop:
 *
 * @code
 * sif_log_tally_t zero = {0};
 * for (uint32_t r = 0; r < n_radii; r++)
 *   if (sigma[r] <= 0)
 *     sif__log_tally_add(&zero, radii[r], 0.0);
 * if (zero.n)
 *   SIF_LOG_WARNING(TAG, "sigma: zero at %u of %u radii, R from %g to %g",
 *     zero.n, n_radii, zero.r_min, zero.r_max);
 * @endcode
 */

#ifndef SIF__UTILS_LOG_TALLY_H
#define SIF__UTILS_LOG_TALLY_H

#include <stdint.h>

/** @brief How often a condition held, over which radii, and how badly. */
typedef struct {
  uint32_t n;
  double r_min;
  double r_max;
  /** The largest value passed to sif__log_tally_add(). */
  double worst;
} sif_log_tally_t;

/** @brief Count one radius @p r where the condition held, with @p value. */
static inline void sif__log_tally_add(
  sif_log_tally_t* t, double r, double value) {
  if (t->n == 0) {
    t->r_min = t->r_max = r;
    t->worst = value;
  } else {
    if (r < t->r_min)
      t->r_min = r;
    if (r > t->r_max)
      t->r_max = r;
    if (value > t->worst)
      t->worst = value;
  }
  t->n++;
}

#endif /* SIF__UTILS_LOG_TALLY_H */
