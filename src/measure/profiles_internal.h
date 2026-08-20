/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file profiles_internal.h
 * @brief Profile-set allocation, for the library's own use.
 *
 * A profile set means nothing until an estimator has filled it, which is why
 * profiles.h publishes only the frees. The reader in io/ is the one other
 * place that can fill one honestly -- it has a file that an estimator wrote --
 * so the allocators live here rather than in the public header.
 */

#ifndef SIF_MEASURE_PROFILES_INTERNAL_H
#define SIF_MEASURE_PROFILES_INTERNAL_H

#include <stdbool.h>

#include "sif/measure/profiles.h"

/**
 * @brief Allocate a density set, zeroed, with room for every row.
 * @return The set, or NULL on allocation failure.
 */
SIF_NODISCARD sif_density_profiles_t* sif__density_profiles_alloc(
  uint64_t n_voids, uint32_t n_bins, sif_real ext, bool differential);

/**
 * @brief Allocate a velocity set, zeroed, with room for every row.
 * @return The set, or NULL on allocation failure.
 */
SIF_NODISCARD sif_velocity_profiles_t* sif__velocity_profiles_alloc(
  uint64_t n_voids, uint32_t n_bins, sif_real ext);

#endif /* SIF_MEASURE_PROFILES_INTERNAL_H */
