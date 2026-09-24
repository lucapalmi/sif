/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The parts of the HDF5 layer that do not touch HDF5, compiled into every
 * build so that hdf5.c and hdf5_off.c answer the same questions the same way:
 * a metadata key refused in one build is refused in the other.
 */

#include "io/hdf5_internal.h"

#include <string.h>

const char* sif__hdf5_group(const char* group) {
  if (!group)
    return NULL;
  while (*group == '/')
    group++;
  return *group ? group : NULL;
}

/*
 * The attributes sif reads back itself, group by group. Overwriting one is
 * how a file stops reading -- a catalogue whose n_voids no longer matches its
 * arrays -- so the metadata setters refuse them. On the root, every name
 * beginning "sif_" is kept for the library, present and future.
 */
typedef struct {
  const char* group;
  const char* const* keys;
} reserved_t;

static const char* const CATALOG_KEYS[] = {"n_voids", NULL};
static const char* const DENSITY_KEYS[] = {
  "n_voids", "n_bins", "ext", "differential", NULL};
static const char* const VELOCITY_KEYS[] = {"n_voids", "n_bins", "ext", NULL};
static const char* const VSF_KEYS[] = {
  "n_bins", "r_min", "r_max", "options", "binning", NULL};

static const reserved_t RESERVED[] = {
  {"catalog", CATALOG_KEYS},
  {"density_profiles", DENSITY_KEYS},
  {"velocity_profiles", VELOCITY_KEYS},
  {"size_function", VSF_KEYS},
};

int sif__hdf5_key_reserved(const char* group, const char* key) {
  if (!group)
    return strncmp(key, "sif_", 4) == 0;

  for (size_t g = 0; g < sizeof(RESERVED) / sizeof(RESERVED[0]); g++) {
    if (strcmp(group, RESERVED[g].group) != 0)
      continue;
    for (const char* const* k = RESERVED[g].keys; *k; k++)
      if (strcmp(*k, key) == 0)
        return 1;
  }
  return 0;
}
