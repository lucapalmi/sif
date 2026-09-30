/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file catalogue_internal.h
 * @brief A catalogue's metadata entries, and what the file formats share
 * about them. Private.
 */

#ifndef SIF__STRUCTURES_CATALOGUE_INTERNAL_H
#define SIF__STRUCTURES_CATALOGUE_INTERNAL_H

#include "sif/structures/catalogue.h"

#include <stdbool.h>
#include <stdint.h>

/** One named value; see the catalogue_meta group in catalogue.h. */
typedef struct sif_catalogue_meta_entry {
  /** Lower case, owned. */
  char* key;
  sif_catalogue_meta_kind_t kind;
  int64_t i;
  /** The value as a double, for a real and for an integer alike. */
  double d;
  /** Owned, for a string; NULL otherwise. */
  char* text;
} sif_catalogue_meta_entry_t;

/**
 * @brief Whether a key (in lower case) is one a file format uses for itself:
 * the readers skip such names rather than read them as metadata, and the
 * setters refuse them.
 */
bool sif__catalogue_meta_reserved(const char* key);

/**
 * @brief Whether the setters would accept @p key: an identifier of at most 64
 * characters, not reserved. For the readers, which skip a key the setters
 * would refuse -- with a warning of their own, where the setter would log an
 * error.
 */
bool sif__catalogue_meta_key_ok(const char* key);

/** @brief Copy every entry of @p from into @p to. */
int sif__catalogue_meta_copy(sif_catalogue_t* to, const sif_catalogue_t* from);

#endif /* SIF__STRUCTURES_CATALOGUE_INTERNAL_H */
