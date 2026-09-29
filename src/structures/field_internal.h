/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* What the functions that read positions as lengths share. */

#ifndef SIF__STRUCTURES_FIELD_INTERNAL_H
#define SIF__STRUCTURES_FIELD_INTERNAL_H

#include "sif/structures/field.h"

/* SIF_OK for a field of Cartesian positions (or no field: the caller's own
 * checks say what is wrong with that); SIF_ERR_INVALID, logged under `tag`,
 * for one still holding sky coordinates, which binning, sorting, bounding or
 * moving would read as lengths. */
int sif__field_require_cartesian(const sif_field_t* field, const char* tag);

#endif /* SIF__STRUCTURES_FIELD_INTERNAL_H */
