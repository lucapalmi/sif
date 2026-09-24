/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file catalog_io.h
 * @brief Reading and writing void catalogues, in plain text.
 *
 * Text rather than binary: a catalogue is small next to the field it came
 * from, and it is the thing a user actually looks at, plots and hands to other
 * tools. The layout is one header line holding the void count, then one line
 * per void with `cx cy cz radius` separated by spaces -- and, for a catalogue
 * that carries a footprint (see sif_finder_exodus_survey()), two more columns,
 * `footprint footprint_shell`. Every row of a file has the same columns.
 */

#ifndef SIF_IO_CATALOG_IO_H
#define SIF_IO_CATALOG_IO_H

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"

/**
 * @brief Write a catalogue to a text file.
 *
 * @param catalog Catalogue to write.
 * @param filepath Path to the output file, truncated if it exists.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL argument, or SIF_ERR_IO if the
 * file could not be written.
 *
 * @note Values are written with #SIF_PRI_REAL, which carries enough
 * significant digits to recover the stored sif_real exactly, so a catalogue
 * survives a write/read round trip unchanged.
 */
int sif_catalog_write_ascii(const sif_catalog_t* catalog, const char* filepath);

/**
 * @brief Read a catalogue written by sif_catalog_write_ascii().
 *
 * The leading count is used to size the allocation up front, so the file must
 * carry it. The first row says whether the footprint columns are there, and
 * the catalogue comes back with them exactly when they are; a file written
 * before they existed reads as it always did. An empty catalogue has no first
 * row and so comes back without them.
 *
 * @param filepath Path to the input file.
 * @return The catalogue, owned by the caller and released with
 * sif_catalog_free(). NULL on failure.
 */
SIF_NODISCARD sif_catalog_t* sif_catalog_read_ascii(const char* filepath);

#endif /* SIF_IO_CATALOG_IO_H */
