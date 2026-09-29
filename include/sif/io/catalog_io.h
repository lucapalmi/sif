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
 * tools. The layout is a two-line header behind `#`, which numpy.loadtxt and
 * most other readers skip on their own, then one line per void:
 *
 * @code
 * #n=1024
 * #cx cy cz r footprint footprint_shell
 * 101.25 250.5 33.125 12.5 1 0.84
 * ...
 * @endcode
 *
 * The first line is the void count; the second names the columns. A
 * catalogue on the sky (sif_catalog_to_sky()) names them `ra dec z r`. The
 * footprint columns (see sif_finder_exodus_survey()) are there only when the
 * catalogue carries a footprint.
 */

#ifndef SIF_IO_CATALOG_IO_H
#define SIF_IO_CATALOG_IO_H

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"

/**
 * @brief Write a catalogue to a text file.
 *
 * @param filepath Path to the output file, truncated if it exists.
 * @param catalog Catalogue to write.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL argument, or SIF_ERR_IO if the
 * file could not be written.
 *
 * @note Values are written with #SIF_PRI_REAL, which carries enough
 * significant digits to recover the stored sif_real exactly, so a catalogue
 * survives a write/read round trip unchanged.
 */
int sif_catalog_write_ascii(const char* filepath, const sif_catalog_t* catalog);

/**
 * @brief Read a catalogue written by sif_catalog_write_ascii().
 *
 * Lines starting with `#` are comments, wherever they are. Among those before
 * the first row, `n=N` gives the count, which sizes the catalogue up front;
 * without one the rows are counted first and then read. A comment naming the
 * columns places them: `cx cy cz` (or `x y z`) for Cartesian centres or `ra
 * dec z` for sky ones, `r` (or `radius`), and optionally `footprint
 * footprint_shell`, in any order -- columns with other names are skipped.
 * Without names, rows are `cx cy cz r`, and six columns add the footprint.
 *
 * Files written before the header -- a first line holding the count alone --
 * read as they always did. Rows have to be as many as the count says, no
 * more and no fewer.
 *
 * @param filepath Path to the input file.
 * @return The catalogue, owned by the caller and released with
 * sif_catalog_free(). NULL on failure.
 */
SIF_NODISCARD sif_catalog_t* sif_catalog_read_ascii(const char* filepath);

#endif /* SIF_IO_CATALOG_IO_H */
