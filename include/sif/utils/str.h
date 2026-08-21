/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file str.h
 * @brief String helpers for parsing ASCII catalogue and field files.
 *
 * These back the ASCII readers, which cannot use scanf: the column layout is
 * not known until runtime, and the files are large enough that the parse has
 * to run once over the buffer without allocating per field.
 */

#ifndef SIF_UTILS_STR_H
#define SIF_UTILS_STR_H

#include "sif/core/macros.h"

#include <stdint.h>

/** @brief What a column of an ASCII file holds. */
typedef enum {
  /** Present but not read. */
  SIF_COL_IGNORE = 0,
  SIF_COL_X,
  SIF_COL_Y,
  SIF_COL_Z,
  SIF_COL_VX,
  SIF_COL_VY,
  SIF_COL_VZ,
  /** Per-particle weight; see sif_field_t::weights. */
  SIF_COL_M
} sif_col_target_t;

/**
 * @brief Translate a column format string into an array of column targets.
 *
 * One character per column, case-insensitive:
 *
 * `x` `y` `z`
 *   position components
 *
 * `u` `v` `w`
 *   velocity components
 *
 * `m`
 *   per-particle weight (`m` for mass, its usual meaning)
 *
 * `*` `/`
 *   ignored column
 *
 * So `"xyz*m"` describes a file whose first three columns are the position,
 * whose fourth is skipped, and whose fifth is the weight.
 *
 * @param fmt Format string as above.
 * @param targets_out Written with one entry per decoded column. Must have room
 * for @p max_cols entries.
 * @param max_cols Capacity of @p targets_out; decoding stops there.
 * @return The number of columns decoded. This is a count, not a status code.
 *
 * @warning Characters outside the table are skipped silently rather than
 * rejected, so a typo in @p fmt yields a shorter layout instead of an error.
 * Compare the result against the column count you expect.
 */
int sif_str_decode_format(
  const char* fmt, sif_col_target_t* targets_out, int max_cols);

/**
 * @brief Read the next numeric field and advance the cursor past it.
 *
 * Skips leading whitespace, parses one number, and leaves @p cursor on the
 * character after the field's trailing delimiter, ready for the next call.
 *
 * @param cursor Address of the read pointer; advanced in place.
 * @param delimiter Field separator. Use `' '` for whitespace-separated files.
 * @param out_val Written with the parsed value.
 * @return 1 if a field was consumed, 0 at end of line or end of string.
 *
 * @warning The return value reports that a field was *consumed*, not that it
 * parsed as a number. A malformed field yields 0.0 in @p out_val, returns 1,
 * and skips to the next delimiter, so a text column inside a numeric file
 * reads as zeros rather than failing.
 */
int sif_str_extract_next_real(char** cursor, char delimiter, sif_real* out_val);

#endif /* SIF_UTILS_STR_H */
