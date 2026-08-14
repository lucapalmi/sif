/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/utils/str.h"

#include <ctype.h>
#include <stdlib.h>

int sif_str_decode_format(
  const char* fmt, sif_col_target_t* targets_out, int max_cols) {
  if (!fmt || !targets_out || max_cols <= 0)
    return 0;

  int col_count = 0;

  /* Unrecognized characters fall through the switch and are skipped, which is
   * what lets a format be written with separators for readability. It is also
   * why a typo yields a short layout instead of an error, and why the caller
   * has to compare the count against what it expected. */
  for (int i = 0; fmt[i] != '\0' && col_count < max_cols; i++) {
    switch (fmt[i]) {
    case '*':
    case '/':
      targets_out[col_count++] = SIF_COL_IGNORE;
      break;
    case 'x':
    case 'X':
      targets_out[col_count++] = SIF_COL_X;
      break;
    case 'y':
    case 'Y':
      targets_out[col_count++] = SIF_COL_Y;
      break;
    case 'z':
    case 'Z':
      targets_out[col_count++] = SIF_COL_Z;
      break;
    case 'u':
    case 'U':
      targets_out[col_count++] = SIF_COL_VX;
      break;
    case 'v':
    case 'V':
      targets_out[col_count++] = SIF_COL_VY;
      break;
    case 'w':
    case 'W':
      targets_out[col_count++] = SIF_COL_VZ;
      break;
    case 'm':
    case 'M':
      targets_out[col_count++] = SIF_COL_M;
      break;
    }
  }

  return col_count;
}

int sif_str_extract_next_real(
  char** cursor, char delimiter, sif_real* out_val) {
  if (!cursor || !*cursor || !out_val)
    return 0;

  if (**cursor == '\0' || **cursor == '\n')
    return 0;

  /* Leading whitespace is skipped, except when the delimiter is itself a
   * whitespace character other than a space -- a tab-separated file has
   * meaningful tabs, and eating them would merge two empty columns into one.
   * A newline reached here ends the line: the row had fewer columns than the
   * format asked for, which the caller needs to be able to tell apart from a
   * column that merely failed to parse. */
  while (isspace((unsigned char)**cursor) &&
         (**cursor != delimiter || delimiter == ' ')) {
    if (**cursor == '\n')
      return 0;
    (*cursor)++;
  }

  char* endptr;

  /* strtod, not strtof, even in a single-precision build: parsing at full
   * precision and narrowing once is correct, while parsing at float precision
   * would round twice. It reads the decimal point according to LC_NUMERIC,
   * which sif never changes and CPython deliberately leaves at "C". */
  *out_val = (sif_real)strtod(*cursor, &endptr);

  if (endptr == *cursor) {
    /* Nothing numeric here. The field is still consumed -- a text column in a
     * numeric file reads as 0.0 rather than derailing the whole row -- so the
     * cursor has to be walked to the next separator by hand. */
    while (**cursor != '\0' && **cursor != '\n' && **cursor != delimiter) {
      (*cursor)++;
    }
  } else {
    *cursor = endptr;
  }

  if (**cursor == delimiter) {
    (*cursor)++;
  }

  return 1;
}
