#ifndef __SIF_STRINGY_H__
#define __SIF_STRINGY_H__

#include "sif/core/macros.h"

#include <stdint.h>

/* Enum to map arbitrary columns to specific targets */
typedef enum {
  COL_IGNORE = 0,
  COL_X,
  COL_Y,
  COL_Z,
  COL_VX,
  COL_VY,
  COL_VZ,
  COL_M
} col_target_t;

/*
 * @brief Translates a format string into an array of targets.
 *
 * @return The total number of columns expected per line.
 */
int str_decode_format(const char* fmt, col_target_t* targets_out, int max_cols);

/*
 * @brief Extracts the next real_t from a string and advances the cursor
 *
 * @return 1 on success, 0 if end of string/line reached.
 */
int str_extract_next_real(char** cursor, char delimiter, real_t* out_val);

#endif /* __SIF_STRINGY_H__ */
