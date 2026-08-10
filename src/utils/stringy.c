#include "sif/utils/stringy.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include "sif/utils/stringy.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

int sif_str_decode_format(
  const char* fmt, sif_col_target_t* targets_out, int max_cols) {
  int col_count = 0;
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

int sif_str_extract_next_real(char** cursor, char delimiter, real_t* out_val) {
  if (!cursor || !*cursor || **cursor == '\0' || **cursor == '\n')
    return 0;

  while (isspace((unsigned char)**cursor) &&
         (**cursor != delimiter || delimiter == ' ')) {
    if (**cursor == '\n')
      return 0;
    (*cursor)++;
  }

  char* endptr;
  *out_val = (real_t)strtod(*cursor, &endptr);

  if (endptr == *cursor) {
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
