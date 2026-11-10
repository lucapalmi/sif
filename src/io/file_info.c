/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/file_info.h"

#include "sif/finder/exodus_finder.h"
#include "sif/io/field_io.h"
#include "sif/io/gadget_io.h"
#include "sif/io/grid_io.h"
#include "sif/structures/grid.h"
#include "sif/utils/logger.h"

#include "io/file_check_internal.h"
#include "io/file_info_internal.h"
#include "io/internal.h"
#include "sif/utils/crc32.h"
#include "structures/field_internal.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define TAG "io"

/* ------------------------------------------------------------------------ */
/* formats                                                                   */
/* ------------------------------------------------------------------------ */

static const char* const FORMAT_NAMES[] = {"unknown", "ascii", "fits", "hdf5",
  "gadget", "xfield", "xgrid", "checkpoint"};

/* As a person writes them, for the description. */
static const char* const FORMAT_TITLES[] = {"unknown", "ASCII", "FITS", "HDF5",
  "GADGET", ".xfield", ".xgrid", "exodus checkpoint"};

#define N_FORMATS (sizeof(FORMAT_NAMES) / sizeof(FORMAT_NAMES[0]))

const char* sif_file_format_name(sif_file_format_t format) {
  return (unsigned)format < N_FORMATS ? FORMAT_NAMES[format] : "unknown";
}

int sif_file_format_parse(const char* name, sif_file_format_t* out) {
  if (!name || !out)
    return SIF_ERR_INVALID;
  for (unsigned k = 1; k < N_FORMATS; k++) {
    if (strcmp(name, FORMAT_NAMES[k]) == 0) {
      *out = (sif_file_format_t)k;
      return SIF_OK;
    }
  }
  return SIF_ERR_INVALID;
}

/* Whether a file starts as text does: no NUL, and few control characters
 * besides the blanks, in its first block. */
static bool looks_like_text(const unsigned char* head, size_t n) {
  size_t odd = 0;
  for (size_t i = 0; i < n; i++) {
    if (head[i] == 0)
      return false;
    if (head[i] < 32 && head[i] != '\n' && head[i] != '\r' && head[i] != '\t')
      odd++;
  }
  return odd * 100 <= n;
}

sif_file_format_t sif_file_format_detect(const char* path) {
  if (!path)
    return SIF_FILE_FORMAT_UNKNOWN;

  struct stat st;
  if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
    return sif_gadget_is_snapshot(path) ? SIF_FILE_FORMAT_GADGET
                                        : SIF_FILE_FORMAT_ASCII;

  unsigned char head[4096];
  size_t n = 0;
  FILE* f = fopen(path, "rb");
  if (f) {
    n = fread(head, 1, sizeof(head), f);
    fclose(f);
  }
  /* Whose reading says what is wrong. */
  if (!f || n == 0)
    return SIF_FILE_FORMAT_ASCII;

  static const unsigned char h5sig[8] = {
    0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
  if (n >= 6 && memcmp(head, "SIMPLE", 6) == 0)
    return SIF_FILE_FORMAT_FITS;
  if (n >= 2 && head[0] == 0x1f && head[1] == 0x8b && strstr(path, ".fits"))
    return SIF_FILE_FORMAT_FITS;
  if (n >= 4 && memcmp(head, SIF_XFIELD_MAGIC, 4) == 0)
    return SIF_FILE_FORMAT_XFIELD;
  if (n >= 4 && memcmp(head, SIF_XGRID_MAGIC, 4) == 0)
    return SIF_FILE_FORMAT_XGRID;
  if (n >= 4 && memcmp(head, SIF_FINDER_CHECKPOINT_MAGIC, 4) == 0)
    return SIF_FILE_FORMAT_CHECKPOINT;
  if (sif_gadget_is_snapshot(path))
    return SIF_FILE_FORMAT_GADGET;
  if (n >= 8 && memcmp(head, h5sig, 8) == 0)
    return SIF_FILE_FORMAT_HDF5;
  if (looks_like_text(head, n))
    return SIF_FILE_FORMAT_ASCII;
  return SIF_FILE_FORMAT_UNKNOWN;
}

/* ------------------------------------------------------------------------ */
/* the description                                                           */
/* ------------------------------------------------------------------------ */

/* The description and what building it needs, which the caller never sees:
 * the public struct is the first member, so the two share an address. */
typedef struct {
  sif_file_info_t info;
  bool failed;
  /* ASCII: the layout the description found, which the check reads by. */
  char ascii_delimiter;
  bool ascii_header;
} info_box_t;

static void fail(sif_file_info_t* info) { ((info_box_t*)info)->failed = true; }

static bool failed(const sif_file_info_t* info) {
  return ((const info_box_t*)info)->failed;
}

static const char* copy_text(sif_file_info_t* info, const char* s) {
  char* d = strdup(s ? s : "");
  if (!d)
    fail(info);
  return d;
}

/* Room for one more of @p size bytes in @p *items, holding @p n. The lists
 * are short -- tens of columns, at most a few hundred keywords -- so they
 * grow by one. */
static bool grow(sif_file_info_t* info, void** items, uint32_t n, size_t size) {
  void* p = realloc(*items, (n + 1) * size);
  if (!p) {
    fail(info);
    return false;
  }
  *items = p;
  memset((char*)p + n * size, 0, size);
  return true;
}

static void meta_free(sif_file_meta_t* meta, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    free((void*)meta[i].key);
    free((void*)meta[i].s);
  }
  free(meta);
}

void sif_file_info_free(sif_file_info_t* info) {
  if (!info)
    return;
  free((void*)info->path);
  free((void*)info->detail);
  meta_free(info->meta, info->n_meta);
  for (uint32_t t = 0; t < info->n_tables; t++) {
    sif_file_table_t* tb = &info->tables[t];
    free((void*)tb->name);
    for (uint32_t c = 0; c < tb->n_columns; c++) {
      free((void*)tb->columns[c].name);
      free((void*)tb->columns[c].type);
      free((void*)tb->columns[c].unit);
    }
    free(tb->columns);
    meta_free(tb->meta, tb->n_meta);
  }
  free(info->tables);
  for (uint32_t a = 0; a < info->n_arrays; a++) {
    free((void*)info->arrays[a].name);
    free((void*)info->arrays[a].type);
    meta_free(info->arrays[a].meta, info->arrays[a].n_meta);
  }
  free(info->arrays);
  free(info);
}

void sif__file_info_detail(sif_file_info_t* info, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  free((void*)info->detail);
  info->detail = copy_text(info, buf);
}

uint32_t sif__file_info_table(
  sif_file_info_t* info, const char* name, uint64_t n_rows) {
  if (!grow(
        info, (void**)&info->tables, info->n_tables, sizeof(sif_file_table_t)))
    return UINT32_MAX;
  sif_file_table_t* tb = &info->tables[info->n_tables];
  tb->name = copy_text(info, name);
  tb->n_rows = n_rows;
  return info->n_tables++;
}

void sif__file_info_table_rename(
  sif_file_info_t* info, uint32_t table, const char* name) {
  if (table >= info->n_tables)
    return;
  free((void*)info->tables[table].name);
  info->tables[table].name = copy_text(info, name);
}

void sif__file_info_column(sif_file_info_t* info, uint32_t table,
  const char* name, const char* type, uint64_t width, const char* unit) {
  if (table >= info->n_tables)
    return;
  sif_file_table_t* tb = &info->tables[table];
  if (!grow(
        info, (void**)&tb->columns, tb->n_columns, sizeof(sif_file_column_t)))
    return;
  sif_file_column_t* c = &tb->columns[tb->n_columns++];
  c->name = copy_text(info, name);
  c->type = copy_text(info, type);
  c->width = width ? width : 1;
  c->unit = unit && unit[0] ? copy_text(info, unit) : NULL;
}

uint32_t sif__file_info_array(sif_file_info_t* info, const char* name,
  const char* type, uint32_t rank, const uint64_t* dims) {
  if (!grow(
        info, (void**)&info->arrays, info->n_arrays, sizeof(sif_file_array_t)))
    return UINT32_MAX;
  sif_file_array_t* a = &info->arrays[info->n_arrays];
  a->name = copy_text(info, name);
  a->type = copy_text(info, type);
  a->rank = rank < SIF_FILE_ARRAY_MAX_RANK ? rank : SIF_FILE_ARRAY_MAX_RANK;
  for (uint32_t k = 0; k < a->rank; k++)
    a->dims[k] = dims[k];
  return info->n_arrays++;
}

/* The next metadata slot of the owner, or NULL. */
static sif_file_meta_t* meta_slot(sif_file_info_t* info,
  sif_file_info_owner_t owner, uint32_t index, const char* key) {
  uint32_t* n;
  sif_file_meta_t** items;
  if (owner == SIF__FILE_INFO_TABLE && index < info->n_tables) {
    n = &info->tables[index].n_meta;
    items = &info->tables[index].meta;
  } else if (owner == SIF__FILE_INFO_ARRAY && index < info->n_arrays) {
    n = &info->arrays[index].n_meta;
    items = &info->arrays[index].meta;
  } else if (owner == SIF__FILE_INFO_FILE) {
    n = &info->n_meta;
    items = &info->meta;
  } else {
    return NULL;
  }
  if (!grow(info, (void**)items, *n, sizeof(sif_file_meta_t)))
    return NULL;
  sif_file_meta_t* m = &(*items)[(*n)++];
  m->key = copy_text(info, key);
  return m;
}

void sif__file_info_meta_int(sif_file_info_t* info, sif_file_info_owner_t owner,
  uint32_t index, const char* key, int64_t value) {
  sif_file_meta_t* m = meta_slot(info, owner, index, key);
  if (m) {
    m->kind = SIF_FILE_META_INT;
    m->i = value;
  }
}

void sif__file_info_meta_real(sif_file_info_t* info,
  sif_file_info_owner_t owner, uint32_t index, const char* key, double value) {
  sif_file_meta_t* m = meta_slot(info, owner, index, key);
  if (m) {
    m->kind = SIF_FILE_META_REAL;
    m->r = value;
  }
}

void sif__file_info_meta_string(sif_file_info_t* info,
  sif_file_info_owner_t owner, uint32_t index, const char* key,
  const char* value) {
  sif_file_meta_t* m = meta_slot(info, owner, index, key);
  if (m) {
    m->kind = SIF_FILE_META_STRING;
    m->s = copy_text(info, value);
  }
}

void sif__file_info_meta_parse(sif_file_info_t* info,
  sif_file_info_owner_t owner, uint32_t index, const char* key,
  const char* text) {
  while (*text == ' ' || *text == '\t')
    text++;
  size_t len = strlen(text);
  while (len > 0 && isspace((unsigned char)text[len - 1]))
    len--;

  char buf[512];
  if (len >= sizeof(buf))
    len = sizeof(buf) - 1;
  memcpy(buf, text, len);
  buf[len] = '\0';

  /* Quoted: a string, whatever it looks like. FITS doubles a quote inside
   * one; a sif ASCII header does not quote them at all. */
  if (len >= 2 && (buf[0] == '\'' || buf[0] == '"') && buf[len - 1] == buf[0]) {
    buf[len - 1] = '\0';
    char* s = buf + 1;
    size_t end = strlen(s);
    while (end > 0 && s[end - 1] == ' ')
      s[--end] = '\0';
    sif__file_info_meta_string(info, owner, index, key, s);
    return;
  }

  if (len > 0) {
    char* stop;
    errno = 0;
    const long long i = strtoll(buf, &stop, 10);
    if (*stop == '\0' && errno == 0) {
      sif__file_info_meta_int(info, owner, index, key, (int64_t)i);
      return;
    }
    /* FORTRAN's exponent, which FITS allows: 1.5D3. */
    for (char* c = buf; *c; c++)
      if (*c == 'D' || *c == 'd')
        *c = 'e';
    const double r = strtod(buf, &stop);
    if (*stop == '\0') {
      sif__file_info_meta_real(info, owner, index, key, r);
      return;
    }
    memcpy(buf, text, len);
    buf[len] = '\0';
  }
  sif__file_info_meta_string(info, owner, index, key, buf);
}

/* ------------------------------------------------------------------------ */
/* ASCII                                                                     */
/* ------------------------------------------------------------------------ */

/* Columns told apart and named in a line, at most. */
#define ASCII_MAX_COLS 1024

static bool is_blank(char c) { return c == ' ' || c == '\t'; }

static bool is_line_end(char c) { return c == '\0' || c == '\n' || c == '\r'; }

/* Splits @p line in place into fields: on runs of blanks for a delimiter of
 * ' ', on the delimiter otherwise, each field trimmed. */
static int split(char* line, char delimiter, char** fields, int max) {
  int n = 0;
  char* p = line;
  for (;;) {
    while (is_blank(*p) && (delimiter == ' ' || *p != delimiter))
      p++;
    if (is_line_end(*p))
      break;
    char* start = p;
    while (
      !is_line_end(*p) && (delimiter == ' ' ? !is_blank(*p) : *p != delimiter))
      p++;
    char* end = p;
    while (end > start && is_blank(end[-1]))
      end--;
    const bool more = !is_line_end(*p);
    *end = '\0';
    if (n < max)
      fields[n] = start;
    n++;
    if (!more)
      break;
    p++;
    if (delimiter != ' ' && is_line_end(*p)) {
      /* A trailing delimiter: one more, empty, field. */
      if (n < max)
        fields[n] = p;
      n++;
      *p = '\0';
      break;
    }
  }
  return n;
}

static bool is_number(const char* s) {
  if (!*s)
    return false;
  char* stop;
  strtod(s, &stop);
  return *stop == '\0';
}

/* The text of a comment line after its '#' or ';', or NULL for a line that
 * is not a comment. */
static char* comment_text(char* line) {
  while (is_blank(*line))
    line++;
  if (*line != '#' && *line != ';')
    return NULL;
  line++;
  while (is_blank(*line))
    line++;
  return line;
}

/* For a comment `key=value`: the key, cut at the '=', and the value. */
static bool key_value(char* text, char** key, char** value) {
  char* p = text;
  if (!isalpha((unsigned char)*p) && *p != '_')
    return false;
  while (isalnum((unsigned char)*p) || *p == '_' || *p == '.' || *p == '-')
    p++;
  char* end = p;
  while (is_blank(*p))
    p++;
  if (*p != '=')
    return false;
  *end = '\0';
  *key = text;
  *value = p + 1;
  char* nl = strpbrk(*value, "\r\n");
  if (nl)
    *nl = '\0';
  return true;
}

/* Data lines from here to the end of the file: a line counts when its first
 * character past any blanks is not a comment's, the same rule the reader
 * applies. Scanned in blocks, not parsed. */
static int count_data_lines(FILE* f, const char* path, uint64_t* rows) {
  char buf[1 << 16];
  bool at_start = true;
  size_t got;
  while ((got = fread(buf, 1, sizeof(buf), f)) > 0) {
    for (size_t i = 0; i < got; i++) {
      const char c = buf[i];
      if (c == '\n') {
        at_start = true;
      } else if (at_start && !is_blank(c)) {
        if (c != '#' && c != ';' && c != '\r')
          (*rows)++;
        at_start = false;
      }
    }
  }
  if (ferror(f))
    return sif__io_os_error(TAG, path, "read", errno);
  return SIF_OK;
}

static int describe_ascii(sif_file_info_t* info) {
  const char* path = info->path;
  FILE* f = fopen(path, "r");
  if (!f)
    return sif__io_os_error(TAG, path, NULL, errno);

  char* fields[ASCII_MAX_COLS];
  char* line = NULL;
  size_t cap = 0;
  char* names_line = NULL;    /* a header line: names, no numbers */
  char* comment_names = NULL; /* the last comment that is not key=value */
  char* data = NULL;          /* the first row */
  char delimiter = ' ';
  bool sniffed = false;
  int status = SIF_OK;

  while (getline(&line, &cap, f) >= 0) {
    char* text = comment_text(line);
    if (text) {
      char *key, *value;
      if (key_value(text, &key, &value)) {
        sif__file_info_meta_parse(info, SIF__FILE_INFO_FILE, 0, key, value);
      } else if (*text && !is_line_end(*text)) {
        free(comment_names);
        comment_names = strdup(text);
      }
      continue;
    }
    const char* p = line;
    while (is_blank(*p))
      p++;
    if (is_line_end(*p))
      continue;

    if (!sniffed) {
      delimiter = strchr(line, ',') ? ',' : strchr(line, ';') ? ';' : ' ';
      sniffed = true;
    }
    if (!names_line && !data) {
      /* A first line without a single number is a header. */
      char* copy = strdup(line);
      if (!copy) {
        fail(info);
        break;
      }
      const int n = split(copy, delimiter, fields, ASCII_MAX_COLS);
      bool numeric = false;
      for (int k = 0; k < n && k < ASCII_MAX_COLS && !numeric; k++)
        numeric = is_number(fields[k]);
      free(copy);
      if (!numeric) {
        names_line = strdup(line);
        continue;
      }
    }
    data = strdup(line);
    break;
  }
  free(line);
  if (!data && ferror(f))
    status = sif__io_os_error(TAG, path, "read", errno);

  uint64_t rows = data ? 1 : 0;
  if (status == SIF_OK && data)
    status = count_data_lines(f, path, &rows);
  fclose(f);

  if (status == SIF_OK) {
    ((info_box_t*)info)->ascii_delimiter = delimiter;
    ((info_box_t*)info)->ascii_header = names_line != NULL;
    sif__file_info_detail(info, "%s%s",
      delimiter == ','   ? "comma-separated"
      : delimiter == ';' ? "semicolon-separated"
                         : "blank-separated",
      names_line ? ", 1 header line" : "");

    const uint32_t t = sif__file_info_table(info, "", rows);

    /* The types from the first row; the names from the header line, or the
     * last comment, when either has one per column. */
    int n_cols = 0;
    const char* types[ASCII_MAX_COLS];
    if (data) {
      n_cols = split(data, delimiter, fields, ASCII_MAX_COLS);
      if (n_cols > ASCII_MAX_COLS)
        n_cols = ASCII_MAX_COLS;
      for (int k = 0; k < n_cols; k++)
        types[k] = is_number(fields[k]) ? "number" : "text";
    }

    char** names = NULL;
    char* name_fields[ASCII_MAX_COLS];
    char* sources[2] = {names_line, comment_names};
    for (int s = 0; s < 2 && !names; s++) {
      if (!sources[s])
        continue;
      const char d = s == 1 && !strchr(sources[s], delimiter) ? ' ' : delimiter;
      const int n = split(sources[s], d, name_fields, ASCII_MAX_COLS);
      if (!data && n <= ASCII_MAX_COLS) {
        n_cols = n;
        for (int k = 0; k < n; k++)
          types[k] = "other";
      }
      if (n == n_cols)
        names = name_fields;
    }

    for (int k = 0; k < n_cols; k++) {
      char number[16];
      snprintf(number, sizeof(number), "%d", k + 1);
      sif__file_info_column(
        info, t, names ? names[k] : number, types[k], 1, NULL);
    }
  }

  free(names_line);
  free(comment_names);
  free(data);
  return status;
}

/* A hint for a field that is not a number, when it looks like two joined by
 * a delimiter other than the file's. */
static const char* delimiter_hint(const char* field, char delimiter) {
  if (delimiter != ',' && strchr(field, ','))
    return " (comma-separated in places?)";
  if (delimiter != ';' && strchr(field, ';'))
    return " (semicolon-separated in places?)";
  return "";
}

/*
 * Every row, by the layout the description found: as many fields as the
 * first row, and a number in every column the first row had one in. The
 * numbers are ranged; a text column is counted.
 */
int sif__file_check_ascii(sif_file_check_t* c) {
  const sif_file_info_t* info = c->info;
  const info_box_t* box = (const info_box_t*)info;
  const char* path = info->path;
  if (info->n_tables != 1)
    return SIF_OK;
  const sif_file_table_t* tb = &info->tables[0];
  const char delimiter = box->ascii_delimiter ? box->ascii_delimiter : ' ';

  FILE* f = fopen(path, "r");
  if (!f) {
    sif__io_os_error(TAG, path, NULL, errno);
    return SIF_OK;
  }

  char* fields[ASCII_MAX_COLS];
  char* line = NULL;
  size_t cap = 0;
  uint64_t lineno = 0, rows = 0;
  bool header = box->ascii_header;
  while (getline(&line, &cap, f) >= 0) {
    lineno++;
    if (comment_text(line))
      continue;
    const char* p = line;
    while (is_blank(*p))
      p++;
    if (is_line_end(*p))
      continue;
    if (header) {
      header = false;
      continue;
    }

    rows++;
    const int n = split(line, delimiter, fields, ASCII_MAX_COLS);
    if ((uint32_t)n != tb->n_columns)
      sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
        "%s:%" PRIu64 ": %d field%s (expected %u, as the first row)", path,
        lineno, n, n == 1 ? "" : "s", tb->n_columns);
    for (int k = 0; k < n && k < ASCII_MAX_COLS && (uint32_t)k < tb->n_columns;
      k++) {
      const sif_file_column_t* col = &tb->columns[k];
      sif_file_stats_t* st = &c->column_stats[0][k];
      if (strcmp(col->type, "number") != 0) {
        st->n++;
        continue;
      }
      if (is_number(fields[k])) {
        sif__file_stats_add(st, strtod(fields[k], NULL));
      } else {
        sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
          "%s:%" PRIu64 ": column %d (%s): '%s' (expected a number)%s", path,
          lineno, k + 1, col->name, fields[k],
          delimiter_hint(fields[k], delimiter));
      }
    }
  }
  free(line);
  if (ferror(f))
    sif__io_os_error(TAG, path, "read", errno);
  fclose(f);

  if (rows != tb->n_rows)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: %" PRIu64 " rows read (expected %" PRIu64
      "; the file changed while it was checked?)",
      path, rows, tb->n_rows);
  return SIF_OK;
}

#undef ASCII_MAX_COLS

/* ------------------------------------------------------------------------ */
/* .xfield and .xgrid                                                        */
/* ------------------------------------------------------------------------ */

/* Reads the first @p size bytes of a file, which has to start with
 * @p magic. */
static int read_head(const char* path, void* head, size_t size,
  const char* magic, const char* what) {
  FILE* f = fopen(path, "rb");
  if (!f)
    return sif__io_os_error(TAG, path, NULL, errno);
  const size_t got = fread(head, 1, size, f);
  fclose(f);
  if (got < 4 || memcmp(head, magic, 4) != 0) {
    SIF_LOG_ERROR(TAG, "%s: not %s (no %s at byte 0)", path, what, magic);
    return SIF_ERR_IO;
  }
  if (got < size) {
    SIF_LOG_ERROR(TAG, "%s: %zu bytes, shorter than %s's header (%zu)", path,
      got, what, size);
    return SIF_ERR_IO;
  }
  return SIF_OK;
}

static int describe_xfield(sif_file_info_t* info) {
  sif_xfield_header_t h;
  const int status =
    read_head(info->path, &h, sizeof(h), SIF_XFIELD_MAGIC, "an .xfield file");
  if (status != SIF_OK)
    return status;

  const char* type = h.is_double ? "float64" : "float32";
  sif__file_info_detail(info, "version %u, %s precision", h.version,
    h.is_double ? "double" : "single");
  sif__file_info_meta_real(
    info, SIF__FILE_INFO_FILE, 0, "box_length", h.box_length);
  if (h.version >= 2)
    sif__file_info_meta_string(
      info, SIF__FILE_INFO_FILE, 0, "checksum", h.crc32 ? "crc32" : "none");

  /* In the order of the blocks in the file. */
  const uint32_t t = sif__file_info_table(info, "particles", h.n_particles);
  static const char* const cols[] = {"x", "y", "z", "vx", "vy", "vz", "w"};
  for (int k = 0; k < 7; k++) {
    if ((k >= 3 && k < 6 && !h.has_velocities) || (k == 6 && !h.has_weights))
      continue;
    sif__file_info_column(info, t, cols[k], type, 1, NULL);
  }
  return SIF_OK;
}

static int describe_xgrid(sif_file_info_t* info) {
  sif_xgrid_header_t h;
  const int status =
    read_head(info->path, &h, sizeof(h), SIF_XGRID_MAGIC, "an .xgrid file");
  if (status != SIF_OK)
    return status;

  static const char* const contents[] = {
    "empty", "density", "density contrast"};
  static const char* const schemes[] = {"unknown", "ngp", "cic", "tsc", "pcs"};
  const uint32_t scheme = h.assignment & 0xffu;

  sif__file_info_detail(info, "version %u, %s precision", h.version,
    h.is_double ? "double" : "single");
  sif__file_info_meta_real(
    info, SIF__FILE_INFO_FILE, 0, "box_length", h.box_length);
  sif__file_info_meta_int(info, SIF__FILE_INFO_FILE, 0, "n_cells", h.n_cells);
  sif__file_info_meta_string(info, SIF__FILE_INFO_FILE, 0, "content",
    h.content < 3 ? contents[h.content] : "unknown");
  sif__file_info_meta_string(info, SIF__FILE_INFO_FILE, 0, "assignment",
    scheme < 5 ? schemes[scheme] : "unknown");
  sif__file_info_meta_int(
    info, SIF__FILE_INFO_FILE, 0, "interlaced", (h.assignment >> 8) & 1u);
  if (h.version >= 2)
    sif__file_info_meta_string(
      info, SIF__FILE_INFO_FILE, 0, "checksum", h.crc32 ? "crc32" : "none");

  const uint64_t dims[3] = {h.n_cells, h.n_cells, h.n_cells};
  sif__file_info_array(
    info, "cells", h.is_double ? "float64" : "float32", 3, dims);
  return SIF_OK;
}

/* Values read at a time by the checks of the binary formats. */
#define CHECK_CHUNK ((size_t)1 << 20)

/* One value of @p elem bytes, either precision. */
static double value_at(const unsigned char* p, size_t elem) {
  if (elem == 8) {
    double v;
    memcpy(&v, p, 8);
    return v;
  }
  float v;
  memcpy(&v, p, 4);
  return v;
}

/*
 * The size the header implies, the checksum over every block, every value;
 * and of the contents, positions inside the box and weights not negative.
 * Either precision is read: a file this build could not load is still
 * checked whole, and the mismatch is a warning.
 */
int sif__file_check_xfield(sif_file_check_t* c) {
  const sif_file_info_t* info = c->info;
  const char* path = info->path;
  sif__file_check_content(c, "sif field");

  FILE* f = fopen(path, "rb");
  if (!f) {
    sif__io_os_error(TAG, path, NULL, errno);
    return SIF_OK;
  }
  sif_xfield_header_t h;
  if (fread(&h, sizeof(h), 1, f) != 1) {
    fclose(f);
    return SIF_OK; /* the description said why */
  }

  const size_t elem = h.is_double ? 8 : 4;
  const int n_blocks = 3 + (h.has_velocities ? 3 : 0) + (h.has_weights ? 1 : 0);
  const uint64_t expect =
    sizeof(h) + (uint64_t)n_blocks * h.n_particles * (uint64_t)elem;

  if (h.version > SIF_XFIELD_VERSION)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: version %u (this build reads up to %u)", path, h.version,
      (unsigned)SIF_XFIELD_VERSION);
  if ((h.is_double != 0) != (sizeof(sif_real) == 8))
    sif__file_check_problem(c, SIF_FILE_PROBLEM_WARNING,
      "%s: %s precision (this build reads %s)", path,
      h.is_double ? "double" : "single",
      sizeof(sif_real) == 8 ? "double" : "single");
  if (info->size_bytes != expect) {
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: %" PRIu64 " bytes (expected %" PRIu64 ", from the header: %s)", path,
      info->size_bytes, expect,
      info->size_bytes < expect ? "truncated" : "trailing bytes");
  }
  if (info->size_bytes < expect || h.version > SIF_XFIELD_VERSION) {
    fclose(f);
    return SIF_OK;
  }

  unsigned char* buf = malloc(CHECK_CHUNK * elem);
  if (!buf) {
    fclose(f);
    SIF_LOG_ERROR(
      TAG, "out of memory: %zu-byte read buffer", CHECK_CHUNK * elem);
    return SIF_ERR_ALLOC;
  }

  /* The blocks in file order, which is the columns' order. */
  uint32_t crc = SIF_CRC32_INIT;
  uint64_t outside[3] = {0, 0, 0}, negative = 0;
  bool read_ok = true;
  for (int b = 0; b < n_blocks && read_ok; b++) {
    sif_file_stats_t* st = sif__file_check_column(c, 0, (uint32_t)b);
    const bool weight = h.has_weights && b == n_blocks - 1;
    for (uint64_t done = 0; done < h.n_particles;) {
      const size_t n =
        (size_t)(h.n_particles - done < CHECK_CHUNK ? h.n_particles - done
                                                    : CHECK_CHUNK);
      if (fread(buf, elem, n, f) != n) {
        sif__io_os_error(TAG, path, info->tables[0].columns[b].name, errno);
        read_ok = false;
        break;
      }
      crc = sif_crc32_update(crc, buf, n * elem);
      for (size_t i = 0; i < n; i++) {
        const double v = value_at(buf + elem * i, elem);
        if (st)
          sif__file_stats_add(st, v);
        if (b < 3 && !sif__box_accepts(v, h.box_length))
          outside[b]++;
        if (weight && v < 0.0)
          negative++;
      }
      done += n;
    }
  }
  free(buf);
  fclose(f);
  if (!read_ok)
    return SIF_OK;

  if (h.version >= 2 && sif_crc32_final(crc) != h.crc32)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: checksum %08x (expected %08x, from the header: corrupt)", path,
      sif_crc32_final(crc), h.crc32);
  if (h.version < 2)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_WARNING,
      "%s: version 1, without a checksum: the values cannot be verified", path);
  static const char* const axes[3] = {"x", "y", "z"};
  for (int k = 0; k < 3; k++)
    if (outside[k])
      sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
        "%s: %s: %" PRIu64 " of %" PRIu64
        " outside [0, %g) or not finite (expected inside the box)",
        path, axes[k], outside[k], h.n_particles, h.box_length);
  if (negative)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: w: %" PRIu64 " negative (expected weights >= 0)", path, negative);
  return SIF_OK;
}

/*
 * The size, the checksum, every cell; and of the contents, a density not
 * negative and a density contrast not below -1.
 */
int sif__file_check_xgrid(sif_file_check_t* c) {
  const sif_file_info_t* info = c->info;
  const char* path = info->path;
  sif__file_check_content(c, "sif grid");

  FILE* f = fopen(path, "rb");
  if (!f) {
    sif__io_os_error(TAG, path, NULL, errno);
    return SIF_OK;
  }
  sif_xgrid_header_t h;
  if (fread(&h, sizeof(h), 1, f) != 1) {
    fclose(f);
    return SIF_OK;
  }

  const size_t elem = h.is_double ? 8 : 4;
  const uint64_t n3 = (uint64_t)h.n_cells * h.n_cells * h.n_cells;
  const uint64_t expect = sizeof(h) + h.total_cells * (uint64_t)elem;
  if (h.version > SIF_XGRID_VERSION)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: version %u (this build reads up to %u)", path, h.version,
      (unsigned)SIF_XGRID_VERSION);
  if ((h.is_double != 0) != (sizeof(sif_real) == 8))
    sif__file_check_problem(c, SIF_FILE_PROBLEM_WARNING,
      "%s: %s precision (this build reads %s)", path,
      h.is_double ? "double" : "single",
      sizeof(sif_real) == 8 ? "double" : "single");
  if (h.total_cells != n3)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: %" PRIu64 " cells (expected %u^3 = %" PRIu64 ")", path,
      h.total_cells, h.n_cells, n3);
  if (info->size_bytes != expect)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: %" PRIu64 " bytes (expected %" PRIu64 ", from the header: %s)", path,
      info->size_bytes, expect,
      info->size_bytes < expect ? "truncated" : "trailing bytes");
  if (info->size_bytes < expect || h.version > SIF_XGRID_VERSION) {
    fclose(f);
    return SIF_OK;
  }

  unsigned char* buf = malloc(CHECK_CHUNK * elem);
  if (!buf) {
    fclose(f);
    SIF_LOG_ERROR(
      TAG, "out of memory: %zu-byte read buffer", CHECK_CHUNK * elem);
    return SIF_ERR_ALLOC;
  }
  sif_file_stats_t* st = sif__file_check_array(c, 0);
  uint32_t crc = SIF_CRC32_INIT;
  bool read_ok = true;
  for (uint64_t done = 0; done < h.total_cells;) {
    const size_t n =
      (size_t)(h.total_cells - done < CHECK_CHUNK ? h.total_cells - done
                                                  : CHECK_CHUNK);
    if (fread(buf, elem, n, f) != n) {
      sif__io_os_error(TAG, path, "cells", errno);
      read_ok = false;
      break;
    }
    crc = sif_crc32_update(crc, buf, n * elem);
    for (size_t i = 0; st && i < n; i++)
      sif__file_stats_add(st, value_at(buf + elem * i, elem));
    done += n;
  }
  free(buf);
  fclose(f);
  if (!read_ok)
    return SIF_OK;

  if (h.version >= 2 && sif_crc32_final(crc) != h.crc32)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: checksum %08x (expected %08x, from the header: corrupt)", path,
      sif_crc32_final(crc), h.crc32);
  if (h.version < 2)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_WARNING,
      "%s: version 1, without a checksum: the values cannot be verified", path);
  if (st && st->n_nonfinite)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: cells: %" PRIu64 " not finite", path, st->n_nonfinite);
  /* Rounding brings an empty cell's contrast a little under -1. */
  if (st && st->n > st->n_nonfinite) {
    if (h.content == SIF_GRID_DENSITY && st->min < 0.0)
      sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
        "%s: cells: smallest density %g (expected >= 0)", path, st->min);
    if (h.content == SIF_GRID_DENSITY_CONTRAST && st->min < -1.0 - 1e-5)
      sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
        "%s: cells: smallest density contrast %g (expected >= -1)", path,
        st->min);
  }
  return SIF_OK;
}

#undef CHECK_CHUNK

/* ------------------------------------------------------------------------ */
/* reading                                                                   */
/* ------------------------------------------------------------------------ */

sif_file_info_t* sif_file_info_read(
  const char* path, sif_file_format_t format) {
  if (!path) {
    SIF_LOG_ERROR(TAG, "sif_file_info_read: NULL path");
    return NULL;
  }
  if ((unsigned)format >= N_FORMATS) {
    SIF_LOG_ERROR(TAG, "sif_file_info_read: format %d out of range", format);
    return NULL;
  }

  if (format == SIF_FILE_FORMAT_UNKNOWN) {
    format = sif_file_format_detect(path);
    if (format == SIF_FILE_FORMAT_UNKNOWN) {
      SIF_LOG_ERROR(TAG,
        "%s: not a format sif reads (ASCII, FITS, HDF5, GADGET, .xfield, "
        ".xgrid, exodus checkpoint)",
        path);
      return NULL;
    }
  }

  /* A snapshot's path may be a base name, and its size is its files'. */
  uint64_t bytes = 0;
  if (format != SIF_FILE_FORMAT_GADGET &&
      sif__io_check_readable(TAG, path, &bytes) != SIF_OK)
    return NULL;

  info_box_t* box = calloc(1, sizeof(info_box_t));
  if (!box) {
    SIF_LOG_ERROR(TAG, "out of memory");
    return NULL;
  }
  sif_file_info_t* info = &box->info;
  info->path = copy_text(info, path);
  info->detail = copy_text(info, "");
  info->format = format;
  info->n_files = 1;
  info->size_bytes = bytes;

  int status = SIF_ERR_INVALID;
  switch (format) {
  case SIF_FILE_FORMAT_ASCII:
    status = describe_ascii(info);
    break;
  case SIF_FILE_FORMAT_FITS:
    status = sif__file_info_fits(info);
    break;
  case SIF_FILE_FORMAT_HDF5:
    status = sif__file_info_hdf5(info);
    break;
  case SIF_FILE_FORMAT_GADGET:
    status = sif__file_info_gadget(info);
    break;
  case SIF_FILE_FORMAT_XFIELD:
    status = describe_xfield(info);
    break;
  case SIF_FILE_FORMAT_XGRID:
    status = describe_xgrid(info);
    break;
  case SIF_FILE_FORMAT_CHECKPOINT:
    status = sif__file_info_checkpoint(info);
    break;
  case SIF_FILE_FORMAT_UNKNOWN:
    break;
  }

  if (status == SIF_OK && failed(info)) {
    SIF_LOG_ERROR(TAG, "%s: out of memory describing the file", path);
    status = SIF_ERR_ALLOC;
  }
  if (status != SIF_OK) {
    sif_file_info_free(info);
    return NULL;
  }
  return info;
}

/* ------------------------------------------------------------------------ */
/* printing                                                                  */
/* ------------------------------------------------------------------------ */

/* 52 MB, 1.4 GB; under a kilobyte, in bytes. Decimal units, as ls -h --si
 * and the file managers give them. */
static void human_size(char* buf, size_t len, uint64_t bytes) {
  static const char* const units[] = {"kB", "MB", "GB", "TB", "PB"};
  if (bytes < 1000) {
    snprintf(buf, len, "%" PRIu64 " byte%s", bytes, bytes == 1 ? "" : "s");
    return;
  }
  double v = (double)bytes / 1000.0;
  int u = 0;
  while (v >= 1000.0 && u < 4) {
    v /= 1000.0;
    u++;
  }
  snprintf(buf, len, "%.3g %s", v, units[u]);
}

static void print_value(const sif_file_meta_t* m, FILE* s) {
  switch (m->kind) {
  case SIF_FILE_META_INT:
    fprintf(s, "%" PRId64, m->i);
    break;
  case SIF_FILE_META_REAL:
    fprintf(s, "%.10g", m->r);
    break;
  case SIF_FILE_META_STRING:
    fputs(m->s ? m->s : "", s);
    break;
  }
}

static void print_meta(
  const sif_file_meta_t* meta, uint32_t n, const char* indent, FILE* s) {
  int width = 12;
  for (uint32_t i = 0; i < n; i++) {
    const int len = (int)strlen(meta[i].key);
    if (len > width)
      width = len;
  }
  for (uint32_t i = 0; i < n; i++) {
    fprintf(s, "%s%-*s  ", indent, width, meta[i].key);
    print_value(&meta[i], s);
    fputc('\n', s);
  }
}

static const char* plural(uint64_t n) { return n == 1 ? "" : "s"; }

/* A value of a check's statistics: "-" when there is none. */
static void stat_text(char* buf, size_t len, double v) {
  if (isnan(v))
    snprintf(buf, len, "-");
  else
    snprintf(buf, len, "%.8g", v);
}

static void print_stats_header(int width, bool units, FILE* s) {
  fprintf(s, "  %-*s  %-14s  %s%12s  %10s  %15s  %15s\n", width, "column",
    "type", units ? "unit          " : "", "read", "non-finite", "min", "max");
}

static void print_stats(const sif_file_stats_t* st, FILE* s) {
  char lo[32], hi[32];
  stat_text(lo, sizeof(lo), st->min);
  stat_text(hi, sizeof(hi), st->max);
  fprintf(s, "%12" PRIu64 "  %10" PRIu64 "  %15s  %15s\n", st->n,
    st->n_nonfinite, lo, hi);
}

int sif_file_info_print(const sif_file_info_t* info, FILE* s) {
  if (!info || !s) {
    SIF_LOG_ERROR(TAG, "sif_file_info_print: NULL argument");
    return SIF_ERR_INVALID;
  }
  return sif__file_info_print_stats(info, NULL, NULL, s);
}

int sif__file_info_print_stats(const sif_file_info_t* info,
  sif_file_stats_t* const* column_stats, const sif_file_stats_t* array_stats,
  FILE* s) {
  char size[32];
  human_size(size, sizeof(size), info->size_bytes);
  fprintf(s, "file      %s\n", info->path);
  fprintf(s, "format    %s%s%s\n", FORMAT_TITLES[info->format],
    info->detail && info->detail[0] ? ", " : "",
    info->detail ? info->detail : "");
  if (info->n_files > 1)
    fprintf(s, "files     %u\n", info->n_files);
  fprintf(s, "size      %s\n", size);

  if (info->n_meta) {
    fputs("\nmetadata\n", s);
    print_meta(info->meta, info->n_meta, "  ", s);
  }

  for (uint32_t t = 0; t < info->n_tables; t++) {
    const sif_file_table_t* tb = &info->tables[t];
    fprintf(s, "\ntable%s%s, %" PRIu64 " row%s, %u column%s\n",
      tb->name[0] ? " " : "", tb->name, tb->n_rows, plural(tb->n_rows),
      tb->n_columns, plural(tb->n_columns));

    int width = 12;
    bool units = false;
    for (uint32_t c = 0; c < tb->n_columns; c++) {
      const int len = (int)strlen(tb->columns[c].name);
      if (len > width)
        width = len;
      units |= tb->columns[c].unit != NULL;
    }
    if (column_stats && tb->n_columns)
      print_stats_header(width, units, s);
    for (uint32_t c = 0; c < tb->n_columns; c++) {
      const sif_file_column_t* col = &tb->columns[c];
      char type[48];
      if (col->width > 1)
        snprintf(type, sizeof(type), "%s[%" PRIu64 "]", col->type, col->width);
      else
        snprintf(type, sizeof(type), "%s", col->type);
      if (column_stats) {
        fprintf(s, "  %-*s  %-14s  ", width, col->name, type);
        if (units)
          fprintf(s, "%-14s", col->unit ? col->unit : "");
        print_stats(&column_stats[t][c], s);
      } else if (col->unit) {
        fprintf(s, "  %-*s  %-12s  %s\n", width, col->name, type, col->unit);
      } else {
        fprintf(s, "  %-*s  %s\n", width, col->name, type);
      }
    }
    if (tb->n_meta) {
      fputs("  metadata\n", s);
      print_meta(tb->meta, tb->n_meta, "    ", s);
    }
  }

  for (uint32_t i = 0; i < info->n_arrays; i++) {
    const sif_file_array_t* a = &info->arrays[i];
    fprintf(s, "\narray %s, %s ", a->name, a->type);
    if (a->rank == 0)
      fputs("scalar", s);
    for (uint32_t k = 0; k < a->rank; k++)
      fprintf(s, "%s%" PRIu64 "%s", k ? ", " : "(", a->dims[k],
        k + 1 == a->rank ? ")" : "");
    fputc('\n', s);
    if (array_stats) {
      fprintf(
        s, "  %12s  %10s  %15s  %15s\n", "read", "non-finite", "min", "max");
      fputs("  ", s);
      print_stats(&array_stats[i], s);
    }
    if (a->n_meta) {
      fputs("  metadata\n", s);
      print_meta(a->meta, a->n_meta, "    ", s);
    }
  }

  fflush(s);
  return ferror(s) ? SIF_ERR_IO : SIF_OK;
}
