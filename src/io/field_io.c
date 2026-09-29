/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/field_io.h"

#include "internal.h"
#include "sif/utils/align.h"
#include "sif/utils/crc32.h"
#include "sif/utils/logger.h"
#include "structures/field_internal.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Columns a format string may describe, and the longest input line the parser
 * accepts. A row wider than this is read in pieces and its tail parses as a
 * row of its own, so the limit has to be generous. */
#define MAX_COLS 32
#define LINE_CAP 2048

/* Wrapped in do/while so an unbraced `if` around one of these cannot swallow
 * the failure path. */
#define CHECK_WRITE(ptr, size, count, stream, msg)                             \
  do {                                                                         \
    if (fwrite((ptr), (size), (count), (stream)) != (count)) {                 \
      SIF_LOG_ERROR("xfield", "failed to write %s", (msg));                    \
      fclose(file);                                                            \
      return SIF_ERR_IO;                                                       \
    }                                                                          \
  } while (0)

#define CHECK_PREAD(fd, dest, bytes, offset, msg, file_ptr)                    \
  do {                                                                         \
    if (sif__io_pread_parallel((fd), (dest), (bytes), (offset)) != SIF_OK) {   \
      SIF_LOG_ERROR("xfield", "parallel read error for %s", (msg));            \
      fclose(file_ptr);                                                        \
      return SIF_ERR_IO;                                                       \
    }                                                                          \
  } while (0)

/* --- binary format --- */

int sif_field_write(
  const char* filepath, const sif_field_t* field, double box_length) {
  if (!filepath || !field) {
    SIF_LOG_ERROR("xfield", "invalid arguments");
    return SIF_ERR_INVALID;
  }

  /* Positions are the one block every .xfield carries, so a field without them
   * has nothing to write. Checked here rather than trusted, because the very
   * next thing this function does is hand the pointers to the checksum. */
  if (field->n_particles == 0 || !field->x || !field->y || !field->z) {
    SIF_LOG_ERROR("xfield", "field carries no positions to write");
    return SIF_ERR_INVALID;
  }

  /* The format has nowhere to say what the positions are, so a field of sky
   * coordinates would read back as one of Cartesian positions. */
  if (sif__field_require_cartesian(field, "xfield") != SIF_OK)
    return SIF_ERR_INVALID;

  FILE* file = fopen(filepath, "wb");
  if (!file) {
    SIF_LOG_ERROR("xfield", "failed to open %s", filepath);
    return SIF_ERR_IO;
  }

  sif_xfield_header_t header;
  memset(&header, 0, sizeof(sif_xfield_header_t));

  strncpy(header.magic, SIF_XFIELD_MAGIC, 4);
  header.version = SIF_XFIELD_VERSION;
  header.n_particles = field->n_particles;
  header.box_length = box_length;
  header.has_weights = (field->weights != NULL) ? 1 : 0;
  header.has_velocities = (field->vx != NULL) ? 1 : 0;
  header.is_double = (sizeof(sif_real) == 8) ? 1 : 0;

  uint64_t n = field->n_particles;
  const size_t array_bytes = (size_t)n * sizeof(sif_real);

  /* The checksum covers the payload in write order, and the header carries it,
   * so it has to be computed before the header goes out. Everything is already
   * in memory, which is why this needs no seeking back. */
  uint32_t crc = SIF_CRC32_INIT;
  crc = sif_crc32_update(crc, field->x, array_bytes);
  crc = sif_crc32_update(crc, field->y, array_bytes);
  crc = sif_crc32_update(crc, field->z, array_bytes);
  if (header.has_velocities) {
    crc = sif_crc32_update(crc, field->vx, array_bytes);
    crc = sif_crc32_update(crc, field->vy, array_bytes);
    crc = sif_crc32_update(crc, field->vz, array_bytes);
  }
  if (header.has_weights)
    crc = sif_crc32_update(crc, field->weights, array_bytes);
  header.crc32 = sif_crc32_final(crc);

  CHECK_WRITE(&header, sizeof(sif_xfield_header_t), 1, file, "header");

  CHECK_WRITE(field->x, sizeof(sif_real), n, file, "x coordinates");
  CHECK_WRITE(field->y, sizeof(sif_real), n, file, "y coordinates");
  CHECK_WRITE(field->z, sizeof(sif_real), n, file, "z coordinates");

  if (header.has_velocities) {
    CHECK_WRITE(field->vx, sizeof(sif_real), n, file, "vx velocities");
    CHECK_WRITE(field->vy, sizeof(sif_real), n, file, "vy velocities");
    CHECK_WRITE(field->vz, sizeof(sif_real), n, file, "vz velocities");
  }

  if (header.has_weights) {
    CHECK_WRITE(field->weights, sizeof(sif_real), n, file, "weights");
  }

  /* fwrite() succeeding only means the bytes reached the stdio buffer. A full
   * disk or an exceeded quota surfaces at the flush inside fclose(), and
   * returning SIF_OK without looking would leave the caller with a truncated
   * file it believes is complete. */
  const bool ok = (ferror(file) == 0);
  if (fclose(file) != 0 || !ok) {
    SIF_LOG_ERROR("xfield", "failed to flush %s to disk", filepath);
    return SIF_ERR_IO;
  }

  SIF_LOG_INFO("xfield", "wrote %" PRIu64 " particles to %s", n, filepath);
  return SIF_OK;
}

/*
 * Rejects a header that this build cannot read into any field.
 *
 * Split out because both readers need it before they trust a single number
 * from the header -- sif_field_read() sizes an allocation from n_particles,
 * and doing that first would mean a wrong file is diagnosed only after trying
 * to allocate whatever its bytes happened to say.
 */
static int header_validate(
  const sif_xfield_header_t* header, const char* filepath) {
  if (strncmp(header->magic, SIF_XFIELD_MAGIC, 4) != 0) {
    SIF_LOG_ERROR("xfield", "%s is not a valid xfield binary", filepath);
    return SIF_ERR_IO;
  }

  /* Older files stay readable; newer ones cannot be, since the layout they
   * describe is one this build has never seen. */
  if (header->version > SIF_XFIELD_VERSION) {
    SIF_LOG_ERROR("xfield",
      "%s is version %u, but this build reads up to version %u", filepath,
      header->version, (unsigned)SIF_XFIELD_VERSION);
    return SIF_ERR_IO;
  }

  /* The payload is raw sif_real, so a float file read by a double build is not
   * a conversion but a reinterpretation. Refused rather than adapted to. */
  const uint32_t current_is_double = (sizeof(sif_real) == 8) ? 1u : 0u;
  if (header->is_double != current_is_double) {
    SIF_LOG_ERROR("xfield", "%s is %s, this build is %s", filepath,
      header->is_double ? "FP64" : "FP32", current_is_double ? "FP64" : "FP32");
    return SIF_ERR_IO;
  }

  return SIF_OK;
}

int sif_field_read_into(sif_field_t* field, const char* filepath) {
  if (!filepath || !field)
    return SIF_ERR_INVALID;

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    SIF_LOG_ERROR("xfield", "could not open %s for reading", filepath);
    return SIF_ERR_IO;
  }

  sif_xfield_header_t header;
  if (fread(&header, sizeof(sif_xfield_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xfield", "failed to read header from %s", filepath);
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header_validate(&header, filepath) != SIF_OK) {
    fclose(file);
    return SIF_ERR_IO;
  }

  /* The blocks are read straight into the caller's arrays, so the shape has to
   * agree exactly -- there is nowhere to put a longer file and no way to fill
   * a shorter one. */
  if (header.n_particles != field->n_particles) {
    SIF_LOG_ERROR("xfield",
      "geometry mismatch: %s has %" PRIu64
      " particles, target field has %" PRIu64,
      filepath, header.n_particles, field->n_particles);
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header.has_velocities && !field->vx) {
    SIF_LOG_ERROR("xfield", "target field lacks velocity allocation");
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header.has_weights && !field->weights) {
    SIF_LOG_ERROR("xfield", "target field lacks a weight allocation");
    fclose(file);
    return SIF_ERR_IO;
  }

  /* Hand the descriptor to the parallel reader. pread() carries its own offset
   * and ignores the stream position, so nothing has to be done about the
   * header bytes stdio has already buffered. */
  int fd = fileno(file);

  uint64_t n = header.n_particles;
  size_t array_bytes = (size_t)n * sizeof(sif_real);
  off_t current_offset = sizeof(sif_xfield_header_t);

  CHECK_PREAD(fd, field->x, array_bytes, current_offset, "x coordinates", file);
  current_offset += array_bytes;

  CHECK_PREAD(fd, field->y, array_bytes, current_offset, "y coordinates", file);
  current_offset += array_bytes;

  CHECK_PREAD(fd, field->z, array_bytes, current_offset, "z coordinates", file);
  current_offset += array_bytes;

  if (header.has_velocities) {
    CHECK_PREAD(
      fd, field->vx, array_bytes, current_offset, "vx velocities", file);
    current_offset += array_bytes;

    CHECK_PREAD(
      fd, field->vy, array_bytes, current_offset, "vy velocities", file);
    current_offset += array_bytes;

    CHECK_PREAD(
      fd, field->vz, array_bytes, current_offset, "vz velocities", file);
    current_offset += array_bytes;
  }

  if (header.has_weights) {
    CHECK_PREAD(
      fd, field->weights, array_bytes, current_offset, "weights", file);
  }

  fclose(file);

  /* Validate what was actually loaded, in the same order the writer folded it
   * in. A version 1 file predates the checksum and carries a zero here, which
   * is why the version rather than the value decides whether to check. */
  if (header.version >= 2) {
    uint32_t crc = SIF_CRC32_INIT;
    crc = sif_crc32_update(crc, field->x, array_bytes);
    crc = sif_crc32_update(crc, field->y, array_bytes);
    crc = sif_crc32_update(crc, field->z, array_bytes);
    if (header.has_velocities) {
      crc = sif_crc32_update(crc, field->vx, array_bytes);
      crc = sif_crc32_update(crc, field->vy, array_bytes);
      crc = sif_crc32_update(crc, field->vz, array_bytes);
    }
    if (header.has_weights)
      crc = sif_crc32_update(crc, field->weights, array_bytes);

    const uint32_t got = sif_crc32_final(crc);
    if (got != header.crc32) {
      SIF_LOG_ERROR("xfield", "%s is corrupt: checksum %08x, expected %08x",
        filepath, got, header.crc32);
      return SIF_ERR_IO;
    }
  } else {
    SIF_LOG_WARNING("xfield",
      "%s is a version %u file and carries no checksum; contents cannot be "
      "validated",
      filepath, header.version);
  }

  field->units = SIF_FIELD_CARTESIAN;
  return SIF_OK;
}

sif_field_t* sif_field_read(const char* filepath, double* out_box_length) {
  if (!filepath)
    return NULL;

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    SIF_LOG_ERROR("xfield", "could not open %s for reading", filepath);
    return NULL;
  }

  sif_xfield_header_t header;
  if (fread(&header, sizeof(sif_xfield_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xfield", "failed to read header from %s", filepath);
    fclose(file);
    return NULL;
  }
  fclose(file); /* the reader below opens it again for the payload */

  /* Validated before anything is sized from it: n_particles in a file that is
   * not an xfield at all is whatever those eight bytes happened to be, and
   * allocating that first turns a wrong filename into an out-of-memory. */
  if (header_validate(&header, filepath) != SIF_OK)
    return NULL;

  if (header.n_particles == 0) {
    SIF_LOG_ERROR("xfield", "%s holds no particles", filepath);
    return NULL;
  }

  if (out_box_length)
    *out_box_length = header.box_length;

  sif_field_t* field = sif_field_alloc(header.n_particles);
  if (!field)
    return NULL;

  /* Reserve, then read straight into the field's own arrays: the file never
   * goes through an intermediate copy. Only the blocks the header announces
   * are reserved, so the field ends up carrying exactly what the file did. */
  if (sif_field_reserve_positions(field) != SIF_OK) {
    SIF_LOG_ERROR("xfield", "OOM allocating position block");
    sif_field_free(field);
    return NULL;
  }

  if (header.has_velocities && sif_field_reserve_velocities(field) != SIF_OK) {
    SIF_LOG_ERROR("xfield", "OOM allocating velocity block");
    sif_field_free(field);
    return NULL;
  }

  if (header.has_weights && sif_field_reserve_weights(field) != SIF_OK) {
    SIF_LOG_ERROR("xfield", "OOM allocating weights block");
    sif_field_free(field);
    return NULL;
  }

  if (sif_field_read_into(field, filepath) != SIF_OK) {
    sif_field_free(field);
    return NULL;
  }

  SIF_LOG_INFO("xfield", "loaded %" PRIu64 " particles from %s",
    header.n_particles, filepath);
  return field;
}

/* --- column formats --- */

/* What one column of a file holds. The order is the bit each takes in a
 * format's record of what it has named. */
typedef enum {
  COL_SKIP,
  COL_X,
  COL_Y,
  COL_Z,
  COL_VX,
  COL_VY,
  COL_VZ,
  COL_W
} col_kind_t;

static const char* const COL_NAMES[] = {
  "*", "x", "y", "z", "vx", "vy", "vz", "w"};

typedef struct {
  col_kind_t kind;
  /* Binary only: the bytes this column takes per particle. 0 for a column
   * the format gave no width, which the binary reader sets from the
   * precision. */
  uint32_t width;
} col_t;

typedef struct {
  col_t cols[MAX_COLS];
  int n_cols;
  bool has_pos, has_vel, has_w;
} format_t;

/* The widest `*N` accepted: wide enough for any record a real file has,
 * narrow enough that a typo'd width is caught rather than skipping the file. */
#define MAX_SKIP_WIDTH 4096u

/*
 * Parses a column format; see the field_format group in field_io.h for the
 * language. Every way a format can be wrong is refused here, with a message
 * saying where: the readers used to skip characters they did not know, which
 * turned a typo into a silently shorter layout.
 *
 * The letters the language used before (u v w for velocity, m for weight)
 * are refused with a pointer to their replacement. w is the one letter whose
 * meaning changed, from vz to the weight, and no old format that was valid can
 * be misread as a new one: a velocity always took all three of u v w, and u is
 * refused.
 */
static int format_parse(const char* fmt, bool allow_widths, format_t* out) {
  memset(out, 0, sizeof(*out));
  if (!fmt) {
    SIF_LOG_ERROR("io", "no column format given");
    return SIF_ERR_INVALID;
  }

  unsigned named = 0;
  const char* p = fmt;

  while (*p) {
    const char c = (char)tolower((unsigned char)*p);
    if (c == ' ' || c == '\t' || c == ',') {
      p++;
      continue;
    }

    const long at = (long)(p - fmt) + 1;
    col_t col = {COL_SKIP, 0};

    switch (c) {
    case 'x':
      col.kind = COL_X;
      p++;
      break;
    case 'y':
      col.kind = COL_Y;
      p++;
      break;
    case 'z':
      col.kind = COL_Z;
      p++;
      break;
    case 'w':
      col.kind = COL_W;
      p++;
      break;
    case 'v': {
      const char d = (char)tolower((unsigned char)p[1]);
      if (d == 'x')
        col.kind = COL_VX;
      else if (d == 'y')
        col.kind = COL_VY;
      else if (d == 'z')
        col.kind = COL_VZ;
      else {
        SIF_LOG_ERROR("io",
          "format '%s', position %ld: 'v' must be followed by x, y or z -- "
          "velocities are vx vy vz",
          fmt, at);
        return SIF_ERR_INVALID;
      }
      p += 2;
      break;
    }
    case '*':
      p++;
      if (isdigit((unsigned char)*p)) {
        if (!allow_widths) {
          SIF_LOG_ERROR("io",
            "format '%s', position %ld: a skip width (*N) only applies to "
            "binary files; in a text file '*' skips one column",
            fmt, at);
          return SIF_ERR_INVALID;
        }
        char* end;
        const unsigned long width = strtoul(p, &end, 10);
        if (width == 0 || width > MAX_SKIP_WIDTH) {
          SIF_LOG_ERROR("io",
            "format '%s', position %ld: a skip width must be 1 to %u bytes",
            fmt, at, MAX_SKIP_WIDTH);
          return SIF_ERR_INVALID;
        }
        col.width = (uint32_t)width;
        p = end;
      }
      break;
    case 'm':
      SIF_LOG_ERROR("io",
        "format '%s', position %ld: 'm' is no longer a column name; the "
        "weight is 'w'",
        fmt, at);
      return SIF_ERR_INVALID;
    case 'u':
      SIF_LOG_ERROR("io",
        "format '%s', position %ld: 'u' is no longer a column name; "
        "velocities are 'vx vy vz'",
        fmt, at);
      return SIF_ERR_INVALID;
    default:
      SIF_LOG_ERROR("io",
        "format '%s', position %ld: '%c' is not a column name (x y z vx vy vz "
        "w, or * to skip)",
        fmt, at, *p);
      return SIF_ERR_INVALID;
    }

    if (col.kind != COL_SKIP) {
      const unsigned bit = 1u << col.kind;
      if (named & bit) {
        SIF_LOG_ERROR(
          "io", "format '%s' names '%s' twice", fmt, COL_NAMES[col.kind]);
        return SIF_ERR_INVALID;
      }
      named |= bit;
    }

    if (out->n_cols == MAX_COLS) {
      SIF_LOG_ERROR(
        "io", "format '%s' has more than %d columns", fmt, MAX_COLS);
      return SIF_ERR_INVALID;
    }
    out->cols[out->n_cols++] = col;
  }

  const unsigned pos = (1u << COL_X) | (1u << COL_Y) | (1u << COL_Z);
  const unsigned vel = (1u << COL_VX) | (1u << COL_VY) | (1u << COL_VZ);
  out->has_pos = (named & pos) != 0;
  out->has_vel = (named & vel) != 0;
  out->has_w = (named & (1u << COL_W)) != 0;

  /* Position and velocity are three-component quantities and are reserved as
   * one block each, so naming two of the three would allocate the third and
   * never write it. */
  if (out->has_pos && (named & pos) != pos) {
    SIF_LOG_ERROR("io", "format '%s' names only part of the position", fmt);
    return SIF_ERR_INVALID;
  }
  if (out->has_vel && (named & vel) != vel) {
    SIF_LOG_ERROR("io", "format '%s' names only part of the velocity", fmt);
    return SIF_ERR_INVALID;
  }

  /* A format of skips alone would consume the file and write nothing, leaving
   * a field whose memory reads as loaded. */
  if (named == 0) {
    SIF_LOG_ERROR("io", "format '%s' names no column to read", fmt);
    return SIF_ERR_INVALID;
  }

  return SIF_OK;
}

/*
 * Makes room for what a format loads, once the field's particle count is
 * known, and refuses what the field cannot take.
 *
 * Reserving is a no-op when a block is already there, which is what lets a
 * second file add columns to a field loaded from a first. That only works in
 * file order, so a field that has been Morton-sorted since cannot take new
 * columns without its positions: they would be matched to the wrong
 * particles.
 */
static int format_reserve(
  sif_field_t* field, const format_t* f, const char* fmt) {
  if (!f->has_pos && !field->x) {
    SIF_LOG_ERROR(
      "io", "format '%s' loads no positions and the field has none", fmt);
    return SIF_ERR_INVALID;
  }
  if (!f->has_pos && (field->state_flags & SIF_FIELD_STATE_MORTON_SORTED)) {
    SIF_LOG_ERROR("io",
      "the field has been Morton-sorted, so its particles are no longer in "
      "file order; read '%s' before sorting, or with the positions",
      fmt);
    return SIF_ERR_INVALID;
  }

  int status = SIF_OK;
  if (f->has_pos)
    status = sif_field_reserve_positions(field);
  if (status == SIF_OK && f->has_vel)
    status = sif_field_reserve_velocities(field);
  if (status == SIF_OK && f->has_w)
    status = sif_field_reserve_weights(field);
  return status;
}

/* After a successful read: whatever was derived from the old positions no
 * longer holds, including the permutation a sort recorded, which would
 * otherwise be used to gather columns assigned later. */
static void format_loaded(sif_field_t* field, const format_t* f) {
  if (!f->has_pos)
    return;
  field->units = SIF_FIELD_CARTESIAN;
  field->state_flags &= ~SIF_FIELD_STATE_BOUNDS_VALID;
  field->state_flags &= ~SIF_FIELD_STATE_MORTON_SORTED;
  sif_free_aligned(field->original_indices);
  field->original_indices = NULL;
}

static inline void col_assign(
  sif_field_t* field, col_kind_t kind, uint64_t k, sif_real v) {
  switch (kind) {
  case COL_X:
    field->x[k] = v;
    break;
  case COL_Y:
    field->y[k] = v;
    break;
  case COL_Z:
    field->z[k] = v;
    break;
  case COL_VX:
    field->vx[k] = v;
    break;
  case COL_VY:
    field->vy[k] = v;
    break;
  case COL_VZ:
    field->vz[k] = v;
    break;
  case COL_W:
    field->weights[k] = v;
    break;
  case COL_SKIP:
    break;
  }
}

/* --- ASCII input --- */

/*
 * Whether a line carries data, as opposed to nothing or a comment.
 *
 * Both the parser and the did-we-reach-the-end check ask this, and they have
 * to agree: a blank final line counted as data would report a file the field
 * was too small for, when in fact everything was read.
 */
static bool line_is_data(const char* line) {
  while (*line == ' ' || *line == '\t')
    line++;
  return *line != '\0' && *line != '\n' && *line != '\r' && *line != '#' &&
         *line != ';';
}

/*
 * Reads the next numeric field and advances the cursor past it: skips leading
 * whitespace, parses one number, and leaves the cursor after the field's
 * trailing delimiter. Returns 1 if a field was consumed, 0 at the end of the
 * line.
 *
 * Consumed is not parsed: a malformed field reads as 0.0, returns 1 and skips
 * to the next delimiter, so a text column inside a numeric file reads as
 * zeros rather than derailing the row.
 */
static int next_real(char** cursor, char delimiter, sif_real* out_val) {
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
    while (**cursor != '\0' && **cursor != '\n' && **cursor != delimiter)
      (*cursor)++;
  } else {
    *cursor = endptr;
  }

  if (**cursor == delimiter)
    (*cursor)++;

  return 1;
}

int sif_field_read_ascii_into(sif_field_t* field, const char* filepath,
  const char* fmt, char delimiter, uint32_t skip_header) {

  if (!field || !filepath || !fmt) {
    SIF_LOG_ERROR("io", "invalid field, path or format string");
    return SIF_ERR_INVALID;
  }

  format_t f;
  int status = format_parse(fmt, false, &f);
  if (status != SIF_OK)
    return status;

  /* A field with no particle count yet is sized from the file. The count is an
   * upper bound -- blank and comment lines are in it -- and the real total is
   * written back once the parse knows it. */
  if (field->n_particles == 0) {
    field->n_particles = sif__io_ascii_row_count(filepath, skip_header);
    if (field->n_particles == 0) {
      SIF_LOG_ERROR("io", "failed to read particles from %s", filepath);
      return SIF_ERR_IO;
    }
  }

  status = format_reserve(field, &f, fmt);
  if (status != SIF_OK)
    return status;

  FILE* file = fopen(filepath, "r");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s", filepath);
    return SIF_ERR_IO;
  }

  char line[LINE_CAP];
  for (uint32_t i = 0; i < skip_header; i++) {
    if (!fgets(line, sizeof(line), file))
      break;
  }

  uint64_t loaded = 0;
  uint64_t skipped = 0;

  while (loaded < field->n_particles && fgets(line, sizeof(line), file)) {
    if (!line_is_data(line)) {
      skipped++;
      continue;
    }

    char* cursor = line;
    sif_real val = 0.0;
    int col = 0;

    for (; col < f.n_cols; col++) {
      if (!next_real(&cursor, delimiter, &val))
        break;
      col_assign(field, f.cols[col].kind, loaded, val);
    }

    /* A row that ran out of columns is not a particle. Not counting it is what
     * makes it harmless: whatever was written at this index is overwritten by
     * the next good row, or falls outside n_particles once the count below is
     * corrected. Advancing instead would leave an entry whose remaining
     * components were never assigned. */
    if (col < f.n_cols) {
      skipped++;
      continue;
    }

    loaded++;
  }

  /* Whether the file had more to give is one more read, not another pass:
   * counting the rows again means streaming the whole file a second time. */
  bool has_more = false;
  while (!has_more && fgets(line, sizeof(line), file)) {
    has_more = line_is_data(line);
  }

  fclose(file);

  if (loaded == 0) {
    SIF_LOG_ERROR(
      "io", "%s holds no parsable rows for format '%s'", filepath, fmt);
    return SIF_ERR_IO;
  }

  if (loaded < field->n_particles) {
    SIF_LOG_INFO("io",
      "loaded %" PRIu64 " particles from %s (%" PRIu64
      " lines skipped as blank, comment or short)",
      loaded, filepath, skipped);
    field->n_particles = loaded;
  } else {
    SIF_LOG_INFO(
      "io", "loaded %" PRIu64 " particles from %s", loaded, filepath);
    if (has_more) {
      SIF_LOG_WARNING("io",
        "field filled at %" PRIu64 " particles; %s holds more rows", loaded,
        filepath);
    }
  }

  format_loaded(field, &f);
  return SIF_OK;
}

/* --- raw binary input --- */

/* Bytes read at a time: large enough that the per-read cost vanishes, small
 * enough to be nothing next to the field. */
#define BINARY_CHUNK ((size_t)4 << 20)

static bool binary_slot_ok(int value, int first, int n, const char* slot) {
  if (value >= first && value < first + n)
    return true;

  const char* meant = NULL;
  switch (value & ~0xff) {
  case SIF_BINARY_ROWS:
    meant = "layout";
    break;
  case SIF_BINARY_FLOAT32:
    meant = "precision";
    break;
  case SIF_BINARY_NATIVE:
    meant = "byte order";
    break;
  }
  if (meant && strcmp(meant, slot) != 0)
    SIF_LOG_ERROR("io",
      "the %s argument was given a %s option (0x%x); are two arguments "
      "swapped?",
      slot, meant, value);
  else
    SIF_LOG_ERROR("io", "0x%x is not a valid %s option", value, slot);
  return false;
}

/* One value of @p size bytes (4 or 8), in the file's byte order, as a
 * sif_real. */
static inline sif_real binary_value(
  const unsigned char* p, uint32_t size, bool swap) {
  if (size == 4) {
    uint32_t u;
    memcpy(&u, p, 4);
    if (swap)
      u = sif__io_bswap32(u);
    float v;
    memcpy(&v, &u, 4);
    return (sif_real)v;
  }
  uint64_t u;
  memcpy(&u, p, 8);
  if (swap)
    u = sif__io_bswap64(u);
  double v;
  memcpy(&v, &u, 8);
  return (sif_real)v;
}

/* Records one after another: every column of a particle together. Read in
 * whole records, so a chunk never ends inside one. */
static int binary_read_rows(FILE* file, sif_field_t* field, const format_t* f,
  uint64_t record, uint32_t size, bool swap, unsigned char* buf) {

  const uint64_t per_chunk = BINARY_CHUNK / record ? BINARY_CHUNK / record : 1;
  const uint64_t n = field->n_particles;

  for (uint64_t start = 0; start < n; start += per_chunk) {
    const uint64_t c = n - start < per_chunk ? n - start : per_chunk;
    if (fread(buf, (size_t)record, (size_t)c, file) != c)
      return SIF_ERR_IO;

    for (uint64_t j = 0; j < c; j++) {
      const unsigned char* r = buf + j * record;
      for (int col = 0; col < f->n_cols; col++) {
        if (f->cols[col].kind != COL_SKIP)
          col_assign(
            field, f->cols[col].kind, start + j, binary_value(r, size, swap));
        r += f->cols[col].width;
      }
    }
  }
  return SIF_OK;
}

/* Columns one after another: every particle of a column together. A skipped
 * column is a whole block, stepped over rather than read. */
static int binary_read_blocks(FILE* file, sif_field_t* field, const format_t* f,
  off_t data_start, uint32_t size, bool swap, unsigned char* buf) {

  const uint64_t n = field->n_particles;
  off_t block = data_start;

  for (int col = 0; col < f->n_cols; col++) {
    const col_t* c = &f->cols[col];
    if (c->kind != COL_SKIP) {
      if (fseeko(file, block, SEEK_SET) != 0)
        return SIF_ERR_IO;

      const uint64_t per_chunk = BINARY_CHUNK / size;
      for (uint64_t start = 0; start < n; start += per_chunk) {
        const uint64_t k = n - start < per_chunk ? n - start : per_chunk;
        if (fread(buf, size, (size_t)k, file) != k)
          return SIF_ERR_IO;
        for (uint64_t j = 0; j < k; j++)
          col_assign(field, c->kind, start + j,
            binary_value(buf + j * size, size, swap));
      }
    }
    block += (off_t)((uint64_t)c->width * n);
  }
  return SIF_OK;
}

int sif_field_read_binary_into(sif_field_t* field, const char* filepath,
  const char* fmt, sif_binary_layout_t layout, sif_binary_precision_t precision,
  sif_binary_endian_t endian, uint64_t header_bytes) {

  if (!field || !filepath || !fmt) {
    SIF_LOG_ERROR("io", "invalid field, path or format string");
    return SIF_ERR_INVALID;
  }

  if (!binary_slot_ok((int)layout, SIF_BINARY_ROWS, 2, "layout") ||
      !binary_slot_ok((int)precision, SIF_BINARY_FLOAT32, 2, "precision") ||
      !binary_slot_ok((int)endian, SIF_BINARY_NATIVE, 3, "byte order"))
    return SIF_ERR_INVALID;

  format_t f;
  int status = format_parse(fmt, true, &f);
  if (status != SIF_OK)
    return status;

  /* Every column without a width of its own is one value at the file's
   * precision; the record is the sum of them all. */
  const uint32_t size = precision == SIF_BINARY_FLOAT64 ? 8u : 4u;
  uint64_t record = 0;
  for (int col = 0; col < f.n_cols; col++) {
    if (f.cols[col].width == 0)
      f.cols[col].width = size;
    record += f.cols[col].width;
  }

  struct stat st;
  if (stat(filepath, &st) != 0 || !S_ISREG(st.st_mode)) {
    SIF_LOG_ERROR("io", "could not open %s", filepath);
    return SIF_ERR_IO;
  }
  const uint64_t file_bytes = (uint64_t)st.st_size;
  if (file_bytes < header_bytes) {
    SIF_LOG_ERROR("io",
      "%s is %" PRIu64 " bytes, shorter than its %" PRIu64 "-byte header",
      filepath, file_bytes, header_bytes);
    return SIF_ERR_IO;
  }
  const uint64_t data_bytes = file_bytes - header_bytes;

  /* Sized from the file only when it divides exactly: a remainder means the
   * format, the precision or the header length is not the file's, and a
   * count rounded down from it would read every particle wrong. */
  if (field->n_particles == 0) {
    if (data_bytes % record != 0 || data_bytes == 0) {
      SIF_LOG_ERROR("io",
        "%s holds %" PRIu64 " bytes after its header, not a whole number of "
        "%" PRIu64 "-byte particles for format '%s'; check the format, the "
        "precision and the header length",
        filepath, data_bytes, record, fmt);
      return SIF_ERR_IO;
    }
    field->n_particles = data_bytes / record;
  } else if (field->n_particles > data_bytes / record) {
    SIF_LOG_ERROR("io",
      "%s holds %" PRIu64 " particles of %" PRIu64 " bytes after its header, "
      "fewer than the %" PRIu64 " asked for",
      filepath, data_bytes / record, record, field->n_particles);
    return SIF_ERR_IO;
  } else if (field->n_particles * record < data_bytes) {
    SIF_LOG_WARNING("io",
      "field filled at %" PRIu64 " particles; %s holds %" PRIu64 " more bytes",
      field->n_particles, filepath, data_bytes - field->n_particles * record);
  }

  status = format_reserve(field, &f, fmt);
  if (status != SIF_OK)
    return status;

  const bool file_little =
    endian == SIF_BINARY_LITTLE ||
    (endian == SIF_BINARY_NATIVE && sif__io_host_is_little());
  const bool swap = file_little != (bool)sif__io_host_is_little();

  FILE* file = fopen(filepath, "rb");
  unsigned char* buf = malloc(record > BINARY_CHUNK ? record : BINARY_CHUNK);
  if (!file || !buf) {
    if (file)
      fclose(file);
    free(buf);
    SIF_LOG_ERROR("io", "failed to open %s", filepath);
    return file ? SIF_ERR_ALLOC : SIF_ERR_IO;
  }

  if (layout == SIF_BINARY_ROWS) {
    status = fseeko(file, (off_t)header_bytes, SEEK_SET) == 0
               ? binary_read_rows(file, field, &f, record, size, swap, buf)
               : SIF_ERR_IO;
  } else {
    status =
      binary_read_blocks(file, field, &f, (off_t)header_bytes, size, swap, buf);
  }

  free(buf);
  fclose(file);

  if (status != SIF_OK) {
    SIF_LOG_ERROR("io", "failed to read %s", filepath);
    return status;
  }

  format_loaded(field, &f);
  SIF_LOG_INFO(
    "io", "loaded %" PRIu64 " particles from %s", field->n_particles, filepath);
  return SIF_OK;
}

/* The allocating readers: an empty field, sized by the _into reader from the
 * file, and released again if the read fails. */

sif_field_t* sif_field_read_ascii(
  const char* filepath, const char* fmt, char delimiter, uint32_t skip_header) {
  sif_field_t* field = sif_field_alloc(0);
  if (!field) {
    SIF_LOG_ERROR("io", "failed to allocate a field for %s",
      filepath ? filepath : "(null)");
    return NULL;
  }
  if (sif_field_read_ascii_into(field, filepath, fmt, delimiter, skip_header) !=
      SIF_OK) {
    sif_field_free(field);
    return NULL;
  }
  return field;
}

sif_field_t* sif_field_read_binary(const char* filepath, const char* fmt,
  sif_binary_layout_t layout, sif_binary_precision_t precision,
  sif_binary_endian_t endian, uint64_t header_bytes) {
  sif_field_t* field = sif_field_alloc(0);
  if (!field) {
    SIF_LOG_ERROR("io", "failed to allocate a field for %s",
      filepath ? filepath : "(null)");
    return NULL;
  }
  if (sif_field_read_binary_into(field, filepath, fmt, layout, precision,
        endian, header_bytes) != SIF_OK) {
    sif_field_free(field);
    return NULL;
  }
  return field;
}

#undef MAX_COLS
#undef LINE_CAP
#undef MAX_SKIP_WIDTH
#undef BINARY_CHUNK
