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
#include "sif/utils/str.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int sif_field_read_into(const char* filepath, sif_field_t* field) {
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

  if (sif_field_read_into(filepath, field) != SIF_OK) {
    sif_field_free(field);
    return NULL;
  }

  SIF_LOG_INFO("xfield", "loaded %" PRIu64 " particles from %s",
    header.n_particles, filepath);
  return field;
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

int sif_field_read_ascii(sif_field_t* field, const char* filepath,
  const char* fmt, char delimiter, uint32_t skip_header) {

  if (!field || !filepath || !fmt) {
    SIF_LOG_ERROR("io", "invalid field, path or format string");
    return SIF_ERR_INVALID;
  }

  sif_col_target_t targets[MAX_COLS];
  const int n_cols = sif_str_decode_format(fmt, targets, MAX_COLS);

  /* decode_format skips characters it does not recognize, so a typo'd format
   * yields a short layout rather than an error. Zero columns is where that
   * stops being recoverable: the parse would consume every line and write
   * nothing, leaving a field of uninitialized memory that reads as loaded. */
  if (n_cols <= 0) {
    SIF_LOG_ERROR("io", "format '%s' describes no usable columns", fmt);
    return SIF_ERR_INVALID;
  }

  bool has_x = false, has_y = false, has_z = false;
  bool has_vx = false, has_vy = false, has_vz = false;
  bool has_mass = false;
  for (int i = 0; i < n_cols; i++) {
    switch (targets[i]) {
    case SIF_COL_X:
      has_x = true;
      break;
    case SIF_COL_Y:
      has_y = true;
      break;
    case SIF_COL_Z:
      has_z = true;
      break;
    case SIF_COL_VX:
      has_vx = true;
      break;
    case SIF_COL_VY:
      has_vy = true;
      break;
    case SIF_COL_VZ:
      has_vz = true;
      break;
    case SIF_COL_M:
      has_mass = true;
      break;
    case SIF_COL_IGNORE:
      break;
    }
  }

  /* Position and velocity are three-component quantities and are reserved as
   * one block each, so naming two of the three would allocate the third and
   * never write it. Demand all three or none. */
  const bool wants_position = has_x || has_y || has_z;
  const bool wants_velocity = has_vx || has_vy || has_vz;

  if (wants_position && !(has_x && has_y && has_z)) {
    SIF_LOG_ERROR("io", "format '%s' names only part of the position", fmt);
    return SIF_ERR_INVALID;
  }
  if (wants_velocity && !(has_vx && has_vy && has_vz)) {
    SIF_LOG_ERROR("io", "format '%s' names only part of the velocity", fmt);
    return SIF_ERR_INVALID;
  }

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

  /* Reserving is a no-op when a block is already there, so re-reading into a
   * populated field is safe and is how a weight-only format is meant to be
   * used: it keeps the positions that are already loaded. */
  int status;
  if (wants_position) {
    status = sif_field_reserve_positions(field);
    if (status != SIF_OK)
      return status;
  } else if (!field->x) {
    SIF_LOG_ERROR(
      "io", "format '%s' loads no positions and the field has none", fmt);
    return SIF_ERR_INVALID;
  }

  if (wants_velocity) {
    status = sif_field_reserve_velocities(field);
    if (status != SIF_OK)
      return status;
  }

  if (has_mass) {
    status = sif_field_reserve_weights(field);
    if (status != SIF_OK)
      return status;
  }

  FILE* f = fopen(filepath, "r");
  if (!f) {
    SIF_LOG_ERROR("io", "failed to open %s", filepath);
    return SIF_ERR_IO;
  }

  char line[LINE_CAP];
  for (uint32_t i = 0; i < skip_header; i++) {
    if (!fgets(line, sizeof(line), f))
      break;
  }

  uint64_t loaded = 0;
  uint64_t skipped = 0;

  while (loaded < field->n_particles && fgets(line, sizeof(line), f)) {
    if (!line_is_data(line)) {
      skipped++;
      continue;
    }

    char* cursor = line;
    sif_real val = 0.0;
    int col = 0;

    for (; col < n_cols; col++) {
      if (!sif_str_extract_next_real(&cursor, delimiter, &val))
        break;

      switch (targets[col]) {
      case SIF_COL_X:
        field->x[loaded] = val;
        break;
      case SIF_COL_Y:
        field->y[loaded] = val;
        break;
      case SIF_COL_Z:
        field->z[loaded] = val;
        break;
      case SIF_COL_VX:
        field->vx[loaded] = val;
        break;
      case SIF_COL_VY:
        field->vy[loaded] = val;
        break;
      case SIF_COL_VZ:
        field->vz[loaded] = val;
        break;
      case SIF_COL_M:
        field->weights[loaded] = val;
        break;
      case SIF_COL_IGNORE:
        break;
      }
    }

    /* A row that ran out of columns is not a particle. Not counting it is what
     * makes it harmless: whatever was written at this index is overwritten by
     * the next good row, or falls outside n_particles once the count below is
     * corrected. Advancing instead would leave an entry whose remaining
     * components were never assigned. */
    if (col < n_cols) {
      skipped++;
      continue;
    }

    loaded++;
  }

  /* Whether the file had more to give is one more read, not another pass:
   * counting the rows again means streaming the whole file a second time. */
  bool has_more = false;
  while (!has_more && fgets(line, sizeof(line), f)) {
    has_more = line_is_data(line);
  }

  fclose(f);

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

  /* The positions are new, so nothing derived from the old ones still holds. */
  field->state_flags &= ~SIF_FIELD_STATE_BOUNDS_VALID;
  field->state_flags &= ~SIF_FIELD_STATE_MORTON_SORTED;

  return SIF_OK;
}

#undef MAX_COLS
#undef LINE_CAP
