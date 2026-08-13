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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Safe macro wrappers to prevent dangling if-statements */
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

int sif_field_write(
  const char* filepath, const sif_field_t* field, double box_length) {
  if (!filepath || !field) {
    SIF_LOG_ERROR("xfield", "invalid arguments");
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
  header.has_masses = (field->masses != NULL) ? 1 : 0;
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
  if (header.has_masses)
    crc = sif_crc32_update(crc, field->masses, array_bytes);
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

  if (header.has_masses) {
    CHECK_WRITE(field->masses, sizeof(sif_real), n, file, "masses");
  }

  fclose(file);
  SIF_LOG_INFO("xfield", "Successfully wrote %lu particles to %s", n, filepath);
  return SIF_OK;
}

int sif_field_read_into(const char* filepath, sif_field_t* field) {
  if (!filepath || !field)
    return SIF_ERR_INVALID;

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    SIF_LOG_ERROR("xfield", "Could not open %s for reading", filepath);
    return SIF_ERR_IO;
  }

  sif_xfield_header_t header;
  if (fread(&header, sizeof(sif_xfield_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xfield", "Failed to read header from %s", filepath);
    fclose(file);
    return SIF_ERR_IO;
  }

  if (strncmp(header.magic, SIF_XFIELD_MAGIC, 4) != 0) {
    SIF_LOG_ERROR("xfield", "File %s is not a valid xfield binary", filepath);
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header.version > SIF_XFIELD_VERSION) {
    SIF_LOG_ERROR("xfield",
      "%s is version %u, but this build reads up to version %u", filepath,
      header.version, (unsigned)SIF_XFIELD_VERSION);
    fclose(file);
    return SIF_ERR_IO;
  }

  uint32_t current_is_double = (sizeof(sif_real) == 8) ? 1u : 0u;
  if (header.is_double != current_is_double) {
    SIF_LOG_ERROR("xfield", "Precision mismatch.");
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header.n_particles != field->n_particles) {
    SIF_LOG_ERROR("xfield",
      "Geometry mismatch. File has %lu particles, target field has %lu",
      header.n_particles, field->n_particles);
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header.has_velocities && !field->vx) {
    SIF_LOG_ERROR("xfield", "Target field lacks velocity allocation");
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header.has_masses && !field->masses) {
    SIF_LOG_ERROR("xfield", "Target field lacks mass allocation");
    fclose(file);
    return SIF_ERR_IO;
  }

  /* Transition to parallel POSIX I/O */
  int fd = fileno(file);
  fflush(file);

  uint64_t n = header.n_particles;
  size_t array_bytes = n * sizeof(sif_real);
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

  if (header.has_masses) {
    CHECK_PREAD(fd, field->masses, array_bytes, current_offset, "masses", file);
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
    if (header.has_masses)
      crc = sif_crc32_update(crc, field->masses, array_bytes);

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
    SIF_LOG_ERROR("xfield", "Could not open %s for reading", filepath);
    return NULL;
  }

  sif_xfield_header_t header;
  if (fread(&header, sizeof(sif_xfield_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xfield", "Failed to read header from %s", filepath);
    fclose(file);
    return NULL;
  }
  fclose(file); /* Close immediately; _into handles the heavy lifting */

  if (out_box_length)
    *out_box_length = header.box_length;

  sif_field_t* field = sif_field_alloc(header.n_particles);
  if (!field)
    return NULL;

  /* Reserve, then read straight into the field's own arrays: the file never
   * goes through an intermediate copy. */
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

  if (header.has_masses && sif_field_reserve_masses(field) != SIF_OK) {
    SIF_LOG_ERROR("xfield", "OOM allocating masses block");
    sif_field_free(field);
    return NULL;
  }

  if (sif_field_read_into(filepath, field) != 0) {
    SIF_LOG_ERROR("xfield", "Failed to read field data into structs");
    sif_field_free(field);
    return NULL;
  }

  SIF_LOG_INFO(
    "xfield", "Loaded %lu particles from %s", header.n_particles, filepath);
  return field;
}

int sif_field_read_ascii(sif_field_t* field, const char* filepath,
  const char* fmt, char delimiter, uint32_t skip_header) {

  if (!field || !filepath || !fmt) {
    SIF_LOG_ERROR("io", "invalid field, path or format string");
    return SIF_ERR_IO;
  }

  sif_col_target_t targets[32];
  int n_cols = sif_str_decode_format(fmt, targets, 32);

  uint8_t requires_mass = 0;
  uint8_t requires_velocity = 0;
  for (int i = 0; i < n_cols; i++) {
    if (targets[i] == SIF_COL_M)
      requires_mass = 1;
    if (targets[i] == SIF_COL_VX || targets[i] == SIF_COL_VY ||
        targets[i] == SIF_COL_VZ)
      requires_velocity = 1;
  }

  if (field->n_particles == 0) {
    field->n_particles = sif__io_ascii_row_count(filepath, skip_header);
    if (field->n_particles == 0) {
      SIF_LOG_ERROR("io", "failed to read particles from file");
      return SIF_ERR_IO;
    }
    SIF_LOG_INFO("io", "found %llu particles in ASCII file",
      (unsigned long long)field->n_particles);
  }

  /* Ensure the field arrays are allocated. Reserving is a no-op when they
   * already are, so re-reading into a populated field is safe. */
  if (sif_field_reserve_positions(field) != SIF_OK)
    return SIF_ERR_IO;

  if (requires_velocity && sif_field_reserve_velocities(field) != SIF_OK)
    return SIF_ERR_IO;

  if (requires_mass && sif_field_reserve_masses(field) != SIF_OK)
    return SIF_ERR_IO;

  FILE* f = fopen(filepath, "r");
  if (!f) {
    SIF_LOG_ERROR("io", "failed to open %s", filepath);
    return SIF_ERR_IO;
  }

  char line[2048];
  for (uint32_t i = 0; i < skip_header; i++) {
    if (!fgets(line, sizeof(line), f))
      break;
  }

  uint64_t loaded = 0;
  while (loaded < field->n_particles && fgets(line, sizeof(line), f)) {
    char* cursor = line;
    sif_real val = 0.0;

    for (int col = 0; col < n_cols; col++) {
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
        if (field->vx)
          field->vx[loaded] = val;
        break;
      case SIF_COL_VY:
        if (field->vy)
          field->vy[loaded] = val;
        break;
      case SIF_COL_VZ:
        if (field->vz)
          field->vz[loaded] = val;
        break;
      case SIF_COL_M:
        if (field->masses)
          field->masses[loaded] = val;
        break;
      case SIF_COL_IGNORE:
        break;
      }
    }
    loaded++;
  }
  fclose(f);

  if (loaded < field->n_particles) {
    SIF_LOG_ERROR("io", "truncating field from %llu to %llu particles",
      field->n_particles, loaded);
    field->n_particles = loaded;
  } else if (sif__io_ascii_row_count(filepath, skip_header) > loaded) {
    SIF_LOG_ERROR("io",
      "field is not large enough to contain the whole file (loaded %llu "
      "particles)",
      loaded);
  }

  /* Loading new data invalidates bounds and sorting */
  field->state_flags &= ~SIF_FIELD_STATE_BOUNDS_VALID;
  field->state_flags &= ~SIF_FIELD_STATE_MORTON_SORTED;

  return SIF_OK;
}