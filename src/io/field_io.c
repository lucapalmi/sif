#include "sif/io/field_io.h"

#include "sif/io/core_io.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/stringy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Safe macro wrappers to prevent dangling if-statements */
#define SIF_CHECK_WRITE(ptr, size, count, stream, msg)                         \
  do {                                                                         \
    if (fwrite((ptr), (size), (count), (stream)) != (count)) {                 \
      SIF_LOG_ERROR("xfield", "failed to write %s", (msg));                    \
      fclose(file);                                                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define SIF_CHECK_PREAD(fd, dest, bytes, offset, msg, file_ptr)                \
  do {                                                                         \
    if (!sif_parallel_pread((fd), (dest), (bytes), (offset))) {                \
      SIF_LOG_ERROR("xfield", "parallel read error for %s", (msg));            \
      fclose(file_ptr);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int sif_field_write(
  const char* filepath, const sif_field_t* field, double box_length) {
  if (!filepath || !field) {
    SIF_LOG_ERROR("xfield", "invalid arguments");
    return 1;
  }

  FILE* file = fopen(filepath, "wb");
  if (!file) {
    SIF_LOG_ERROR("xfield", "failed to open %s", filepath);
    return 1;
  }

  sif_xfield_header_t header;
  memset(&header, 0, sizeof(sif_xfield_header_t));

  strncpy(header.magic, __SIF_XFIELD_MAGIC, 4);
  header.version = __SIF_XFIELD_VERSION;
  header.n_particles = field->n_particles;
  header.box_length = box_length;
  header.has_masses = (field->masses != NULL) ? 1 : 0;
  header.has_velocities = (field->vx != NULL) ? 1 : 0;
  header.is_double = (sizeof(real_t) == 8) ? 1 : 0;

  SIF_CHECK_WRITE(&header, sizeof(sif_xfield_header_t), 1, file, "header");

  uint64_t n = field->n_particles;

  SIF_CHECK_WRITE(field->x, sizeof(real_t), n, file, "x coordinates");
  SIF_CHECK_WRITE(field->y, sizeof(real_t), n, file, "y coordinates");
  SIF_CHECK_WRITE(field->z, sizeof(real_t), n, file, "z coordinates");

  if (header.has_velocities) {
    SIF_CHECK_WRITE(field->vx, sizeof(real_t), n, file, "vx velocities");
    SIF_CHECK_WRITE(field->vy, sizeof(real_t), n, file, "vy velocities");
    SIF_CHECK_WRITE(field->vz, sizeof(real_t), n, file, "vz velocities");
  }

  if (header.has_masses) {
    SIF_CHECK_WRITE(field->masses, sizeof(real_t), n, file, "masses");
  }

  fclose(file);
  SIF_LOG_INFO("xfield", "Successfully wrote %lu particles to %s", n, filepath);
  return 0;
}

int sif_field_read_into(const char* filepath, sif_field_t* field) {
  if (!filepath || !field) return 1;

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    SIF_LOG_ERROR("xfield", "Could not open %s for reading", filepath);
    return 1;
  }

  sif_xfield_header_t header;
  if (fread(&header, sizeof(sif_xfield_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xfield", "Failed to read header from %s", filepath);
    fclose(file);
    return 1;
  }

  if (strncmp(header.magic, __SIF_XFIELD_MAGIC, 4) != 0) {
    SIF_LOG_ERROR("xfield", "File %s is not a valid xfield binary", filepath);
    fclose(file);
    return 1;
  }

  uint32_t current_is_double = (sizeof(real_t) == 8) ? 1u : 0u;
  if (header.is_double != current_is_double) {
    SIF_LOG_ERROR("xfield", "Precision mismatch.");
    fclose(file);
    return 1;
  }

  if (header.n_particles != field->n_particles) {
    SIF_LOG_ERROR("xfield", "Geometry mismatch. File has %lu particles, target field has %lu",
                  header.n_particles, field->n_particles);
    fclose(file);
    return 1;
  }

  if (header.has_velocities && !field->vx) {
    SIF_LOG_ERROR("xfield", "Target field lacks velocity allocation");
    fclose(file);
    return 1;
  }

  if (header.has_masses && !field->masses) {
    SIF_LOG_ERROR("xfield", "Target field lacks mass allocation");
    fclose(file);
    return 1;
  }

  /* Transition to parallel POSIX I/O */
  int fd = fileno(file);
  fflush(file);

  uint64_t n = header.n_particles;
  size_t array_bytes = n * sizeof(real_t);
  off_t current_offset = sizeof(sif_xfield_header_t);

  SIF_CHECK_PREAD(fd, field->x, array_bytes, current_offset, "x coordinates", file);
  current_offset += array_bytes;

  SIF_CHECK_PREAD(fd, field->y, array_bytes, current_offset, "y coordinates", file);
  current_offset += array_bytes;

  SIF_CHECK_PREAD(fd, field->z, array_bytes, current_offset, "z coordinates", file);
  current_offset += array_bytes;

  if (header.has_velocities) {
    SIF_CHECK_PREAD(fd, field->vx, array_bytes, current_offset, "vx velocities", file);
    current_offset += array_bytes;

    SIF_CHECK_PREAD(fd, field->vy, array_bytes, current_offset, "vy velocities", file);
    current_offset += array_bytes;

    SIF_CHECK_PREAD(fd, field->vz, array_bytes, current_offset, "vz velocities", file);
    current_offset += array_bytes;
  }

  if (header.has_masses) {
    SIF_CHECK_PREAD(fd, field->masses, array_bytes, current_offset, "masses", file);
  }

  fclose(file);
  return 0;
}

sif_field_t* sif_field_read(const char* filepath, double* out_box_length) {
  if (!filepath) return NULL;

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

  if (out_box_length) *out_box_length = header.box_length;

  sif_field_t* field = sif_field_alloc(header.n_particles);
  if (!field) return NULL;

  uint64_t align_elements = __SIF_CACHE_LINE / sizeof(real_t);
  uint64_t padded_n = (header.n_particles + align_elements - 1) & ~(align_elements - 1);

  field->_position_block = sif_malloc_aligned(3 * padded_n * sizeof(real_t));
  if (!field->_position_block) {
    SIF_LOG_ERROR("xfield", "OOM allocating position block");
    sif_field_free(field);
    return NULL;
  }

  field->x = field->_position_block;
  field->y = field->_position_block + padded_n;
  field->z = field->_position_block + (2 * padded_n);
  field->state_flags |= __FIELD_STATE_OWNS_POSITIONS;

  if (header.has_velocities) {
    field->_velocity_block = sif_malloc_aligned(3 * padded_n * sizeof(real_t));
    if (!field->_velocity_block) {
      SIF_LOG_ERROR("xfield", "OOM allocating velocity block");
      sif_field_free(field);
      return NULL;
    }

    field->vx = field->_velocity_block;
    field->vy = field->_velocity_block + padded_n;
    field->vz = field->_velocity_block + (2 * padded_n);
    field->state_flags |= __FIELD_STATE_OWNS_VELOCITIES;
  }

  if (header.has_masses) {
    field->masses = sif_malloc_aligned(header.n_particles * sizeof(real_t));
    if (!field->masses) {
      SIF_LOG_ERROR("xfield", "OOM allocating masses block");
      sif_field_free(field);
      return NULL;
    }
    field->state_flags |= __FIELD_STATE_OWNS_MASSES;
  }

  if (sif_field_read_into(filepath, field) != 0) {
    SIF_LOG_ERROR("xfield", "Failed to read field data into structs");
    sif_field_free(field);
    return NULL;
  }

  SIF_LOG_INFO("xfield", "Loaded %lu particles from %s", header.n_particles, filepath);
  return field;
}

int sif_field_read_ascii(sif_field_t* field, const char* filepath,
  const char* fmt, char delimiter, uint32_t skip_header) {
  
  if (!field || !filepath || !fmt) {
    SIF_LOG_ERROR("io", "invalid field, path or format string");
    return 1;
  }

  sif_col_target_t targets[32];
  int n_cols = sif_str_decode_format(fmt, targets, 32);

  uint8_t requires_mass = 0;
  uint8_t requires_velocity = 0;
  for (int i = 0; i < n_cols; i++) {
    if (targets[i] == SIF_COL_M) requires_mass = 1;
    if (targets[i] == SIF_COL_VX || targets[i] == SIF_COL_VY || targets[i] == SIF_COL_VZ)
      requires_velocity = 1;
  }

  if (field->n_particles == 0) {
    field->n_particles = sif_count_ascii_rows(filepath, skip_header);
    if (field->n_particles == 0) {
      SIF_LOG_ERROR("io", "failed to read particles from file");
      return 1;
    }
    SIF_LOG_INFO("io", "found %llu particles in ASCII file",
      (unsigned long long)field->n_particles);
  }

  /* Ensure the field arrays are allocated */
  if (!(field->state_flags & __FIELD_STATE_OWNS_POSITIONS)) {
    uint64_t align_elements = __SIF_CACHE_LINE / sizeof(real_t);
    uint64_t padded_n = (field->n_particles + align_elements - 1) & ~(align_elements - 1);

    field->_position_block = sif_malloc_aligned(3 * padded_n * sizeof(real_t));
    field->x = field->_position_block;
    field->y = field->_position_block + padded_n;
    field->z = field->_position_block + (2 * padded_n);
    field->state_flags |= __FIELD_STATE_OWNS_POSITIONS;
  }

  if (requires_velocity && !(field->state_flags & __FIELD_STATE_OWNS_VELOCITIES)) {
    uint64_t align_elements = __SIF_CACHE_LINE / sizeof(real_t);
    uint64_t padded_n = (field->n_particles + align_elements - 1) & ~(align_elements - 1);

    field->_velocity_block = sif_malloc_aligned(3 * padded_n * sizeof(real_t));
    field->vx = field->_velocity_block;
    field->vy = field->_velocity_block + padded_n;
    field->vz = field->_velocity_block + (2 * padded_n);
    field->state_flags |= __FIELD_STATE_OWNS_VELOCITIES;
  }

  if (requires_mass && !(field->state_flags & __FIELD_STATE_OWNS_MASSES)) {
    field->masses = sif_malloc_aligned(field->n_particles * sizeof(real_t));
    field->state_flags |= __FIELD_STATE_OWNS_MASSES;
  }

  FILE* f = fopen(filepath, "r");
  if (!f) {
    SIF_LOG_ERROR("io", "failed to open %s", filepath);
    return 1;
  }

  char line[2048];
  for (uint32_t i = 0; i < skip_header; i++) {
    if (!fgets(line, sizeof(line), f)) break;
  }

  uint64_t loaded = 0;
  while (loaded < field->n_particles && fgets(line, sizeof(line), f)) {
    char* cursor = line;
    real_t val = 0.0;

    for (int col = 0; col < n_cols; col++) {
      if (!sif_str_extract_next_real(&cursor, delimiter, &val)) break;

      switch (targets[col]) {
      case SIF_COL_X: field->x[loaded] = val; break;
      case SIF_COL_Y: field->y[loaded] = val; break;
      case SIF_COL_Z: field->z[loaded] = val; break;
      case SIF_COL_VX: if (field->vx) field->vx[loaded] = val; break;
      case SIF_COL_VY: if (field->vy) field->vy[loaded] = val; break;
      case SIF_COL_VZ: if (field->vz) field->vz[loaded] = val; break;
      case SIF_COL_M: if (field->masses) field->masses[loaded] = val; break;
      case SIF_COL_IGNORE: break;
      }
    }
    loaded++;
  }
  fclose(f);

  if (loaded < field->n_particles) {
    SIF_LOG_ERROR("io", "truncating field from %llu to %llu particles",
      field->n_particles, loaded);
    field->n_particles = loaded;
  } else if (sif_count_ascii_rows(filepath, skip_header) > loaded) {
    SIF_LOG_ERROR("io", "field is not large enough to contain the whole file (loaded %llu particles)", loaded);
  }

  /* Loading new data invalidates bounds and sorting */
  field->state_flags &= ~__FIELD_STATE_BOUNDS_VALID;
  field->state_flags &= ~__FIELD_STATE_MORTON_SORTED;

  return 0;
}