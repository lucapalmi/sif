#include "sif/io/grid_io.h"

#include "sif/io/core_io.h"
#include "sif/utils/logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SIF_CHECK_WRITE(ptr, size, count, stream, msg)                         \
  do {                                                                         \
    if (fwrite((ptr), (size), (count), (stream)) != (count)) {                 \
      SIF_LOG_ERROR("xgrid", "failed to write %s", (msg));                     \
      fclose(file);                                                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define SIF_CHECK_PREAD(fd, dest, bytes, offset, msg, file_ptr)                \
  do {                                                                         \
    if (!sif_parallel_pread((fd), (dest), (bytes), (offset))) {                \
      SIF_LOG_ERROR("xgrid", "parallel read error for %s", (msg));             \
      fclose(file_ptr);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int sif_grid_write(const char* filepath, const sif_grid_t* grid) {
  if (!filepath || !grid) return 1;

  FILE* file = fopen(filepath, "wb");
  if (!file) {
    SIF_LOG_ERROR("xgrid", "Could not open %s for writing", filepath);
    return 1;
  }

  sif_xgrid_header_t header;
  memset(&header, 0, sizeof(sif_xgrid_header_t));

  strncpy(header.magic, __SIF_XGRID_MAGIC, 4);
  header.version = __SIF_XGRID_VERSION;
  header.n_cells = grid->n_cells;
  header.is_double = (sizeof(real_t) == 8) ? 1 : 0;
  header.total_cells = grid->total_cells;
  header.box_length = (double)grid->box_length;

  SIF_CHECK_WRITE(&header, sizeof(sif_xgrid_header_t), 1, file, "header");
  SIF_CHECK_WRITE(grid->delta, sizeof(real_t), grid->total_cells, file, "delta array");

  fclose(file);
  return 0;
}

int sif_grid_read_into(const char* filepath, sif_grid_t* grid) {
  if (!filepath || !grid) return 1;

  FILE* file = fopen(filepath, "rb");
  if (!file) return 1; /* Silent failure allows cache logic to smoothly fall back to compute */

  sif_xgrid_header_t header;
  if (fread(&header, sizeof(sif_xgrid_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xgrid", "Failed to read header from %s", filepath);
    fclose(file);
    return 1;
  }

  if (strncmp(header.magic, __SIF_XGRID_MAGIC, 4) != 0) {
    SIF_LOG_ERROR("xgrid", "File %s is not a valid xgrid binary", filepath);
    fclose(file);
    return 1;
  }

  uint32_t current_is_double = (sizeof(real_t) == 8) ? 1u : 0u;
  if (header.is_double != current_is_double) {
    SIF_LOG_ERROR("xgrid", "xgrid precision mismatch. File is %s, Library compiled as %s.",
                  header.is_double ? "FP64" : "FP32", current_is_double ? "FP64" : "FP32");
    fclose(file);
    return 1;
  }

  /* Cache-hit integrity check: make sure the file matches the target grid shape */
  if (header.n_cells != grid->n_cells || header.box_length != (double)grid->box_length) {
    SIF_LOG_WARNING("xgrid", "xgrid geometry mismatch. Cannot load into the target grid buffer.");
    fclose(file);
    return 1;
  }

  /* Transition to parallel POSIX I/O */
  int fd = fileno(file);
  fflush(file);

  size_t array_bytes = grid->total_cells * sizeof(real_t);
  off_t current_offset = sizeof(sif_xgrid_header_t);

  SIF_CHECK_PREAD(fd, grid->delta, array_bytes, current_offset, "delta array", file);

  fclose(file);
  return 0;
}

sif_grid_t* sif_grid_read(const char* filepath) {
  if (!filepath) return NULL;

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    SIF_LOG_ERROR("xgrid", "Could not open %s for reading", filepath);
    return NULL;
  }

  sif_xgrid_header_t header;
  if (fread(&header, sizeof(sif_xgrid_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xgrid", "Failed to read header from %s", filepath);
    fclose(file);
    return NULL;
  }
  fclose(file); /* Close immediately, _into will reopen and validate */

  sif_grid_t* grid = sif_grid_alloc(header.n_cells, (real_t)header.box_length);
  if (!grid) {
    SIF_LOG_ERROR("xgrid", "OOM allocating grid");
    return NULL;
  }

  if (sif_grid_read_into(filepath, grid) != 0) {
    SIF_LOG_ERROR("xgrid", "Failed to read grid data into struct");
    sif_grid_free(grid);
    return NULL;
  }

  SIF_LOG_INFO("xgrid", "Loaded %u^3 grid from %s", header.n_cells, filepath);
  return grid;
}