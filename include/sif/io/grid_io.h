#ifndef __SIF_GRID_IO_H__
#define __SIF_GRID_IO_H__

#include "sif/core/macros.h"
#include "sif/structures/grid.h"

#include <stdint.h>

#define __SIF_XGRID_MAGIC "XGRD"
#define __SIF_XGRID_VERSION 1

/*
 * @brief 64-byte rigid header for the .xgrid binary format.
 */
typedef struct {
  char magic[4];             /* "XGRD" */
  uint32_t version;          /* Format version */
  uint32_t n_cells;          /* Number of cells per side */
  uint32_t is_double;        /* 1 if real_t is 64-bit, 0 if 32-bit */
  uint64_t total_cells;      /* Total cells (n_cells^3) */
  double box_length;         /* Simulation box size (fixed double for ABI safety) */
  char padding[32];          /* Reserved space to maintain exactly 64 bytes */
} sif_xgrid_header_t;

/*
 * @brief Allocates and reads an .xgrid binary into memory.
 *
 * @param filepath Path to the input .xgrid file
 * @return Allocated and populated sif_grid_t (NULL on failure)
 */
NODISCARD sif_grid_t* sif_grid_read(const char* filepath);

/*
 * @brief Reads an .xgrid binary directly into an existing, pre-allocated grid.
 *
 * @param filepath Path to the input .xgrid file
 * @param grid Pre-allocated grid to read into
 * @return 0 on success, non-zero on failure
 */
int sif_grid_read_into(const char* filepath, sif_grid_t* grid);

/*
 * @brief Dumps an sif_grid_t to disk in the .xgrid format.
 *
 * @param filepath Path for the output .xgrid file
 * @param grid The grid to write
 * @return 0 on success, non-zero on failure
 */
int sif_grid_write(const char* filepath, const sif_grid_t* grid);

#endif /* __SIF_GRID_IO_H__ */