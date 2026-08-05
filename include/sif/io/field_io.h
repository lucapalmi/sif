#ifndef __SIF_FIELD_IO_H__
#define __SIF_FIELD_IO_H__

#include "sif/core/macros.h"
#include "sif/structures/field.h"

#include <stdint.h>

#define __SIF_XFIELD_MAGIC "XFLD"
#define __SIF_XFIELD_VERSION 1

/*
 * @brief 64-byte rigid header for the .xfield binary format.
 */
typedef struct {
  char magic[4];             /* "XFLD" */
  uint32_t version;          /* Format version */
  uint64_t n_particles;      /* Total particle count */
  double box_length;         /* Simulation box size (fixed double for ABI safety) */
  uint32_t has_masses;       /* 1 if mass block exists, 0 otherwise */
  uint32_t has_velocities;   /* 1 if vx, vy, vz blocks exist, 0 otherwise */
  uint32_t is_double;        /* 1 if real_t is 64-bit, 0 if 32-bit */
  char padding[28];          /* Reserved space to maintain exactly 64 bytes */
} sif_xfield_header_t;

/*
 * @brief Reads an optimized .xfield binary directly into cache-aligned memory.
 * * @param filepath Path to the input .xfield file
 * @param out_box_length Optional pointer to store the retrieved box length
 * @return Allocated and populated sif_field_t (NULL on failure)
 */
NODISCARD sif_field_t* sif_field_read(const char* filepath, double* out_box_length);

/*
 * @brief Reads an .xfield binary directly into an existing, pre-allocated field.
 *
 * @param filepath Path to the input .xfield file
 * @param field Pre-allocated field to read into (must have correct capacity)
 * @return 0 on success, non-zero on failure
 */
int sif_field_read_into(const char* filepath, sif_field_t* field);

/*
 * @brief Dumps an sif_field_t to disk in the .xfield format.
 * * @param filepath Path for the output .xfield file
 * @param field The field to write
 * @param box_length The physical box length to store in the header
 * @return 0 on success, non-zero on failure
 */
int sif_field_write(const char* filepath, const sif_field_t* field, double box_length);

/*
 * @brief Reads an ASCII field file, extracting specified columns into the field.
 *
 * @param field The pre-allocated field to populate
 * @param filepath The path to the ASCII file
 * @param fmt The format string defining the columns (e.g., "x,y,z,vx,vy,vz,m")
 * @param delimiter The character separating the columns
 * @param skip_header The number of lines to skip before reading data
 *
 * @return 0 on success, non-zero on failure
 */
int sif_field_read_ascii(sif_field_t* field, const char* filepath,
  const char* fmt, char delimiter, uint32_t skip_header);

#endif /* __SIF_FIELD_IO_H__ */