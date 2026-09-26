/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file gadget_internal.h
 * @brief The HDF5 half of the GADGET reader. Private.
 *
 * gadget_io.c reads the binary formats itself and hands HDF5 files to what is
 * declared here, implemented by gadget_hdf5.c in a build with HDF5 and by
 * refusals in hdf5_off.c otherwise -- the same split as the rest of the HDF5
 * layer, so gadget_io.c never includes hdf5.h.
 */

#ifndef SIF__IO_GADGET_INTERNAL_H
#define SIF__IO_GADGET_INTERNAL_H

#include "sif/io/gadget_io.h"

#include <stdint.h>

/** @brief The blocks the reader reads. */
typedef enum {
  SIF__GADGET_BLOCK_POS,
  SIF__GADGET_BLOCK_VEL,
  SIF__GADGET_BLOCK_MASS
} sif_gadget_block_t;

/** @brief An open HDF5 snapshot file. */
typedef struct sif_gadget_h5_file sif_gadget_h5_file_t;

/**
 * @brief Open an HDF5 snapshot file and read its header.
 *
 * @param path The file.
 * @param out_file The open file, released with sif__gadget_h5_close().
 * @param out_header Filled from the file's `Header` group, and from
 * `Parameters` and `Units` for what GADGET-4 keeps there.
 * @return SIF_OK; SIF_ERR_IO for a file that is not an HDF5 snapshot;
 * SIF_ERR_ALLOC; SIF_ERR_UNSUPPORTED in a build without HDF5.
 */
int sif__gadget_h5_open(const char* path, sif_gadget_h5_file_t** out_file,
  sif_gadget_header_t* out_header);

/**
 * @brief Read @p count particles of one type from one block, starting at
 * @p start within that type, converted to double.
 *
 * @param out 3 * @p count values, interleaved per particle, for POS and VEL;
 * @p count for MASS.
 * @return SIF_OK, or SIF_ERR_IO for a missing or wrongly shaped dataset.
 */
int sif__gadget_h5_read(sif_gadget_h5_file_t* file, uint32_t ptype,
  sif_gadget_block_t block, uint64_t start, uint64_t count, double* out);

/** @brief Close a file. NULL is accepted. */
void sif__gadget_h5_close(sif_gadget_h5_file_t* file);

#endif /* SIF__IO_GADGET_INTERNAL_H */
