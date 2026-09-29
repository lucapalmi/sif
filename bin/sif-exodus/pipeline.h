/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* The exodus pipeline in a periodic box, from a particle file to a void
 * catalogue on disk:
 *
 *   field    read the positions (and weights, if asked for), fold them into
 *            the box
 *   grid     CIC-deposit them, turn the grid into a density contrast
 *   mesh     bin them into a chain mesh, which takes the field's storage over
 *   finder   sif_finder_exodus() over the grid and the mesh
 *   catalog  write it, with what made it
 *
 * A survey, with randoms in place of the box mean, will be a pipeline of its
 * own beside this one (pipeline_survey()), chosen between in main.c.
 *
 * Everything the run needs is in exodus_params_t, which says nothing about
 * where it came from: the configuration file fills it, the pipeline only
 * reads it. The library has to be initialized before pipeline_box(); the
 * thread count, log level and FFT tuning are the caller's to apply, since
 * they are sif_init()'s arguments and not the pipeline's.
 */

#ifndef SIF_EXODUS_PIPELINE_H
#define SIF_EXODUS_PIPELINE_H

#include "sif/core/macros.h"
#include "sif/io/field_io.h"
#include "sif/io/gadget_io.h"

#include <stdbool.h>
#include <stdint.h>

/* The kind of particle file. */
typedef enum {
  /* sif's own .xfield: box and weights come from the file. */
  EXODUS_INPUT_XFIELD,
  /* A text table, laid out by `columns`. */
  EXODUS_INPUT_ASCII,
  /* A raw binary file, laid out by `columns` and the `binary` block. */
  EXODUS_INPUT_BINARY,
  /* A GADGET snapshot, binary or HDF5, one particle type. */
  EXODUS_INPUT_GADGET
} exodus_input_kind_t;

/* What the radii are measured in. */
typedef enum {
  /* The units of the box, which are the catalogue's. */
  EXODUS_RADII_PHYSICAL,
  /* Mean tracer separations, box / N^(1/3): scaled to the box's units once
   * the field has been read, and recorded that way in the catalogue. */
  EXODUS_RADII_MPS
} exodus_radii_units_t;

/* The format the catalogue is written in. */
typedef enum {
  /* /catalog in an HDF5 file, with the run's parameters as attributes. In a
   * build without HDF5 the library writes text next to the path instead. */
  EXODUS_OUTPUT_HDF5,
  /* sif_catalog_write_ascii(): the voids only, no parameters. */
  EXODUS_OUTPUT_ASCII
} exodus_output_kind_t;

typedef struct {
  /* --- input ---------------------------------------------------------- */
  struct {
    exodus_input_kind_t kind;
    const char* path;

    /* Side of the periodic box. 0 takes it from the file, which only an
     * .xfield or a GADGET snapshot records; a text or raw binary input has to
     * give it. Given for a file that records one, it wins, with a warning if
     * the two disagree. */
    double box_length;

    /* ASCII and binary: the column format of sif_field_read_ascii() --
     * "x y z", "* x y z w", ... Positions, and a weight if the run is to be
     * weighted; velocity columns are refused, since the finder never reads
     * them. An .xfield is read whole: it is sif's own, and carries what it
     * was written with. */
    const char* columns;

    /* ASCII only. */
    char delimiter;
    uint32_t skip_header;

    /* Binary only. */
    struct {
      sif_binary_layout_t layout;
      sif_binary_precision_t precision;
      sif_binary_endian_t endian;
      uint64_t header_bytes;
    } binary;

    /* GADGET only. Velocities are never read. */
    struct {
      sif_gadget_format_t format;
      sif_gadget_ptype_t ptype;
      sif_gadget_length_t length;
      /* Read particle masses into the weights. */
      bool masses;
      /* Share of the particles to keep, in (0, 1], and the subsample's
       * seed. */
      double fraction;
      uint64_t seed;
    } gadget;
  } input;

  /* --- density grid --------------------------------------------------- */
  struct {
    /* Cells per side; 0 takes one cell per mean tracer separation,
     * N^(1/3), rounded up to a size the FFT is fast at. */
    uint32_t n_cells;
  } grid;

  /* --- chain mesh ----------------------------------------------------- */
  struct {
    /* Cells per side; 0 takes sif_finder_suggest_mesh_cells(). Only speed
     * and memory depend on it, never the catalogue. */
    uint32_t n_cells;
  } mesh;

  /* --- finder --------------------------------------------------------- */
  struct {
    /* The radius ladder, in any order, in `radii_units`. */
    const sif_real* radii;
    uint32_t n_radii;
    exodus_radii_units_t radii_units;
    /* Density contrast a void is grown to; negative. */
    double threshold;
    /* Allowed overlap, as a fraction of the smaller void's radius. */
    double overlap_fraction;
    /* How far past a rung the rescaling looks, as a multiple of it: 1.25,
     * 1.5, 1.75 or 2.0 (the SIF_FINDER_SEARCH_* family). */
    double search_factor;
  } finder;

  /* --- output --------------------------------------------------------- */
  /* Written beside the path and moved into place when complete, replacing
   * whatever was there. */
  struct {
    exodus_output_kind_t kind;
    const char* path;
  } output;
} exodus_params_t;

/* Cells per side of the default grid for n tracers: one per mean
 * separation, rounded up to a size the FFT is fast at. */
uint32_t pipeline_default_grid_cells(uint64_t n_tracers);

/* About the most memory a box run holds at once, in bytes, from the sizes
 * alone. */
uint64_t pipeline_peak_bytes(
  uint64_t n_tracers, bool weighted, uint32_t grid_cells, uint32_t mesh_cells);

/* Run the pipeline on a periodic box. Returns SIF_OK, or the status of the
 * step that failed, which has already been logged. */
SIF_NODISCARD int pipeline_box(const exodus_params_t* params);

#endif /* SIF_EXODUS_PIPELINE_H */
