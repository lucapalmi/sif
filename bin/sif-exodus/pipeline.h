/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* The exodus pipelines, from particle files to a void catalogue on disk.
 *
 * In a periodic box, pipeline_box():
 *
 *   field    read the positions (and weights, if asked for), fold them into
 *            the box
 *   grid     CIC-deposit them, turn the grid into a density contrast
 *   mesh     bin them into a chain mesh, which takes the field's storage over
 *   finder   sif_finder_exodus() over the grid and the mesh
 *   catalogue  write it, with what made it
 *
 * On a survey, pipeline_survey(), with randoms in place of the box mean:
 *
 *   fields   read the data and the randoms; take both off the sky with the
 *            cosmology, if they are on it
 *   box      measure the footprint off the randoms, choose the padded box
 *            the survey is searched in, move both into it
 *   grids    CIC-deposit each, left as densities
 *   meshes   one each, sized for the density inside the footprint
 *   finder   sif_finder_exodus_survey()
 *   catalogue  move the voids back out of the box -- and onto the sky, if
 *            asked -- and write it, with what made it
 *
 * main.c chooses between them by exodus_params_t::mode.
 *
 * Everything the run needs is in exodus_params_t, which says nothing about
 * where it came from: the configuration file fills it, the pipeline only
 * reads it. The library has to be initialised before pipeline_box(); the
 * thread count, log level and FFT tuning are the caller's to apply, since
 * they are sif_init()'s arguments and not the pipeline's.
 */

#ifndef SIF_EXODUS_PIPELINE_H
#define SIF_EXODUS_PIPELINE_H

#include "sif/core/macros.h"
#include "sif/io/field_io.h"
#include "sif/io/fits_io.h"
#include "sif/io/gadget_io.h"
#include "sif/model/cosmology.h"

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
  EXODUS_INPUT_GADGET,
  /* A catalogue of one or more FITS tables, Cartesian positions. */
  EXODUS_INPUT_FITS,
  /* Datasets of one or more HDF5 files of any layout. */
  EXODUS_INPUT_HDF5
} exodus_input_kind_t;

/* What the radii are measured in. */
typedef enum {
  /* The units of the box, which are the catalogue's. */
  EXODUS_RADII_PHYSICAL,
  /* Mean tracer separations, box / N^(1/3) -- on a survey, the data's
   * inside the footprint: scaled to the box's units once the fields have
   * been read, and recorded that way in the catalogue. */
  EXODUS_RADII_MPS
} exodus_radii_units_t;

/* The format the catalogue is written in. */
typedef enum {
  /* /catalogue in an HDF5 file, with the run's parameters as attributes. In a
   * build without HDF5 the library writes text next to the path instead. */
  EXODUS_OUTPUT_HDF5,
  /* sif_catalogue_write_ascii(): the voids, the run's parameters as
   * '#key=value' header lines. */
  EXODUS_OUTPUT_ASCII,
  /* sif_catalogue_write_fits(): a VOIDS table, with the run's parameters as
   * its keywords. */
  EXODUS_OUTPUT_FITS
} exodus_output_kind_t;

/* Which pipeline runs. */
typedef enum {
  /* A simulation: one tracer set in a periodic box. */
  EXODUS_MODE_BOX,
  /* A survey: data and randoms, with edges, possibly on the sky. */
  EXODUS_MODE_SURVEY
} exodus_mode_t;

/* One particle file, or set of files, and how to read it: the tracers of a
 * box, or a survey's data or randoms. */
typedef struct {
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

  /* FITS and HDF5: the files, read as one catalogue in order -- `path`
   * above is the first of them -- and what fills x, y, z and the weight:
   * FITS columns or expressions, HDF5 datasets. Never velocities, which
   * the finder does not read. */
  const char* const* paths;
  uint32_t n_paths;
  sif_field_columns_t named;

  /* FITS only. */
  struct {
    /* Row filter, or NULL. */
    const char* where;
    /* The table's EXTNAME or extension number, or NULL for the first. */
    const char* hdu;
    /* Share of the filtered rows to keep, in (0, 1], and its seed. */
    double fraction;
    uint64_t seed;
  } fits;

  /* HDF5 only. */
  struct {
    /* What Cartesian positions are multiplied by: 1e-3 from kpc/h. */
    double length_scale;
    double fraction;
    uint64_t seed;
  } hdf5;
} exodus_input_t;

typedef struct {
  exodus_mode_t mode;

  /* --- input ---------------------------------------------------------- */
  /* The tracers: a box's, or a survey's data. */
  exodus_input_t input;
  /* Survey only: the randoms, which describe the footprint and the
   * selection. Read like the data, from any format; box_length is not used. */
  exodus_input_t randoms;

  /* --- survey --------------------------------------------------------- */
  struct {
    /* The background that takes sky coordinates to comoving positions, in
     * Mpc/h, and the voids back. Needed when the inputs are on the sky -- and
     * both have to be, or neither -- and when sky is set. */
    bool has_cosmology;
    sif_cosmology_t cosmology;
    /* Write the voids' centres as ra, dec and redshift; otherwise as
     * comoving positions with the observer at the origin, the frame the
     * inputs were in or were converted to. The configuration sets it from
     * input.coordinates, so the voids come out as the inputs went in. */
    bool sky;
  } survey;

  /* --- density grid --------------------------------------------------- */
  struct {
    /* Cells per side; 0 takes one cell per mean tracer separation, rounded
     * up to a size the FFT is fast at. In a survey the separation is the
     * data's inside the footprint, and the cells span the padded box. */
    uint32_t n_cells;
  } grid;

  /* --- chain mesh ----------------------------------------------------- */
  struct {
    /* Cells per side -- a survey's data mesh -- where 0 takes
     * sif_finder_suggest_mesh_cells(), or on a survey
     * sif_finder_suggest_mesh_cells_survey(). Only speed and memory depend
     * on it, never the catalogue. */
    uint32_t n_cells;
    /* Survey only: the randoms' mesh, the same way. */
    uint32_t n_cells_random;
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

/* Run the pipeline on a survey, the same way. */
SIF_NODISCARD int pipeline_survey(const exodus_params_t* params);

/* What a survey run will be. Unlike a box's, its shape is only known once
 * the randoms are read -- the footprint sizes the box, the grid and the
 * meshes -- so working it out reads both files and measures the footprint,
 * as the run does first. */
typedef struct {
  uint64_t n_tracers;
  uint64_t n_randoms;
  bool weighted;
  bool randoms_weighted;
  /* The footprint's comoving volume, and the data's mean separation in it. */
  double footprint_volume;
  double mps;
  /* The padded box the survey is searched in. */
  double box;
  /* The ladder's ends, in comoving units whatever it was given in. */
  double r_min;
  double r_max;
  uint32_t grid_cells;
  /* The meshes: as given, or about what the run will choose. */
  uint32_t mesh_cells;
  uint32_t random_mesh_cells;
  uint64_t peak_bytes;
} survey_plan_t;

/* Work out `plan`, with the library initialised. Returns SIF_OK, or the
 * status of what failed, already logged. */
SIF_NODISCARD int pipeline_survey_plan(
  const exodus_params_t* params, survey_plan_t* plan);

/* About the most memory a survey run holds at once, in bytes. */
uint64_t pipeline_survey_peak_bytes(uint64_t n_data, bool data_weighted,
  uint64_t n_randoms, bool randoms_weighted, uint32_t grid_cells,
  uint32_t data_mesh_cells, uint32_t random_mesh_cells);

#endif /* SIF_EXODUS_PIPELINE_H */
