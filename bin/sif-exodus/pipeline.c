/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "pipeline.h"

#include "sif/core/macros.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/catalog_io.h"
#include "sif/io/field_io.h"
#include "sif/io/gadget_io.h"
#include "sif/io/hdf5_io.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/utils/logger.h"
#include "sif/utils/timer.h"

#include <inttypes.h>
#include <math.h>
#include <string.h>

#define TAG "sif-exodus"

static const char* input_kind_name(exodus_input_kind_t kind) {
  switch (kind) {
  case EXODUS_INPUT_XFIELD:
    return "xfield";
  case EXODUS_INPUT_ASCII:
    return "ascii";
  case EXODUS_INPUT_BINARY:
    return "binary";
  case EXODUS_INPUT_GADGET:
    return "gadget";
  }
  return "unknown";
}

static double seconds_since(sif_timer_t* timer) {
  sif_timer_stop(timer);
  return sif_timer_elapsed_ms(timer) / 1000.0;
}

/* The SIF_FINDER_SEARCH_* flag for a factor, or SIF_ERR_INVALID in *status
 * for one the finder does not offer. */
static sif_option search_option(double factor, int* status) {
  static const struct {
    double factor;
    sif_option opt;
  } table[] = {{1.25, SIF_FINDER_SEARCH_1_25}, {1.5, SIF_FINDER_SEARCH_1_50},
    {1.75, SIF_FINDER_SEARCH_1_75}, {2.0, SIF_FINDER_SEARCH_2_00}};

  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
    if (factor == table[i].factor) {
      *status = SIF_OK;
      return table[i].opt;
    }
  }

  SIF_LOG_ERROR(
    TAG, "search factor %g is not one of 1.25, 1.5, 1.75, 2.0", factor);
  *status = SIF_ERR_INVALID;
  return SIF_DEFAULT;
}

/* --- grid -------------------------------------------------------------- */

/* Cells per mean tracer separation in the default grid. */
#define GRID_CELLS_PER_SEPARATION 4.0

/* The smallest n >= target that FFTW transforms fastest: even, for the
 * real-to-complex transform's halved axis, with no prime factor above 5. */
static uint32_t fft_friendly(uint32_t target) {
  for (uint32_t n = target + (target & 1u);; n += 2) {
    uint32_t m = n;
    while (m % 2 == 0)
      m /= 2;
    while (m % 3 == 0)
      m /= 3;
    while (m % 5 == 0)
      m /= 5;
    if (m == 1)
      return n;
  }
}

/* The default grid: GRID_CELLS_PER_SEPARATION cells per mean tracer
 * separation, box / N^(1/3). */
static uint32_t default_grid_cells(uint64_t n_tracers) {
  const double target =
    ceil(GRID_CELLS_PER_SEPARATION * cbrt((double)n_tracers));
  return fft_friendly((uint32_t)target);
}

/* --- field ------------------------------------------------------------- */

/* A column format that names a velocity, which would be read, carried into
 * the mesh and never looked at. Column names are x, y, z, vx, vy, vz, w and
 * `*`, so any `v` is one. */
static bool columns_name_velocities(const char* columns) {
  return strchr(columns, 'v') || strchr(columns, 'V');
}

/* Read the tracers. *file_box is the box the file records, or 0. */
static int read_field(
  const exodus_params_t* p, sif_field_t** out, double* file_box) {

  sif_field_t* field = NULL;
  int status = SIF_OK;
  *file_box = 0.0;

  switch (p->input.kind) {
  case EXODUS_INPUT_XFIELD:
    field = sif_field_read(p->input.path, file_box);
    if (!field)
      return SIF_ERR_IO;
    break;

  case EXODUS_INPUT_ASCII:
  case EXODUS_INPUT_BINARY:
    if (!p->input.columns) {
      SIF_LOG_ERROR(TAG, "a %s input needs its column format",
        input_kind_name(p->input.kind));
      return SIF_ERR_INVALID;
    }
    if (columns_name_velocities(p->input.columns)) {
      SIF_LOG_ERROR(TAG,
        "columns \"%s\" read velocities, which the finder never uses; "
        "skip them with *",
        p->input.columns);
      return SIF_ERR_INVALID;
    }

    /* A count of 0: the readers size the field from the file. */
    field = sif_field_alloc(0);
    if (!field)
      return SIF_ERR_ALLOC;

    if (p->input.kind == EXODUS_INPUT_ASCII)
      status = sif_field_read_ascii(field, p->input.path, p->input.columns,
        p->input.delimiter, p->input.skip_header);
    else
      status = sif_field_read_binary(field, p->input.path, p->input.columns,
        p->input.binary.layout, p->input.binary.precision,
        p->input.binary.endian, p->input.binary.header_bytes);
    break;

  case EXODUS_INPUT_GADGET:
    status = sif_field_read_gadget(p->input.path, p->input.gadget.format,
      p->input.gadget.ptype, SIF_GADGET_VELOCITY_SKIP,
      p->input.gadget.masses ? SIF_GADGET_MASS_READ : SIF_GADGET_MASS_SKIP,
      p->input.gadget.length, p->input.gadget.fraction, p->input.gadget.seed,
      &field, file_box);
    break;

  default:
    SIF_LOG_ERROR(TAG, "unknown input kind %d", (int)p->input.kind);
    return SIF_ERR_INVALID;
  }

  if (status != SIF_OK) {
    sif_field_free(field);
    return status;
  }

  *out = field;
  return SIF_OK;
}

/* The box the run uses: the configured one if given, else the file's. */
static int resolve_box(const exodus_params_t* p, double file_box, double* box) {
  const double given = p->input.box_length;

  if (given > 0.0) {
    if (file_box > 0.0 && fabs(given - file_box) > 1e-6 * file_box)
      SIF_LOG_WARNING(TAG, "box_length %g overrides the %g recorded in %s",
        given, file_box, p->input.path);
    *box = given;
    return SIF_OK;
  }

  if (file_box > 0.0) {
    *box = file_box;
    return SIF_OK;
  }

  SIF_LOG_ERROR(
    TAG, "%s does not record the box: give box_length", p->input.path);
  return SIF_ERR_INVALID;
}

/* --- catalogue --------------------------------------------------------- */

/* Record what made the catalogue on /catalog, where it goes with it: a
 * catalogue written over this one later takes these away. */
static int describe_catalog(const exodus_params_t* p, double box,
  uint32_t grid_cells, uint32_t mesh_cells, uint64_t n_tracers, bool weighted) {

  const char* f = p->output.path;
  const char* g = "catalog";

  sif_real r_min = p->finder.radii[0], r_max = p->finder.radii[0];
  for (uint32_t i = 1; i < p->finder.n_radii; i++) {
    r_min = fmin(r_min, p->finder.radii[i]);
    r_max = fmax(r_max, p->finder.radii[i]);
  }

  int s = SIF_OK;
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_string(f, g, "finder", "exodus");
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(f, g, "threshold", p->finder.threshold);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(
      f, g, "overlap_fraction", p->finder.overlap_fraction);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(f, g, "search_factor", p->finder.search_factor);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "n_radii", p->finder.n_radii);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(f, g, "r_min", r_min);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(f, g, "r_max", r_max);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(f, g, "box_length", box);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "grid_n_cells", grid_cells);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "mesh_n_cells", mesh_cells);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "n_tracers", (int64_t)n_tracers);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "weighted", weighted);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_string(f, g, "input", p->input.path);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_string(
      f, g, "input_kind", input_kind_name(p->input.kind));
  return s;
}

/* --- the pipeline ------------------------------------------------------ */

int pipeline_box(const exodus_params_t* p) {
  sif_field_t* field = NULL;
  sif_grid_t* grid = NULL;
  sif_chain_mesh_t* mesh = NULL;
  sif_catalog_t* cat = NULL;
  sif_timer_t t_total, t;
  int status;

  if (!p || !p->input.path || !p->output.path || !p->finder.radii ||
      p->finder.n_radii == 0) {
    SIF_LOG_ERROR(TAG, "incomplete parameters");
    return SIF_ERR_INVALID;
  }

  const sif_option search = search_option(p->finder.search_factor, &status);
  if (status != SIF_OK)
    return status;

  /* Only an .xfield or a snapshot records its box; say so before reading a
   * file that does not, rather than after. */
  const bool file_has_box = p->input.kind == EXODUS_INPUT_XFIELD ||
                            p->input.kind == EXODUS_INPUT_GADGET;
  if (!file_has_box && !(p->input.box_length > 0.0)) {
    SIF_LOG_ERROR(TAG, "%s input records no box: give box_length",
      input_kind_name(p->input.kind));
    return SIF_ERR_INVALID;
  }

  sif_timer_start(&t_total);

  /* 1. Field. */
  sif_timer_start(&t);
  double file_box, box;
  status = read_field(p, &field, &file_box);
  if (status != SIF_OK)
    goto done;
  status = resolve_box(p, file_box, &box);
  if (status != SIF_OK)
    goto done;

  /* Always. What it is for is rounding: a coordinate stored in single
   * precision lands on the far face of the box as often as not, and the CIC
   * and the mesh both refuse it there. The library says what it folded, and
   * warns if a coordinate was genuinely outside the box rather than on its
   * face. */
  status = sif_field_wrap_periodic(field, (sif_real)box, NULL, NULL);
  if (status != SIF_OK)
    goto done;

  const uint64_t n_tracers = field->n_particles;
  const bool weighted = field->weights != NULL;
  SIF_LOG_INFO(TAG, "field: %" PRIu64 " %s tracers from %s, box %g (%.2f s)",
    n_tracers, weighted ? "weighted" : "unweighted", p->input.path, box,
    seconds_since(&t));

  /* 2. Grid. Built from the field before the mesh takes it over. */
  sif_timer_start(&t);
  const uint32_t grid_cells =
    p->grid.n_cells ? p->grid.n_cells : default_grid_cells(n_tracers);
  grid = sif_grid_alloc(grid_cells, (sif_real)box);
  if (!grid) {
    status = SIF_ERR_ALLOC;
    goto done;
  }
  status = sif_grid_assign_cic(grid, field);
  if (status != SIF_OK)
    goto done;
  status = sif_grid_to_density_contrast(grid);
  if (status != SIF_OK)
    goto done;
  SIF_LOG_INFO(TAG, "grid: %u^3 cells of %g%s, %.2f GiB (%.2f s)", grid_cells,
    (double)grid->cell_length,
    p->grid.n_cells ? "" : " (4 per mean separation)",
    (double)grid->total_cells * sizeof(sif_real) / (1024.0 * 1024.0 * 1024.0),
    seconds_since(&t));

  /* 3. Mesh. It takes the field's storage over rather than copying it, so
   * the tracers are held once from here on; the empty field goes at once. */
  sif_timer_start(&t);
  sif_real r_max = p->finder.radii[0];
  for (uint32_t i = 1; i < p->finder.n_radii; i++)
    r_max = fmax(r_max, p->finder.radii[i]);

  uint32_t mesh_cells = p->mesh.n_cells;
  if (mesh_cells == 0) {
    mesh_cells = sif_finder_suggest_mesh_cells(n_tracers, (sif_real)box, r_max);
    if (mesh_cells == 0) {
      SIF_LOG_ERROR(TAG,
        "the largest radius, %g, needs a search sphere wider than the box",
        (double)r_max);
      status = SIF_ERR_INVALID;
      goto done;
    }
  }

  mesh = sif_chain_mesh_alloc_consume(
    mesh_cells, (sif_real)box, field, SIF_MESH_DROP_INDICES);
  sif_field_free(field);
  field = NULL;
  if (!mesh) {
    status = SIF_ERR_ALLOC;
    goto done;
  }
  SIF_LOG_INFO(TAG, "mesh: %u^3 cells (%.2f s)", mesh_cells, seconds_since(&t));

  /* 4. Finder. The grid is not needed afterwards, so it is not restored. */
  sif_timer_start(&t);
  cat = sif_finder_exodus(grid, mesh, p->finder.radii, p->finder.n_radii,
    (sif_real)p->finder.threshold, (sif_real)p->finder.overlap_fraction,
    SIF_FINDER_CONSUME_GRID | search);
  sif_chain_mesh_free(mesh);
  mesh = NULL;
  sif_grid_free(grid);
  grid = NULL;
  if (!cat) {
    status = SIF_ERR_INVALID;
    goto done;
  }
  SIF_LOG_INFO(
    TAG, "finder: %" PRIu64 " voids (%.2f s)", cat->n_voids, seconds_since(&t));

  /* 5. Catalogue. */
  sif_timer_start(&t);
  if (p->output.kind == EXODUS_OUTPUT_HDF5) {
    status = sif_catalog_write_hdf5(p->output.path, cat);
    if (status == SIF_OK)
      status =
        describe_catalog(p, box, grid_cells, mesh_cells, n_tracers, weighted);
  } else {
    status = sif_catalog_write_ascii(cat, p->output.path);
  }
  if (status != SIF_OK)
    goto done;
  SIF_LOG_INFO(
    TAG, "catalog: written to %s (%.2f s)", p->output.path, seconds_since(&t));

  SIF_LOG_INFO(TAG, "done in %.2f s", seconds_since(&t_total));

done:
  sif_catalog_free(cat);
  sif_chain_mesh_free(mesh);
  sif_grid_free(grid);
  sif_field_free(field);
  return status;
}
