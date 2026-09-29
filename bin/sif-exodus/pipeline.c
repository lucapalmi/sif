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

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
#define GRID_CELLS_PER_SEPARATION 1.0

/* The default grid's floor, for a handful of tracers: below it there is no
 * room for a top-hat to be anything but a cell or two. */
#define GRID_MIN_CELLS 16

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

/* GRID_CELLS_PER_SEPARATION cells per mean tracer separation, box / N^(1/3).
 * The finder wants its smallest radius to span two cells at least, so on this
 * grid a ladder starts at two separations. */
uint32_t pipeline_default_grid_cells(uint64_t n_tracers) {
  const double target =
    ceil(GRID_CELLS_PER_SEPARATION * cbrt((double)n_tracers));
  return fft_friendly(
    target > GRID_MIN_CELLS ? (uint32_t)target : GRID_MIN_CELLS);
}

/* --- memory ------------------------------------------------------------ */

static uint64_t cube(uint32_t n) { return (uint64_t)n * n * n; }

/* The most the run holds at once is at one of three moments, and which one
 * depends on the sizes:
 *
 *   the CIC deposit   the tracers, the grid, per-thread slabs that add up to
 *                     about another grid, and 8 bytes a tracer to sort them;
 *   the mesh build    the tracers, the grid, and the sort's scratch: its key,
 *                     8 bytes a tracer, and one column to shuffle through;
 *   the finder        the tracers (the mesh's by now), the grid's spectrum
 *                     and its working copy, the accepted-void mask -- and,
 *                     measured, as much again as the grid itself.
 *
 * The last is what a run holds by the process's own count, not what the
 * finder's code allocates by name: the forward transform's working memory
 * sits there, and a grid-sized allowance is what matches the resident size at
 * 200^3 to 512^3, with one thread or eight. FFTW tuning a new size plans on a
 * scratch grid at a moment that holds less than this, so it adds nothing.
 * Allocations too small to matter -- the catalogue, lookup tables, per-thread
 * histograms -- are left out. */
uint64_t pipeline_peak_bytes(
  uint64_t n_tracers, bool weighted, uint32_t grid_cells, uint32_t mesh_cells) {

  const uint64_t r = sizeof(sif_real);
  const uint64_t tracers = n_tracers * (weighted ? 4 : 3) * r;
  const uint64_t grid = cube(grid_cells) * r;
  const uint64_t spectrum =
    (uint64_t)grid_cells * grid_cells * (grid_cells / 2 + 1) * 2 * r;
  const uint64_t offsets = (cube(mesh_cells) + 1) * sizeof(uint64_t);
  const uint64_t mask = cube(grid_cells) / 8;

  const uint64_t cic = tracers + 2 * grid + n_tracers * sizeof(uint64_t);
  const uint64_t mesh =
    tracers + grid + offsets + n_tracers * (sizeof(uint64_t) + r);
  const uint64_t finder = tracers + offsets + mask + grid + 2 * spectrum;

  /* The program itself, its libraries and FFTW's plans. */
  const uint64_t baseline = 16ull << 20;

  uint64_t peak = cic > mesh ? cic : mesh;
  return baseline + (finder > peak ? finder : peak);
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

/* What the run found out along the way, for the log and the catalogue. */
typedef struct {
  double box;
  /* Mean tracer separation, box / N^(1/3). */
  double mps;
  uint64_t n_tracers;
  bool weighted;
  uint32_t grid_cells;
  uint32_t mesh_cells;
  /* The ladder's ends, in the box's units whatever it was given in. */
  sif_real r_min;
  sif_real r_max;
} run_facts_t;

/* Record what made the catalogue on /catalog, where it goes with it: a
 * catalogue written over this one later takes these away. */
static int describe_catalog(
  const char* f, const exodus_params_t* p, const run_facts_t* r) {
  const char* g = "catalog";

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
    s = sif_hdf5_set_attr_real(f, g, "r_min", r->r_min);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(f, g, "r_max", r->r_max);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(f, g, "box_length", r->box);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_real(f, g, "mean_separation", r->mps);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "grid_n_cells", r->grid_cells);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "mesh_n_cells", r->mesh_cells);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "n_tracers", (int64_t)r->n_tracers);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_int(f, g, "weighted", r->weighted);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_string(f, g, "input", p->input.path);
  if (s == SIF_OK)
    s = sif_hdf5_set_attr_string(
      f, g, "input_kind", input_kind_name(p->input.kind));
  return s;
}

/* Write the catalogue beside the output, then move it into place. An output
 * that already exists is replaced only by a complete one -- a run that fails
 * or is killed leaves it as it was -- and never merged into: an HDF5 file is
 * rewritten whole, not just its /catalog. */
static int write_output(
  const exodus_params_t* p, const run_facts_t* r, const sif_catalog_t* cat) {

  const char* path = p->output.path;
  const size_t len = strlen(path) + 32;
  char* tmp = malloc(len);
  if (!tmp)
    return SIF_ERR_ALLOC;
  snprintf(tmp, len, "%s.tmp.%ld", path, (long)getpid());

  int status;
  if (p->output.kind == EXODUS_OUTPUT_HDF5) {
    status = sif_catalog_write_hdf5(tmp, cat);
    if (status == SIF_OK)
      status = describe_catalog(tmp, p, r);
  } else {
    status = sif_catalog_write_ascii(cat, tmp);
  }

  if (status == SIF_OK && rename(tmp, path) != 0) {
    SIF_LOG_ERROR(TAG, "could not move %s into place as %s: %s", tmp, path,
      strerror(errno));
    status = SIF_ERR_IO;
  }
  if (status != SIF_OK)
    remove(tmp);

  free(tmp);
  return status;
}

/* --- the pipeline ------------------------------------------------------ */

int pipeline_box(const exodus_params_t* p) {
  sif_field_t* field = NULL;
  sif_grid_t* grid = NULL;
  sif_chain_mesh_t* mesh = NULL;
  sif_catalog_t* cat = NULL;
  sif_real* radii = NULL;
  run_facts_t r = {0};
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
  double file_box;
  status = read_field(p, &field, &file_box);
  if (status != SIF_OK)
    goto done;
  status = resolve_box(p, file_box, &r.box);
  if (status != SIF_OK)
    goto done;
  const double box = r.box;

  /* Always. What it is for is rounding: a coordinate stored in single
   * precision lands on the far face of the box as often as not, and the CIC
   * and the mesh both refuse it there. The library says what it folded, and
   * warns if a coordinate was genuinely outside the box rather than on its
   * face. */
  status = sif_field_wrap_periodic(field, (sif_real)box, NULL, NULL);
  if (status != SIF_OK)
    goto done;

  r.n_tracers = field->n_particles;
  r.weighted = field->weights != NULL;
  r.mps = box / cbrt((double)r.n_tracers);
  SIF_LOG_INFO(TAG,
    "field: %" PRIu64 " %s tracers from %s, box %g, mean separation %g "
    "(%.2f s)",
    r.n_tracers, r.weighted ? "weighted" : "unweighted", p->input.path, box,
    r.mps, seconds_since(&t));

  /* The ladder in the box's units, which is what the finder and the
   * catalogue work in. */
  const double scale = p->finder.radii_units == EXODUS_RADII_MPS ? r.mps : 1.0;
  radii = malloc(p->finder.n_radii * sizeof(sif_real));
  if (!radii) {
    status = SIF_ERR_ALLOC;
    goto done;
  }
  for (uint32_t i = 0; i < p->finder.n_radii; i++)
    radii[i] = (sif_real)(p->finder.radii[i] * scale);
  r.r_min = r.r_max = radii[0];
  for (uint32_t i = 1; i < p->finder.n_radii; i++) {
    r.r_min = fmin(r.r_min, radii[i]);
    r.r_max = fmax(r.r_max, radii[i]);
  }
  if (p->finder.radii_units == EXODUS_RADII_MPS)
    SIF_LOG_INFO(TAG, "radii: %g to %g mean separations, %g to %g in the box",
      r.r_min / scale, r.r_max / scale, (double)r.r_min, (double)r.r_max);

  /* 2. Grid. Built from the field before the mesh takes it over. */
  sif_timer_start(&t);
  r.grid_cells = p->grid.n_cells ? p->grid.n_cells
                                 : pipeline_default_grid_cells(r.n_tracers);
  grid = sif_grid_alloc(r.grid_cells, (sif_real)box);
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
  SIF_LOG_INFO(TAG, "grid: %u^3 cells of %g%s, %.2f GiB (%.2f s)", r.grid_cells,
    (double)grid->cell_length,
    p->grid.n_cells ? "" : " (one per mean separation)",
    (double)grid->total_cells * sizeof(sif_real) / (1024.0 * 1024.0 * 1024.0),
    seconds_since(&t));

  /* 3. Mesh. It takes the field's storage over rather than copying it, so
   * the tracers are held once from here on; the empty field goes at once. */
  sif_timer_start(&t);
  r.mesh_cells = p->mesh.n_cells;
  if (r.mesh_cells == 0) {
    r.mesh_cells =
      sif_finder_suggest_mesh_cells(r.n_tracers, (sif_real)box, r.r_max);
    if (r.mesh_cells == 0) {
      SIF_LOG_ERROR(TAG,
        "the largest radius, %g, needs a search sphere wider than the box",
        (double)r.r_max);
      status = SIF_ERR_INVALID;
      goto done;
    }
  }

  mesh = sif_chain_mesh_alloc_consume(
    r.mesh_cells, (sif_real)box, field, SIF_MESH_DROP_INDICES);
  sif_field_free(field);
  field = NULL;
  if (!mesh) {
    status = SIF_ERR_ALLOC;
    goto done;
  }
  SIF_LOG_INFO(
    TAG, "mesh: %u^3 cells (%.2f s)", r.mesh_cells, seconds_since(&t));

  /* 4. Finder. The grid is not needed afterwards, so it is not restored. */
  sif_timer_start(&t);
  cat = sif_finder_exodus(grid, mesh, radii, p->finder.n_radii,
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
  status = write_output(p, &r, cat);
  if (status != SIF_OK)
    goto done;
  SIF_LOG_INFO(
    TAG, "catalog: written to %s (%.2f s)", p->output.path, seconds_since(&t));

  SIF_LOG_INFO(TAG, "done in %.2f s", seconds_since(&t_total));

done:
  free(radii);
  sif_catalog_free(cat);
  sif_chain_mesh_free(mesh);
  sif_grid_free(grid);
  sif_field_free(field);
  return status;
}
