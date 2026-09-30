/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "pipeline.h"

#include "sif/core/macros.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/catalogue_io.h"
#include "sif/io/field_io.h"
#include "sif/io/fits_io.h"
#include "sif/io/gadget_io.h"
#include "sif/io/hdf5_io.h"
#include "sif/model/cosmology.h"
#include "sif/structures/catalogue.h"
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
  case EXODUS_INPUT_FITS:
    return "fits";
  case EXODUS_INPUT_HDF5:
    return "hdf5";
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

/* Read one input. *file_box is the box the file records, or 0. */
static int read_field(
  const exodus_input_t* in, sif_field_t** out, double* file_box) {

  sif_field_t* field = NULL;
  *file_box = 0.0;

  switch (in->kind) {
  case EXODUS_INPUT_XFIELD:
    field = sif_field_read(in->path, file_box);
    break;

  case EXODUS_INPUT_ASCII:
  case EXODUS_INPUT_BINARY:
    if (!in->columns) {
      SIF_LOG_ERROR(
        TAG, "a %s input needs its column format", input_kind_name(in->kind));
      return SIF_ERR_INVALID;
    }
    if (columns_name_velocities(in->columns)) {
      SIF_LOG_ERROR(TAG,
        "columns \"%s\" read velocities, which the finder never uses; "
        "skip them with *",
        in->columns);
      return SIF_ERR_INVALID;
    }

    if (in->kind == EXODUS_INPUT_ASCII)
      field = sif_field_read_ascii(
        in->path, in->columns, in->delimiter, in->skip_header);
    else
      field = sif_field_read_binary(in->path, in->columns, in->binary.layout,
        in->binary.precision, in->binary.endian, in->binary.header_bytes);
    break;

  case EXODUS_INPUT_GADGET:
    field = sif_field_read_gadget(in->path, in->gadget.format, in->gadget.ptype,
      SIF_GADGET_VELOCITY_SKIP,
      in->gadget.masses ? SIF_GADGET_MASS_READ : SIF_GADGET_MASS_SKIP,
      in->gadget.length, in->gadget.fraction, in->gadget.seed, file_box);
    break;

  case EXODUS_INPUT_FITS:
    field = sif_field_read_fits(in->paths, in->n_paths, in->fits.hdu,
      &in->named, in->fits.where, in->fits.fraction, in->fits.seed);
    break;

  case EXODUS_INPUT_HDF5:
    field = sif_field_read_hdf5(in->paths, in->n_paths, &in->named,
      in->hdf5.length_scale, in->hdf5.fraction, in->hdf5.seed);
    break;

  default:
    SIF_LOG_ERROR(TAG, "unknown input kind %d", (int)in->kind);
    return SIF_ERR_INVALID;
  }

  /* Every reader returns NULL on failure, and has logged why. */
  if (!field)
    return SIF_ERR_IO;

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
  /* The periodic box; on a survey, the padded box it was searched in. */
  double box;
  /* Mean tracer separation, box / N^(1/3); on a survey, the data's inside
   * the footprint. */
  double mps;
  uint64_t n_tracers;
  bool weighted;
  uint32_t grid_cells;
  uint32_t mesh_cells;
  /* The ladder's ends, in the box's units whatever it was given in. */
  sif_real r_min;
  sif_real r_max;

  /* Survey only. */
  uint64_t n_randoms;
  bool randoms_weighted;
  uint32_t random_mesh_cells;
  /* The footprint's comoving volume, as the randoms cover it. */
  double footprint_volume;
  /* The inputs were on the sky, and converted with the cosmology. */
  bool sky_input;
} run_facts_t;

/* Record what made the catalogue in its metadata, which every format then
 * writes: attributes of /catalogue in HDF5, keywords of the VOIDS table in
 * FITS, '#key=value' lines in the ASCII header. A value that cannot be
 * recorded -- a path with a double quote in it, say -- is warned about and
 * left out; it is no reason to lose the run. */
static void describe_catalogue(
  sif_catalogue_t* cat, const exodus_params_t* p, const run_facts_t* r) {
  int s = sif_catalogue_meta_string_set(cat, "finder", "exodus");
  s |= sif_catalogue_meta_real_set(cat, "threshold", p->finder.threshold);
  s |= sif_catalogue_meta_real_set(
    cat, "overlap_fraction", p->finder.overlap_fraction);
  s |= sif_catalogue_meta_real_set(cat, "search_factor", p->finder.search_factor);
  s |= sif_catalogue_meta_int_set(cat, "n_radii", p->finder.n_radii);
  s |= sif_catalogue_meta_real_set(cat, "r_min", r->r_min);
  s |= sif_catalogue_meta_real_set(cat, "r_max", r->r_max);
  s |= sif_catalogue_meta_string_set(
    cat, "mode", p->mode == EXODUS_MODE_SURVEY ? "survey" : "box");
  s |= sif_catalogue_meta_real_set(cat, "box_length", r->box);
  s |= sif_catalogue_meta_real_set(cat, "mean_separation", r->mps);
  s |= sif_catalogue_meta_int_set(cat, "grid_n_cells", r->grid_cells);
  s |= sif_catalogue_meta_int_set(cat, "mesh_n_cells", r->mesh_cells);
  s |= sif_catalogue_meta_int_set(cat, "n_tracers", (int64_t)r->n_tracers);
  s |= sif_catalogue_meta_int_set(cat, "weighted", r->weighted);
  s |= sif_catalogue_meta_string_set(cat, "input", p->input.path);
  s |= sif_catalogue_meta_string_set(
    cat, "input_kind", input_kind_name(p->input.kind));

  if (p->mode == EXODUS_MODE_SURVEY) {
    s |= sif_catalogue_meta_string_set(cat, "randoms", p->randoms.path);
    s |= sif_catalogue_meta_string_set(
      cat, "randoms_kind", input_kind_name(p->randoms.kind));
    s |= sif_catalogue_meta_int_set(cat, "n_randoms", (int64_t)r->n_randoms);
    s |= sif_catalogue_meta_int_set(cat, "randoms_weighted", r->randoms_weighted);
    s |= sif_catalogue_meta_int_set(
      cat, "random_mesh_n_cells", r->random_mesh_cells);
    s |=
      sif_catalogue_meta_real_set(cat, "footprint_volume", r->footprint_volume);
    s |= sif_catalogue_meta_int_set(cat, "sky_input", r->sky_input);
    if (p->survey.has_cosmology) {
      const sif_cosmology_t* c = &p->survey.cosmology;
      s |= sif_catalogue_meta_real_set(cat, "omega_m", c->omega_m);
      s |= sif_catalogue_meta_real_set(cat, "omega_de", c->omega_de);
      s |= sif_catalogue_meta_real_set(cat, "omega_r", c->omega_r);
      s |= sif_catalogue_meta_real_set(cat, "w0", c->w0);
      s |= sif_catalogue_meta_real_set(cat, "wa", c->wa);
    }
  }
  if (s != SIF_OK)
    SIF_LOG_WARNING(
      TAG, "some of the run's settings could not be recorded in the catalogue");
}

/* Write the catalogue beside the output, then move it into place. An output
 * that already exists is replaced only by a complete one -- a run that fails
 * or is killed leaves it as it was -- and never merged into: an HDF5 file is
 * rewritten whole, not just its /catalogue, and a FITS file likewise. */
static int write_output(
  const exodus_params_t* p, const run_facts_t* r, sif_catalogue_t* cat) {

  const char* path = p->output.path;
  const size_t len = strlen(path) + 32;
  char* tmp = malloc(len);
  if (!tmp)
    return SIF_ERR_ALLOC;
  snprintf(tmp, len, "%s.tmp.%ld", path, (long)getpid());

  int status;
  describe_catalogue(cat, p, r);
  if (p->output.kind == EXODUS_OUTPUT_HDF5) {
    status = sif_catalogue_write_hdf5(tmp, cat);
  } else if (p->output.kind == EXODUS_OUTPUT_FITS) {
    status = sif_catalogue_write_fits(tmp, cat);
  } else {
    status = sif_catalogue_write_ascii(tmp, cat);
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

/* --- the pipelines ----------------------------------------------------- */

/* The ladder in the box's units, which is what the finder and the catalogue
 * work in, and its ends into r. r->mps has to be known. NULL if out of
 * memory. */
static sif_real* box_ladder(const exodus_params_t* p, run_facts_t* r) {
  const double scale = p->finder.radii_units == EXODUS_RADII_MPS ? r->mps : 1.0;
  sif_real* radii = malloc(p->finder.n_radii * sizeof(sif_real));
  if (!radii)
    return NULL;
  for (uint32_t i = 0; i < p->finder.n_radii; i++)
    radii[i] = (sif_real)(p->finder.radii[i] * scale);
  r->r_min = r->r_max = radii[0];
  for (uint32_t i = 1; i < p->finder.n_radii; i++) {
    r->r_min = fmin(r->r_min, radii[i]);
    r->r_max = fmax(r->r_max, radii[i]);
  }
  if (p->finder.radii_units == EXODUS_RADII_MPS)
    SIF_LOG_INFO(TAG, "radii: %g to %g mean separations, %g to %g in the box",
      r->r_min / scale, r->r_max / scale, (double)r->r_min, (double)r->r_max);
  return radii;
}

int pipeline_box(const exodus_params_t* p) {
  sif_field_t* field = NULL;
  sif_grid_t* grid = NULL;
  sif_chain_mesh_t* mesh = NULL;
  sif_catalogue_t* cat = NULL;
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
  status = read_field(&p->input, &field, &file_box);
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

  radii = box_ladder(p, &r);
  if (!radii) {
    status = SIF_ERR_ALLOC;
    goto done;
  }

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
    TAG, "catalogue: written to %s (%.2f s)", p->output.path, seconds_since(&t));

  SIF_LOG_INFO(TAG, "done in %.2f s", seconds_since(&t_total));

done:
  free(radii);
  sif_catalogue_free(cat);
  sif_chain_mesh_free(mesh);
  sif_grid_free(grid);
  sif_field_free(field);
  return status;
}

/* --- survey ------------------------------------------------------------ */

/* Cells per side of the grid the footprint is measured on. Coarse on
 * purpose: a cell counts as observed if any random reaches it, so it has to
 * be wide enough that the randoms leave no observed cell empty, even where
 * the selection is sparsest. */
#define FOOTPRINT_CELLS 128

/*
 * The footprint's comoving volume: the cells of a grid over the randoms'
 * bounding cube that any random reaches, by their CIC deposit. It is what
 * turns the data's count into a density, and so into the mean separation
 * the default grid and a ladder in mps are measured in. CIC spreads each
 * random over eight cells, so the volume comes out a little large -- by
 * about a cell along the footprint's boundary.
 *
 * The randoms are moved into the grid and back, which leaves them where they
 * were to within a rounding.
 */
static int footprint_volume(sif_field_t* randoms, double* volume) {
  if (sif_field_refresh_bounds(randoms) != SIF_OK) {
    SIF_LOG_ERROR(TAG, "the randoms have no positions");
    return SIF_ERR_INVALID;
  }
  double extent = 0.0;
  for (int k = 0; k < 3; k++)
    extent = fmax(extent, randoms->max_p[k] - randoms->min_p[k]);
  if (!(extent > 0.0) || !isfinite(extent)) {
    SIF_LOG_ERROR(TAG, "the randoms span no volume");
    return SIF_ERR_INVALID;
  }

  /* Two empty cells on the low side, and at least as many on the high one,
   * so the deposit never reaches a face and wraps. */
  const double cell = extent / (FOOTPRINT_CELLS - 4);
  sif_real there[3], back[3];
  for (int k = 0; k < 3; k++) {
    there[k] = (sif_real)(2.0 * cell - randoms->min_p[k]);
    back[k] = -there[k];
  }

  sif_grid_t* grid =
    sif_grid_alloc(FOOTPRINT_CELLS, (sif_real)(FOOTPRINT_CELLS * cell));
  if (!grid)
    return SIF_ERR_ALLOC;
  int status = sif_field_translate(randoms, there);
  if (status == SIF_OK)
    status = sif_grid_assign_cic(grid, randoms);
  const int moved_back = sif_field_translate(randoms, back);
  if (status == SIF_OK)
    status = moved_back;

  uint64_t observed = 0;
  if (status == SIF_OK)
    for (uint64_t i = 0; i < grid->total_cells; i++)
      observed += grid->values[i] > 0;
  sif_grid_free(grid);

  *volume = (double)observed * cell * cell * cell;
  return status;
}

/*
 * The grid's cells per side, and the padded box the survey is searched in,
 * with the offset that moves it there. A grid given in the configuration is
 * taken as it is; the default is one cell per mean separation. The box
 * depends on the cells -- its padding has a cell in it -- and the cells on
 * the box, so the default is found by going back and forth until the two
 * agree, which takes a step or two: the padding is mostly the search sphere.
 */
static int survey_geometry(const exodus_params_t* p, const sif_field_t* data,
  const sif_field_t* randoms,
  const sif_real* radii, sif_option search, double mps, uint32_t* n_cells,
  sif_real offset[3], sif_real* box) {

  uint32_t n = p->grid.n_cells ? p->grid.n_cells : 64;
  for (int step = 0;; step++) {
    if (sif_finder_exodus_survey_box(data, randoms, radii, p->finder.n_radii, n,
          search, offset, box) != SIF_OK) {
      SIF_LOG_ERROR(TAG, "no box can hold the survey with a %u^3 grid", n);
      return SIF_ERR_INVALID;
    }
    if (p->grid.n_cells || step == 8)
      break;

    const double target = ceil(GRID_CELLS_PER_SEPARATION * *box / mps);
    const uint32_t next =
      fft_friendly(target > GRID_MIN_CELLS ? (uint32_t)target : GRID_MIN_CELLS);
    if (next == n)
      break;
    n = next;
  }
  *n_cells = n;
  return SIF_OK;
}

/*
 * The part of a survey run that decides its shape: read the data and the
 * randoms, take them off the sky, measure the footprint, and choose the grid
 * and the padded box -- the offset into it is not applied. What --check
 * does, and the first thing a run does, so the two cannot disagree. On
 * failure nothing is left allocated.
 */
static int survey_prepare(const exodus_params_t* p, sif_option search,
  sif_field_t** data_out, sif_field_t** randoms_out, run_facts_t* rf,
  sif_real** radii_out, sif_real offset[3], sif_real* box_out) {

  sif_field_t* data = NULL;
  sif_field_t* randoms = NULL;
  sif_real* radii = NULL;
  run_facts_t r = *rf;
  sif_timer_t t;
  sif_real box;
  int status;

  /* 1. Fields. The files record no box, and a survey needs none: the box is
   * chosen around the footprint below. */
  sif_timer_start(&t);
  double unused_box;
  status = read_field(&p->input, &data, &unused_box);
  if (status != SIF_OK)
    goto fail;
  status = read_field(&p->randoms, &randoms, &unused_box);
  if (status != SIF_OK)
    goto fail;

  /* Both on the sky, or neither: data converted and randoms not would sit
   * in different frames, and nothing downstream could tell. */
  if (data->units != randoms->units) {
    SIF_LOG_ERROR(TAG,
      "the data are %s and the randoms %s: read both as ra, dec, z or both "
      "as x, y, z",
      data->units == SIF_COORDINATES_SKY ? "on the sky" : "positions",
      randoms->units == SIF_COORDINATES_SKY ? "on the sky" : "positions");
    status = SIF_ERR_INVALID;
    goto fail;
  }
  r.sky_input = data->units == SIF_COORDINATES_SKY;
  if (r.sky_input) {
    if (!p->survey.has_cosmology) {
      SIF_LOG_ERROR(TAG, "the inputs are on the sky: give a cosmology");
      status = SIF_ERR_INVALID;
      goto fail;
    }
    status = sif_field_convert_sky_coordinates(data, &p->survey.cosmology);
    if (status == SIF_OK)
      status = sif_field_convert_sky_coordinates(randoms, &p->survey.cosmology);
    if (status != SIF_OK)
      goto fail;
  }

  r.n_tracers = data->n_particles;
  r.weighted = data->weights != NULL;
  r.n_randoms = randoms->n_particles;
  r.randoms_weighted = randoms->weights != NULL;
  SIF_LOG_INFO(TAG,
    "fields: %" PRIu64 " %s tracers from %s, %" PRIu64
    " %s randoms from %s%s (%.2f s)",
    r.n_tracers, r.weighted ? "weighted" : "unweighted", p->input.path,
    r.n_randoms, r.randoms_weighted ? "weighted" : "unweighted",
    p->randoms.path, r.sky_input ? ", off the sky" : "", seconds_since(&t));
  if (r.n_randoms < r.n_tracers)
    SIF_LOG_WARNING(TAG,
      "fewer randoms than tracers: the footprint and the mean density are "
      "only as good as the randoms, which are normally many times the data");

  /* 2. Footprint and box. The footprint gives the data's mean separation,
   * which a ladder in mps is measured in and the default grid is sized by;
   * the ladder and the grid then give the padded box. */
  sif_timer_start(&t);
  status = footprint_volume(randoms, &r.footprint_volume);
  if (status != SIF_OK)
    goto fail;
  r.mps = cbrt(r.footprint_volume / (double)r.n_tracers);

  radii = box_ladder(p, &r);
  if (!radii) {
    status = SIF_ERR_ALLOC;
    goto fail;
  }

  status = survey_geometry(
    p, data, randoms, radii, search, r.mps, &r.grid_cells, offset, &box);
  if (status != SIF_OK)
    goto fail;
  r.box = box;
  *box_out = box;
  *data_out = data;
  *randoms_out = randoms;
  *radii_out = radii;
  *rf = r;
  return SIF_OK;

fail:
  free(radii);
  sif_field_free(randoms);
  sif_field_free(data);
  return status;
}

/* The padding the survey box was given assumes mesh cells no wider than the
 * search sphere, which the suggested sizes always are; a size given in the
 * configuration has to be checked. */
static int check_survey_meshes(const exodus_params_t* p, const run_facts_t* r) {
  const double sphere = p->finder.search_factor * (double)r->r_max;
  const uint32_t min_cells = (uint32_t)ceil(r->box / sphere);
  const uint32_t given[2] = {p->mesh.n_cells, p->mesh.n_cells_random};
  const char* const which[2] = {"n_cells_data", "n_cells_random"};
  int status = SIF_OK;
  for (int i = 0; i < 2; i++) {
    if (given[i] && given[i] < min_cells) {
      SIF_LOG_ERROR(TAG,
        "mesh.%s = %u makes cells of %g, wider than the search sphere (%g) "
        "the box around the survey is padded for: give at least %u, or "
        "\"auto\"",
        which[i], given[i], r->box / given[i], sphere, min_cells);
      status = SIF_ERR_INVALID;
    }
  }
  return status;
}

/* About the most memory a survey run holds at once, in bytes: the same three
 * moments as a box's (pipeline_peak_bytes()), with two tracer sets, two
 * grids and two meshes.
 *
 *   the CIC deposits  both sets, both grids, a thread slab grid, the sort key
 *   the mesh builds   both sets, both grids, and the larger set once more,
 *                     sorted into its mesh, with its key
 *   the finder        both sets in their meshes, both grids, and for each a
 *                     spectrum and a working copy, and the FFT's own scratch
 *
 * Fitted to the resident size of runs of 6 and 16 million tracers and
 * randoms on 128^3 to 400^3 grids (macOS, eight threads), whose peak is not a
 * clean function of the grid: it depends on how the FFT factors the size.
 * This follows the largest of them, and sits up to half above the rest. */
uint64_t pipeline_survey_peak_bytes(uint64_t n_data, bool data_weighted,
  uint64_t n_randoms, bool randoms_weighted, uint32_t grid_cells,
  uint32_t data_mesh_cells, uint32_t random_mesh_cells) {

  const uint64_t r = sizeof(sif_real);
  const uint64_t n_max = n_data > n_randoms ? n_data : n_randoms;
  const uint64_t tracers = n_data * (data_weighted ? 4 : 3) * r +
                           n_randoms * (randoms_weighted ? 4 : 3) * r;
  const uint64_t grid = cube(grid_cells) * r;
  const uint64_t spectrum =
    (uint64_t)grid_cells * grid_cells * (grid_cells / 2 + 1) * 2 * r;
  const uint64_t offsets =
    (cube(data_mesh_cells) + cube(random_mesh_cells) + 2) * sizeof(uint64_t);
  const uint64_t masks = 2 * (cube(grid_cells) / 8);

  const uint64_t cic = tracers + 3 * grid + n_max * sizeof(uint64_t);
  const uint64_t mesh =
    tracers + 2 * grid + offsets + n_max * (sizeof(uint64_t) + 4 * r);
  const uint64_t finder = tracers + offsets + masks + 3 * grid + 4 * spectrum;
  const uint64_t baseline = 16ull << 20;

  uint64_t peak = cic > mesh ? cic : mesh;
  return baseline + (finder > peak ? finder : peak);
}

int pipeline_survey_plan(const exodus_params_t* p, survey_plan_t* plan) {
  sif_field_t* data = NULL;
  sif_field_t* randoms = NULL;
  sif_real* radii = NULL;
  run_facts_t r = {0};
  int status;

  memset(plan, 0, sizeof *plan);
  const sif_option search = search_option(p->finder.search_factor, &status);
  if (status != SIF_OK)
    return status;

  sif_real offset[3], box;
  status = survey_prepare(p, search, &data, &randoms, &r, &radii, offset, &box);
  sif_field_free(data);
  sif_field_free(randoms);
  free(radii);
  if (status != SIF_OK)
    return status;

  plan->n_tracers = r.n_tracers;
  plan->n_randoms = r.n_randoms;
  plan->weighted = r.weighted;
  plan->randoms_weighted = r.randoms_weighted;
  plan->footprint_volume = r.footprint_volume;
  plan->mps = r.mps;
  plan->box = r.box;
  plan->r_min = r.r_min;
  plan->r_max = r.r_max;
  plan->grid_cells = r.grid_cells;

  /* The meshes as sif_finder_suggest_mesh_cells_survey() sizes them, from
   * the footprint measured here rather than the run's own random grid: the
   * same rule, on a coarser count of the footprint, so a cell or so apart
   * at most. */
  const double filled = (double)r.box * r.box * r.box / r.footprint_volume;
  const uint64_t counts[2] = {r.n_tracers, r.n_randoms};
  const uint32_t given[2] = {p->mesh.n_cells, p->mesh.n_cells_random};
  uint32_t* cells[2] = {&plan->mesh_cells, &plan->random_mesh_cells};
  for (int i = 0; i < 2; i++)
    *cells[i] = given[i] ? given[i]
                         : sif_finder_suggest_mesh_cells(
                             (uint64_t)ceil((double)counts[i] * filled),
                             (sif_real)r.box, r.r_max);
  if (plan->mesh_cells == 0 || plan->random_mesh_cells == 0) {
    SIF_LOG_ERROR(TAG,
      "the largest radius, %g, needs a search sphere wider than the box",
      (double)r.r_max);
    return SIF_ERR_INVALID;
  }
  status = check_survey_meshes(p, &r);
  if (status != SIF_OK)
    return status;

  plan->peak_bytes = pipeline_survey_peak_bytes(r.n_tracers, r.weighted,
    r.n_randoms, r.randoms_weighted, r.grid_cells, plan->mesh_cells,
    plan->random_mesh_cells);
  return SIF_OK;
}

int pipeline_survey(const exodus_params_t* p) {
  sif_field_t* data = NULL;
  sif_field_t* randoms = NULL;
  sif_grid_t* data_grid = NULL;
  sif_grid_t* random_grid = NULL;
  sif_chain_mesh_t* data_mesh = NULL;
  sif_chain_mesh_t* random_mesh = NULL;
  sif_catalogue_t* cat = NULL;
  sif_real* radii = NULL;
  run_facts_t r = {0};
  sif_timer_t t_total, t;
  int status;

  if (!p || !p->input.path || !p->randoms.path || !p->output.path ||
      !p->finder.radii || p->finder.n_radii == 0) {
    SIF_LOG_ERROR(TAG, "incomplete parameters");
    return SIF_ERR_INVALID;
  }
  if (p->survey.sky && !p->survey.has_cosmology) {
    SIF_LOG_ERROR(TAG, "voids on the sky need a cosmology");
    return SIF_ERR_INVALID;
  }

  const sif_option search = search_option(p->finder.search_factor, &status);
  if (status != SIF_OK)
    return status;

  sif_timer_start(&t_total);

  /* 1-2. Fields, footprint, box. */
  sif_real offset[3], box;
  status = survey_prepare(p, search, &data, &randoms, &r, &radii, offset, &box);
  if (status != SIF_OK)
    goto done;

  status = sif_field_translate(data, offset);
  if (status == SIF_OK)
    status = sif_field_translate(randoms, offset);
  if (status != SIF_OK)
    goto done;
  SIF_LOG_INFO(TAG,
    "box: footprint %.4g (Mpc/h)^3, mean separation %g; searched in a box of "
    "%g, moved by (%g, %g, %g)",
    r.footprint_volume, r.mps, (double)box, (double)offset[0],
    (double)offset[1], (double)offset[2]);

  /* 3. Grids. Densities both: the finder compares them itself. */
  sif_timer_start(&t);
  data_grid = sif_grid_alloc(r.grid_cells, box);
  random_grid = sif_grid_alloc(r.grid_cells, box);
  if (!data_grid || !random_grid) {
    status = SIF_ERR_ALLOC;
    goto done;
  }
  status = sif_grid_assign_cic(data_grid, data);
  if (status == SIF_OK)
    status = sif_grid_assign_cic(random_grid, randoms);
  if (status != SIF_OK)
    goto done;
  SIF_LOG_INFO(TAG, "grids: 2 x %u^3 cells of %g%s, %.2f GiB (%.2f s)",
    r.grid_cells, (double)data_grid->cell_length,
    p->grid.n_cells ? "" : " (one per mean separation)",
    2.0 * (double)data_grid->total_cells * sizeof(sif_real) /
      (1024.0 * 1024.0 * 1024.0),
    seconds_since(&t));

  /* 4. Meshes, each sized for its own density inside the footprint, which
   * is read off the random grid. They take the fields' storage over. */
  sif_timer_start(&t);
  r.mesh_cells = p->mesh.n_cells ? p->mesh.n_cells
                                 : sif_finder_suggest_mesh_cells_survey(
                                     r.n_tracers, random_grid, r.r_max);
  r.random_mesh_cells =
    p->mesh.n_cells_random
      ? p->mesh.n_cells_random
      : sif_finder_suggest_mesh_cells_survey(r.n_randoms, random_grid, r.r_max);
  if (r.mesh_cells == 0 || r.random_mesh_cells == 0) {
    SIF_LOG_ERROR(TAG,
      "the largest radius, %g, needs a search sphere wider than the box",
      (double)r.r_max);
    status = SIF_ERR_INVALID;
    goto done;
  }

  status = check_survey_meshes(p, &r);
  if (status != SIF_OK)
    goto done;

  data_mesh = sif_chain_mesh_alloc_consume(
    r.mesh_cells, box, data, SIF_MESH_DROP_INDICES);
  sif_field_free(data);
  data = NULL;
  if (!data_mesh) {
    status = SIF_ERR_ALLOC;
    goto done;
  }
  random_mesh = sif_chain_mesh_alloc_consume(
    r.random_mesh_cells, box, randoms, SIF_MESH_DROP_INDICES);
  sif_field_free(randoms);
  randoms = NULL;
  if (!random_mesh) {
    status = SIF_ERR_ALLOC;
    goto done;
  }
  SIF_LOG_INFO(TAG,
    "meshes: %u^3 cells for the data, %u^3 for the randoms "
    "(%.2f s)",
    r.mesh_cells, r.random_mesh_cells, seconds_since(&t));

  /* 5. Finder. */
  sif_timer_start(&t);
  cat = sif_finder_exodus_survey(data_grid, random_grid, data_mesh, random_mesh,
    radii, p->finder.n_radii, (sif_real)p->finder.threshold,
    (sif_real)p->finder.overlap_fraction, SIF_FINDER_CONSUME_GRID | search);
  sif_chain_mesh_free(random_mesh);
  sif_chain_mesh_free(data_mesh);
  random_mesh = data_mesh = NULL;
  sif_grid_free(random_grid);
  sif_grid_free(data_grid);
  random_grid = data_grid = NULL;
  if (!cat) {
    status = SIF_ERR_INVALID;
    goto done;
  }
  SIF_LOG_INFO(
    TAG, "finder: %" PRIu64 " voids (%.2f s)", cat->n_voids, seconds_since(&t));

  /* 6. Catalogue: out of the box into the inputs' frame -- the observer at
   * the origin -- and onto the sky, if asked. */
  sif_timer_start(&t);
  const sif_real back[3] = {-offset[0], -offset[1], -offset[2]};
  status = sif_catalogue_translate(cat, back);
  if (status == SIF_OK && p->survey.sky)
    status = sif_catalogue_to_sky(cat, &p->survey.cosmology);
  if (status != SIF_OK)
    goto done;
  status = write_output(p, &r, cat);
  if (status != SIF_OK)
    goto done;
  SIF_LOG_INFO(TAG, "catalogue: written to %s%s (%.2f s)", p->output.path,
    p->survey.sky ? ", on the sky" : "", seconds_since(&t));

  SIF_LOG_INFO(TAG, "done in %.2f s", seconds_since(&t_total));

done:
  free(radii);
  sif_catalogue_free(cat);
  sif_chain_mesh_free(random_mesh);
  sif_chain_mesh_free(data_mesh);
  sif_grid_free(random_grid);
  sif_grid_free(data_grid);
  sif_field_free(randoms);
  sif_field_free(data);
  return status;
}
