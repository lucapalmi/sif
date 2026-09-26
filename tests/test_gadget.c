/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * GADGET snapshots, against the synthetic files in tests/data/gadget.
 *
 * The fixtures were written by make_fixtures.py from the GADGET-4 manual, not
 * by sif, and hold the same particles in every format -- SnapFormat 1 with
 * both headers and both precisions, SnapFormat 2 byte-swapped, HDF5 in two
 * layouts -- split differently over their files. So every format is checked
 * against the same closed-form values, and against each other.
 *
 * Without HDF5, the HDF5 fixtures are checked to be refused as unsupported
 * rather than misread.
 */
#include "sif/core/system.h"
#include "sif/io/gadget_io.h"
#include "sif/structures/field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#ifndef SIF_TEST_DATA_DIR
#  error "SIF_TEST_DATA_DIR must point at tests/data"
#endif
#define DATA SIF_TEST_DATA_DIR "/gadget/"

/* What make_fixtures.py wrote; see its docstring. */
static const uint64_t N_TOTAL[3] = {40, 100, 30};
static const double MASS_TABLE[3] = {0.0, 0.5, 0.0};
#define TIME 0.25
#define BOX  10000.0

static double x_of(uint32_t t, uint64_t g) {
  return 100.0 * t + 0.5 * (double)g + 0.125;
}
static double vx_of(uint32_t t, uint64_t g) {
  return 10.0 * t + 0.25 * (double)g;
}
static double m_of(uint64_t g) { return 1.0 + (double)g / 1024.0; }

typedef struct {
  const char* name;
  const char* path;
  sif_gadget_format_t format;
  int is_hdf5;
  int is_legacy_header;
  int is_swapped;
  uint32_t precision;
  uint32_t n_files;
  int has_cosmology;
  int has_unit;
  /* NTYPES=2 and only type 1 populated, like a dark-matter-only GADGET-4 run.
   * Its type 1 is every other fixture's type 1. */
  int dm_only;
} fixture_t;

static const fixture_t FIXTURES[] = {
  {"f1 legacy", DATA "f1_legacy/snap_005.0", SIF_GADGET_FORMAT_1, 0, 1, 0, 4, 3,
    1, 0, 0},
  {"f1 g4 double", DATA "f1_g4/snapdir_005/snap_005.0", SIF_GADGET_FORMAT_1, 0,
    0, 0, 8, 2, 0, 0, 0},
  {"f1 g4 u32", DATA "f1_g4_u32/snap_005", SIF_GADGET_FORMAT_1, 0, 0, 0, 4, 1,
    0, 0, 0},
  {"f2 swapped", DATA "f2_swapped/snap_005.0", SIF_GADGET_FORMAT_2, 0, 1, 1, 4,
    3, 1, 0, 0},
  {"hdf5 g4", DATA "hdf5_g4/snap_005.0.hdf5", SIF_GADGET_FORMAT_HDF5, 1, 0, 0,
    4, 3, 1, 1, 0},
  {"hdf5 legacy", DATA "hdf5_legacy/snap_005.hdf5", SIF_GADGET_FORMAT_HDF5, 1,
    0, 0, 8, 1, 1, 1, 0},
  {"f1 g4 ntypes2", DATA "f1_g4_ntypes2/snapdir_005/snap_005.0",
    SIF_GADGET_FORMAT_1, 0, 0, 0, 4, 2, 0, 0, 1},
};
#define N_FIXTURES (sizeof(FIXTURES) / sizeof(FIXTURES[0]))

static uint32_t fx_types(const fixture_t* fx) { return fx->dm_only ? 2 : 6; }

static uint64_t fx_total(const fixture_t* fx, uint32_t t) {
  if (fx->dm_only)
    return t == 1 ? N_TOTAL[1] : 0;
  return t < 3 ? N_TOTAL[t] : 0;
}

static double fx_mass(const fixture_t* fx, uint32_t t) {
  if (fx->dm_only)
    return t == 1 ? MASS_TABLE[1] : 0.0;
  return t < 3 ? MASS_TABLE[t] : 0.0;
}

static int available(const fixture_t* fx) {
#ifdef SIF_HAVE_HDF5
  (void)fx;
  return 1;
#else
  return !fx->is_hdf5;
#endif
}

static int read_all(const char* path, sif_gadget_ptype_t ptype,
  sif_gadget_velocity_t vel, sif_gadget_mass_t mass, sif_gadget_length_t len,
  double fraction, uint64_t seed, sif_field_t** out, double* box) {
  return sif_field_read_gadget(path, SIF_GADGET_FORMAT_AUTO, ptype, vel, mass,
    len, fraction, seed, out, box);
}

/* --- headers --- */

static void test_headers(void) {
  printf("headers\n");

  for (size_t i = 0; i < N_FIXTURES; i++) {
    const fixture_t* fx = &FIXTURES[i];
    sif_gadget_header_t h;
    const int status =
      sif_gadget_read_header(fx->path, SIF_GADGET_FORMAT_AUTO, &h);

    if (!available(fx)) {
      CHECK(status == SIF_ERR_UNSUPPORTED,
        "%s: expected UNSUPPORTED without HDF5, got %d", fx->name, status);
      continue;
    }

    CHECK(status == SIF_OK, "%s: read_header returned %d", fx->name, status);
    if (status != SIF_OK)
      continue;

    CHECK(h.format == fx->format, "%s: format %d", fx->name, (int)h.format);
    CHECK(h.n_types == fx_types(fx), "%s: %u types", fx->name, h.n_types);
    CHECK(h.n_files == fx->n_files, "%s: %u files", fx->name, h.n_files);
    CHECK(
      h.precision == fx->precision, "%s: precision %u", fx->name, h.precision);
    CHECK(h.time == TIME && h.redshift == 3.0 && h.box_size == BOX,
      "%s: time %g, z %g, box %g", fx->name, h.time, h.redshift, h.box_size);
    CHECK(h.has_cosmology == fx->has_cosmology, "%s: cosmology %d", fx->name,
      h.has_cosmology);
    if (h.has_cosmology)
      CHECK(h.omega0 == 0.3 && h.omega_lambda == 0.7 && h.hubble_param == 0.7,
        "%s: cosmology values", fx->name);
    CHECK((h.unit_length_in_cm > 0) == fx->has_unit, "%s: unit %g", fx->name,
      h.unit_length_in_cm);

    if (!fx->is_hdf5) {
      CHECK(h.is_legacy_header == fx->is_legacy_header, "%s: legacy %d",
        fx->name, h.is_legacy_header);
      CHECK(h.is_swapped == fx->is_swapped, "%s: swapped %d", fx->name,
        h.is_swapped);
    }

    for (uint32_t t = 0; t < fx_types(fx); t++) {
      CHECK(h.n_part_total[t] == fx_total(fx, t), "%s: type %u total %" PRIu64,
        fx->name, t, h.n_part_total[t]);
      CHECK(h.mass_table[t] == fx_mass(fx, t), "%s: type %u mass %g", fx->name,
        t, h.mass_table[t]);
    }

    /* The labelled formats name their blocks; SnapFormat 1 only counts. */
    if (fx->format == SIF_GADGET_FORMAT_2)
      CHECK(h.n_blocks == 6 && strcmp(h.blocks[1], "POS") == 0 &&
              strcmp(h.blocks[4], "MASS") == 0,
        "%s: blocks", fx->name);
    else if (fx->format == SIF_GADGET_FORMAT_1)
      /* HEAD POS VEL ID, then MASS and U where there is gas or a type with
       * individual masses. */
      CHECK(h.n_blocks == (fx->dm_only ? 4u : 6u) && h.blocks[0][0] == '\0',
        "%s: %u blocks", fx->name, h.n_blocks);
    else {
      int has_coords = 0, has_masses = 0;
      for (uint32_t b = 0; b < h.n_blocks; b++) {
        has_coords |= strcmp(h.blocks[b], "Coordinates") == 0;
        has_masses |= strcmp(h.blocks[b], "Masses") == 0;
      }
      CHECK(has_coords && has_masses, "%s: datasets", fx->name);
    }

    /* The printed form carries the format name. */
    FILE* tmp = tmpfile();
    if (tmp) {
      sif_gadget_print_header(&h, tmp);
      rewind(tmp);
      char buf[4096];
      const size_t n = fread(buf, 1, sizeof(buf) - 1, tmp);
      buf[n] = '\0';
      fclose(tmp);
      CHECK(strstr(buf, fx->is_hdf5 ? "HDF5" : "SnapFormat") != NULL &&
              strstr(buf, fx->dm_only ? "0.5" : "per particle") != NULL,
        "%s: printed header:\n%s", fx->name, buf);
    }
  }
}

/* --- full reads --- */

/* Every particle of every type, in every format, against the closed form. */
static void test_full_reads(void) {
  printf("full reads\n");

  for (size_t i = 0; i < N_FIXTURES; i++) {
    const fixture_t* fx = &FIXTURES[i];
    if (!available(fx))
      continue;

    for (uint32_t t = 0; t < 3; t++) {
      sif_field_t* f = NULL;
      double box = 0;
      const int status = read_all(fx->path,
        (sif_gadget_ptype_t)(SIF_GADGET_PTYPE_0 + t), SIF_GADGET_VELOCITY_RAW,
        SIF_GADGET_MASS_READ, SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, &box);

      /* A type the snapshot has none of -- or, with NTYPES=2, does not have
       * at all -- is refused rather than read as empty. */
      if (fx_total(fx, t) == 0) {
        CHECK(status == SIF_ERR_INVALID && !f, "%s type %u: status %d",
          fx->name, t, status);
        sif_field_free(f);
        continue;
      }

      CHECK(
        status == SIF_OK && f, "%s type %u: status %d", fx->name, t, status);
      if (!f)
        continue;

      CHECK(f->n_particles == N_TOTAL[t], "%s type %u: %" PRIu64 " particles",
        fx->name, t, f->n_particles);
      CHECK(box == (double)BOX * 1e-3, "%s: box %g", fx->name, box);
      CHECK(f->vx != NULL, "%s type %u: no velocities", fx->name, t);
      CHECK((f->weights != NULL) == (fx_mass(fx, t) == 0),
        "%s type %u: weights %p", fx->name, t, (void*)f->weights);

      uint64_t bad = 0;
      for (uint64_t g = 0; g < f->n_particles && f->vx; g++) {
        const double x = x_of(t, g), vx = vx_of(t, g);
        bad += f->x[g] != (sif_real)(x * 1e-3);
        bad += f->y[g] != (sif_real)((x + 1000.0) * 1e-3);
        bad += f->z[g] != (sif_real)((x + 2000.0) * 1e-3);
        bad += f->vx[g] != (sif_real)vx;
        bad += f->vy[g] != (sif_real)(-vx);
        bad += f->vz[g] != (sif_real)(vx + 1.0);
        if (f->weights)
          bad += f->weights[g] != (sif_real)m_of(g);
      }
      CHECK(bad == 0, "%s type %u: %" PRIu64 " wrong values", fx->name, t, bad);
      sif_field_free(f);
    }
  }
}

/* --- options --- */

static void test_options(void) {
  printf("options\n");
  const char* path = DATA "f2_swapped/snap_005";
  sif_field_t* f = NULL;
  double box = 0;

  /* Peculiar velocities are u * sqrt(a), a = 0.25. */
  int status = read_all(path, SIF_GADGET_PTYPE_2, SIF_GADGET_VELOCITY_PECULIAR,
    SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_MPC, 1.0, 0, &f, &box);
  CHECK(status == SIF_OK && f, "peculiar: status %d", status);
  if (f) {
    CHECK(f->weights == NULL, "MASS_SKIP still read weights");
    CHECK(box == BOX, "MPC converted the box: %g", box);
    uint64_t bad = 0;
    for (uint64_t g = 0; g < f->n_particles; g++) {
      bad += f->x[g] != (sif_real)x_of(2, g);
      bad += f->vx[g] != (sif_real)(0.5 * vx_of(2, g));
    }
    CHECK(bad == 0, "peculiar/MPC: %" PRIu64 " wrong values", bad);
    sif_field_free(f);
    f = NULL;
  }

  /* No velocities asked for, none reserved. */
  status = read_all(path, SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
    SIF_GADGET_MASS_READ, SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, NULL);
  CHECK(status == SIF_OK && f && !f->vx && !f->weights,
    "skip velocities: status %d", status);
  sif_field_free(f);
  f = NULL;

  /* A binary file cannot say what its unit is. */
  status = read_all(path, SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
    SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_AUTO, 1.0, 0, &f, NULL);
  CHECK(status == SIF_ERR_INVALID && !f, "AUTO on binary: status %d", status);

#ifdef SIF_HAVE_HDF5
  /* An HDF5 file can: kpc/h, found in /Parameters and in /Units. */
  const char* h5[2] = {DATA "hdf5_g4/snap_005", DATA "hdf5_legacy/snap_005"};
  for (int i = 0; i < 2; i++) {
    status = read_all(h5[i], SIF_GADGET_PTYPE_0, SIF_GADGET_VELOCITY_SKIP,
      SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_AUTO, 1.0, 0, &f, &box);
    CHECK(status == SIF_OK && f, "AUTO on %s: status %d", h5[i], status);
    if (f) {
      CHECK(fabs(box - 10.0) < 1e-9, "AUTO box %.12g", box);
      double worst = 0;
      for (uint64_t g = 0; g < f->n_particles; g++)
        worst = fmax(worst, fabs(f->x[g] - x_of(0, g) * 1e-3));
      CHECK(worst < 1e-5, "AUTO positions off by %g", worst);
      sif_field_free(f);
      f = NULL;
    }
  }
#endif
}

/* --- finding the files --- */

static void test_paths(void) {
  printf("paths\n");

  typedef struct {
    const char* path;
    int is_hdf5;
  } case_t;
  const case_t cases[] = {
    {DATA "f1_legacy/snap_005", 0},           /* base -> .0 */
    {DATA "f1_legacy/snap_005.2", 0},         /* any file of the set */
    {DATA "f1_g4/snap_005", 0},               /* base -> snapdir */
    {DATA "f1_g4/snapdir_005/snap_005.1", 0}, /* inside the snapdir */
    {DATA "f1_g4_u32/snap_005", 0},           /* single, unnumbered */
    {DATA "hdf5_g4/snap_005", 1},             /* base -> .0.hdf5 */
    {DATA "hdf5_g4/snap_005.1.hdf5", 1},      /* any file of the set */
    {DATA "hdf5_legacy/snap_005", 1},         /* base -> .hdf5 */
    /* base name inside the snapdir, as a real run is usually named */
    {DATA "f1_g4_ntypes2/snapdir_005/snap_005", 0},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    sif_field_t* f = NULL;
    const int status =
      read_all(cases[i].path, SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
        SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, NULL);
#ifndef SIF_HAVE_HDF5
    if (cases[i].is_hdf5) {
      CHECK(status == SIF_ERR_UNSUPPORTED && !f,
        "%s: expected UNSUPPORTED, got %d", cases[i].path, status);
      continue;
    }
#endif
    CHECK(status == SIF_OK && f && f->n_particles == N_TOTAL[1],
      "%s: status %d", cases[i].path, status);
    sif_field_free(f);
  }

  sif_field_t* f = NULL;
  int status = read_all(DATA "nothing/snap_005", SIF_GADGET_PTYPE_1,
    SIF_GADGET_VELOCITY_SKIP, SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0,
    0, &f, NULL);
  CHECK(status == SIF_ERR_IO && !f, "missing snapshot: status %d", status);

  /* Asking for a format the file is not in. */
  status = sif_field_read_gadget(DATA "f1_legacy/snap_005", SIF_GADGET_FORMAT_2,
    SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP, SIF_GADGET_MASS_SKIP,
    SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, NULL);
  CHECK(status == SIF_ERR_IO && !f, "wrong format: status %d", status);

  /* A file that is not a snapshot at all. */
  status = read_all(DATA "make_fixtures.py", SIF_GADGET_PTYPE_1,
    SIF_GADGET_VELOCITY_SKIP, SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0,
    0, &f, NULL);
  CHECK(status == SIF_ERR_IO && !f, "not a snapshot: status %d", status);
}

/* --- subsampling --- */

/* The kept particles as global indices, recovered from x. */
static void indices_of(const sif_field_t* f, uint32_t t, uint64_t* out) {
  for (uint64_t k = 0; k < f->n_particles; k++)
    out[k] =
      (uint64_t)llround(((double)f->x[k] * 1e3 - 100.0 * t - 0.125) / 0.5);
}

static void test_subsample(void) {
  printf("subsample\n");
  const uint32_t t = 1;
  const uint64_t n = N_TOTAL[t];

  uint64_t ref[100], other[100];
  sif_field_t* f = NULL;

  int status = read_all(DATA "f1_legacy/snap_005", SIF_GADGET_PTYPE_1,
    SIF_GADGET_VELOCITY_RAW, SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 0.3,
    42, &f, NULL);
  CHECK(status == SIF_OK && f, "subsample: status %d", status);
  if (!f)
    return;

  /* Exactly round(0.3 * 100), in file order, each with its own velocity. */
  CHECK(f->n_particles == 30, "kept %" PRIu64 ", expected 30", f->n_particles);
  indices_of(f, t, ref);
  uint64_t bad = 0;
  for (uint64_t k = 0; k < f->n_particles; k++) {
    bad += ref[k] >= n || (k > 0 && ref[k] <= ref[k - 1]);
    bad += f->vx[k] != (sif_real)vx_of(t, ref[k]);
  }
  CHECK(bad == 0, "subsample: %" PRIu64 " out of order or mismatched", bad);
  const uint64_t n_ref = f->n_particles;
  sif_field_free(f);
  f = NULL;

  /* The same seed picks the same particles from every format, however the
   * snapshot is split over its files. */
  for (size_t i = 0; i < N_FIXTURES; i++) {
    if (!available(&FIXTURES[i]))
      continue;
    status =
      read_all(FIXTURES[i].path, SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
        SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 0.3, 42, &f, NULL);
    CHECK(status == SIF_OK && f && f->n_particles == n_ref, "%s: status %d",
      FIXTURES[i].name, status);
    if (f && f->n_particles == n_ref) {
      indices_of(f, t, other);
      CHECK(memcmp(ref, other, n_ref * sizeof(uint64_t)) == 0,
        "%s: a different subsample for the same seed", FIXTURES[i].name);
    }
    sif_field_free(f);
    f = NULL;
  }

  /* Another seed, another subsample. */
  status = read_all(DATA "f1_legacy/snap_005", SIF_GADGET_PTYPE_1,
    SIF_GADGET_VELOCITY_SKIP, SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 0.3,
    43, &f, NULL);
  if (f) {
    indices_of(f, t, other);
    CHECK(memcmp(ref, other, n_ref * sizeof(uint64_t)) != 0,
      "seeds 42 and 43 drew the same subsample");
    sif_field_free(f);
    f = NULL;
  }

  /* Every particle is equally likely: over many seeds each index is kept
   * about 30% of the time. */
  const int n_seeds = SIF_TEST_SCALE(2000);
  uint32_t hits[100] = {0};
  for (int s = 0; s < n_seeds; s++) {
    status = read_all(DATA "f1_legacy/snap_005", SIF_GADGET_PTYPE_1,
      SIF_GADGET_VELOCITY_SKIP, SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC,
      0.3, (uint64_t)s, &f, NULL);
    if (!f)
      break;
    indices_of(f, t, other);
    for (uint64_t k = 0; k < f->n_particles; k++)
      hits[other[k]]++;
    sif_field_free(f);
    f = NULL;
  }
  /* Binomial(n_seeds, 0.3): 5 sigma either side. */
  const double mean = 0.3 * n_seeds, sd = sqrt(n_seeds * 0.3 * 0.7);
  int outliers = 0;
  for (uint64_t g = 0; g < n; g++)
    outliers += fabs(hits[g] - mean) > 5 * sd;
  CHECK(outliers == 0, "%d indices kept far from 30%% of the time", outliers);
}

/* --- refusals --- */

static void test_refusals(void) {
  printf("refusals\n");
  const char* path = DATA "f1_legacy/snap_005";
  sif_field_t* f = NULL;
  int status;

  /* Two options swapped: the enum types differ, so this takes casts -- which
   * is what a wrong call through an int would look like. */
  status = sif_field_read_gadget(path, SIF_GADGET_FORMAT_AUTO,
    SIF_GADGET_PTYPE_1, (sif_gadget_velocity_t)SIF_GADGET_MASS_READ,
    (sif_gadget_mass_t)SIF_GADGET_VELOCITY_RAW, SIF_GADGET_LENGTH_KPC, 1.0, 0,
    &f, NULL);
  CHECK(status == SIF_ERR_INVALID && !f, "swapped options: status %d", status);

  status = sif_field_read_gadget(path, SIF_GADGET_FORMAT_AUTO,
    (sif_gadget_ptype_t)1, SIF_GADGET_VELOCITY_SKIP, SIF_GADGET_MASS_SKIP,
    SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, NULL);
  CHECK(status == SIF_ERR_INVALID && !f, "bare int type: status %d", status);

  const double fractions[] = {0.0, -0.5, 1.5, NAN, 0.001};
  for (size_t i = 0; i < sizeof(fractions) / sizeof(*fractions); i++) {
    status = read_all(path, SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
      SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, fractions[i], 0, &f, NULL);
    CHECK(status == SIF_ERR_INVALID && !f, "fraction %g: status %d",
      fractions[i], status);
  }

  /* A type the snapshot has none of. */
  status = read_all(path, SIF_GADGET_PTYPE_4, SIF_GADGET_VELOCITY_SKIP,
    SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, NULL);
  CHECK(status == SIF_ERR_INVALID && !f, "empty type: status %d", status);

  status = read_all(path, SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
    SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0, 0, NULL, NULL);
  CHECK(status == SIF_ERR_INVALID, "NULL out_field: status %d", status);
}

/* --- broken snapshots --- */

static int copy_file(const char* from, const char* to, long truncate_to) {
  FILE* in = fopen(from, "rb");
  FILE* out = fopen(to, "wb");
  if (!in || !out) {
    if (in)
      fclose(in);
    if (out)
      fclose(out);
    return 0;
  }
  char buf[8192];
  size_t n;
  long written = 0;
  while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
    if (truncate_to >= 0 && written + (long)n > truncate_to)
      n = (size_t)(truncate_to - written);
    fwrite(buf, 1, n, out);
    written += (long)n;
    if (truncate_to >= 0 && written >= truncate_to)
      break;
  }
  fclose(in);
  fclose(out);
  return 1;
}

static void test_broken(void) {
  printf("broken snapshots\n");
  char from[512], to[64];
  sif_field_t* f = NULL;

  /* One file of three missing. */
  for (int i = 0; i < 2; i++) {
    snprintf(from, sizeof(from), DATA "f1_legacy/snap_005.%d", i);
    snprintf(to, sizeof(to), "test_gadget_snap.%d", i);
    copy_file(from, to, -1);
  }
  remove("test_gadget_snap.2");
  int status =
    read_all("test_gadget_snap", SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
      SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, NULL);
  CHECK(status == SIF_ERR_IO && !f, "missing file: status %d", status);

  /* All three, the last cut off in its positions block. */
  copy_file(DATA "f1_legacy/snap_005.2", "test_gadget_snap.2", 600);
  status =
    read_all("test_gadget_snap", SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
      SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, NULL);
  CHECK(status == SIF_ERR_IO && !f, "truncated file: status %d", status);

  /* A file from another snapshot in the set: the counts no longer add up. */
  copy_file(DATA "f1_legacy/snap_005.1", "test_gadget_snap.2", -1);
  status =
    read_all("test_gadget_snap", SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP,
      SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0, 0, &f, NULL);
  CHECK(status == SIF_ERR_IO && !f, "inconsistent counts: status %d", status);

  for (int i = 0; i < 3; i++) {
    snprintf(to, sizeof(to), "test_gadget_snap.%d", i);
    remove(to);
  }
}

int main(void) {
  if (sif_init(SIF_CONFIG_QUIET) != SIF_OK) {
    printf("FAIL: sif_init\n");
    return 1;
  }

  test_headers();
  test_full_reads();
  test_options();
  test_paths();
  test_subsample();
  test_refusals();
  test_broken();

  sif_finalize();

  if (failures) {
    printf("%d failure(s)\n", failures);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
