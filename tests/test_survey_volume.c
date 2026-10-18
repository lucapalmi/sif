/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * sif_survey_volume() against footprints of known volume: a sphere, a shell
 * and a wedge whose randoms follow a steep n(z), and the wedge again with
 * star-mask holes. Then the observer argument, independence from the thread
 * count, and the refusals.
 */
#include "sif/measure/survey_volume.h"

#include "sif/core/system.h"
#include "sif/structures/field.h"
#include "sif/utils/random.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _OPENMP
#  include <omp.h>
#endif

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define PI 3.14159265358979323846

/* At this count the estimate scatters by about 0.07%; the tolerance is
 * several times that, and far below the few per cent the cell-counting
 * estimate it replaces was off by. */
#define N_RANDOMS SIF_TEST_SCALE(200000)
#define TOLERANCE (SIF_TEST_INSTRUMENTED ? 0.03 : 0.005)

typedef struct {
  double r1, r2;      /* radial range */
  double cos_max;     /* cap: directions with z / r >= cos_max */
  double peak, width; /* n(r) ~ exp(-((r - peak) / width)^2); width 0: flat */
  int n_holes;
  double hole_cos; /* each hole: a cap of this cos(half-angle) */
  double holes[8][3];
} shape_t;

static double profile(const shape_t* s, double r) {
  if (s->width <= 0.0)
    return 1.0;
  const double t = (r - s->peak) / s->width;
  return exp(-t * t);
}

static int in_hole(const shape_t* s, const double u[3]) {
  for (int h = 0; h < s->n_holes; h++)
    if (u[0] * s->holes[h][0] + u[1] * s->holes[h][1] + u[2] * s->holes[h][2] >
        s->hole_cos)
      return 1;
  return 0;
}

/* The holes are disjoint and lie wholly inside the cap, so each takes away
 * a cap of its own. */
static double shape_volume(const shape_t* s) {
  const double omega =
    2.0 * PI * (1.0 - s->cos_max) - s->n_holes * 2.0 * PI * (1.0 - s->hole_cos);
  return omega * (pow(s->r2, 3) - pow(s->r1, 3)) / 3.0;
}

static sif_field_t* shape_randoms(
  const shape_t* s, uint64_t n, uint64_t seed, const double shift[3]) {

  sif_prng_state_t rng;
  sif_prng_init(&rng, seed);

  /* Density per unit volume ~ profile(r), so radii are drawn from
   * profile(r) r^2, by rejection against its maximum on a fine grid. */
  double p_max = 0.0;
  for (int i = 0; i <= 1000; i++) {
    const double r = s->r1 + (s->r2 - s->r1) * i / 1000.0;
    p_max = fmax(p_max, profile(s, r) * r * r);
  }

  sif_real* x = malloc(n * sizeof(sif_real));
  sif_real* y = malloc(n * sizeof(sif_real));
  sif_real* z = malloc(n * sizeof(sif_real));
  for (uint64_t i = 0; i < n;) {
    const double mu =
      s->cos_max + (1.0 - s->cos_max) * sif_prng_next_double(&rng);
    const double phi = 2.0 * PI * sif_prng_next_double(&rng);
    const double st = sqrt(1.0 - mu * mu);
    const double u[3] = {st * cos(phi), st * sin(phi), mu};
    if (in_hole(s, u))
      continue;

    double r;
    do {
      r = s->r1 + (s->r2 - s->r1) * sif_prng_next_double(&rng);
    } while (sif_prng_next_double(&rng) * p_max > profile(s, r) * r * r);

    x[i] = (sif_real)(r * u[0] + shift[0]);
    y[i] = (sif_real)(r * u[1] + shift[1]);
    z[i] = (sif_real)(r * u[2] + shift[2]);
    i++;
  }

  sif_field_t* f = sif_field_alloc(n);
  if (!f || sif_field_assign_positions(f, x, y, z) != SIF_OK) {
    sif_field_free(f);
    f = NULL;
  }
  free(x);
  free(y);
  free(z);
  return f;
}

static const double no_shift[3] = {0.0, 0.0, 0.0};

static shape_t wedge_with_holes(void) {
  /* Six 1.5-degree holes around a 20-degree cap, 10 degrees off its axis
   * and 60 degrees apart: disjoint, and clear of the cap's edge. */
  shape_t s = {200.0, 600.0, cos(20.0 * PI / 180.0), 350.0, 150.0, 6,
    cos(1.5 * PI / 180.0), {{0}}};
  const double t = 10.0 * PI / 180.0;
  for (int h = 0; h < 6; h++) {
    const double phi = h * PI / 3.0;
    s.holes[h][0] = sin(t) * cos(phi);
    s.holes[h][1] = sin(t) * sin(phi);
    s.holes[h][2] = cos(t);
  }
  return s;
}

static void run_shapes(void) {
  printf("known volumes\n");

  const struct {
    const char* name;
    shape_t s;
  } cases[] = {
    {"sphere", {0.0, 100.0, -1.0, 0.0, 0.0, 0, 0.0, {{0}}}},
    {"shell, steep n(z)", {50.0, 150.0, -1.0, 90.0, 40.0, 0, 0.0, {{0}}}},
    {"wedge, n(z)",
      {200.0, 600.0, cos(20.0 * PI / 180.0), 350.0, 150.0, 0, 0.0, {{0}}}},
    {"wedge with holes", wedge_with_holes()},
  };

  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    sif_field_t* f = shape_randoms(&cases[c].s, N_RANDOMS, 11 + c, no_shift);
    CHECK(f, "%s: field construction failed", cases[c].name);
    if (!f)
      continue;

    sif_real v = 0.0f;
    const int status = sif_survey_volume(f, NULL, &v);
    const double want = shape_volume(&cases[c].s);
    const double err = (double)v / want - 1.0;
    CHECK(status == SIF_OK && fabs(err) < TOLERANCE,
      "%s: status %d, volume %g against %g (%+.3f%%)", cases[c].name, status,
      (double)v, want, 100.0 * err);
    printf("  %-18s %+.3f%%\n", cases[c].name, 100.0 * err);
    sif_field_free(f);
  }
}

/* The same randoms moved by an offset, with the observer moved with them. */
static void run_observer(void) {
  printf("observer\n");

  const shape_t s = wedge_with_holes();
  const double shift[3] = {700.0, 650.0, 640.0};
  const sif_real observer[3] = {700.0f, 650.0f, 640.0f};

  sif_field_t* at_origin = shape_randoms(&s, N_RANDOMS, 5, no_shift);
  sif_field_t* moved = shape_randoms(&s, N_RANDOMS, 5, shift);
  sif_real v0 = 0.0f, v1 = 0.0f;
  CHECK(at_origin && moved &&
          sif_survey_volume(at_origin, NULL, &v0) == SIF_OK &&
          sif_survey_volume(moved, observer, &v1) == SIF_OK,
    "a run failed: '%s'", sif_error_message());
  /* Only the rounding of the shifted positions differs. */
  CHECK(fabs((double)v1 / (double)v0 - 1.0) < 1e-3,
    "moved with its observer: %g against %g", (double)v1, (double)v0);

  sif_field_free(at_origin);
  sif_field_free(moved);
  printf("  ok\n");
}

static void run_thread_independence(void) {
#ifdef _OPENMP
  printf("thread count\n");

  const shape_t s = wedge_with_holes();
  sif_field_t* f = shape_randoms(&s, N_RANDOMS, 3, no_shift);
  const int threads = omp_get_max_threads();

  sif_real many = 0.0f, one = 0.0f;
  CHECK(f && sif_survey_volume(f, NULL, &many) == SIF_OK, "a run failed: '%s'",
    sif_error_message());
  omp_set_num_threads(1);
  CHECK(f && sif_survey_volume(f, NULL, &one) == SIF_OK, "a run failed: '%s'",
    sif_error_message());
  omp_set_num_threads(threads);
  CHECK(memcmp(&many, &one, sizeof(sif_real)) == 0,
    "%d threads give %.9g, one gives %.9g", threads, (double)many, (double)one);

  sif_field_free(f);
  printf("  ok\n");
#endif
}

/*
 * sif_survey_volume() with the library logging warnings, and what it printed
 * to stderr captured into log.
 */
static int volume_logged(
  const sif_field_t* f, sif_real* v, char* log, size_t size) {

  sif_config_t warn = {
    .fft_config = NULL, .omp_config = NULL, .log_level = SIF_LOG_LEVEL_WARNING};
  sif_config_t quiet = warn;
  quiet.log_level = SIF_LOG_LEVEL_ERROR;

  log[0] = '\0';
  FILE* cap = tmpfile();
  if (!cap)
    return SIF_ERR_IO;
  fflush(stderr);
  const int saved = dup(2);
  dup2(fileno(cap), 2);

  sif_system_finalise();
  int status = sif_system_init(&warn);
  if (status == SIF_OK)
    status = sif_survey_volume(f, NULL, v);
  fflush(stderr);
  sif_system_finalise();
  if (sif_system_init(&quiet) != SIF_OK)
    status = SIF_ERR_UNINITIALISED;

  dup2(saved, 2);
  close(saved);
  rewind(cap);
  const size_t n = fread(log, 1, size - 1, cap);
  log[n] = '\0';
  fclose(cap);
  return status;
}

/* Separable footprints, however steep their n(z) or holed their mask, must
 * raise no warning. */
static void run_no_false_alarm(void) {
  printf("separable: no warning\n");

  const shape_t shell = {50.0, 150.0, -1.0, 90.0, 40.0, 0, 0.0, {{0}}};
  const shape_t holed = wedge_with_holes();
  const shape_t* shapes[2] = {&shell, &holed};
  char log[4096];

  for (int c = 0; c < 2; c++) {
    sif_field_t* f = shape_randoms(shapes[c], N_RANDOMS, 31 + c, no_shift);
    sif_real v = 0.0f;
    const int status = volume_logged(f, &v, log, sizeof(log));
    CHECK(status == SIF_OK && !strstr(log, "radial range differs"),
      "shape %d: status %d, logged '%s'", c, status, log);
    sif_field_free(f);
  }
  printf("  ok\n");
}

/* A footprint whose two halves reach different distances: still measured,
 * with a warning that says where. */
static void run_non_separable(void) {
  printf("non-separable\n");

  const shape_t near = {
    100.0, 300.0, cos(15.0 * PI / 180.0), 0.0, 0.0, 0, 0.0, {{0}}};
  shape_t far = near;
  far.r2 = 400.0;

  sif_field_t* a = shape_randoms(&near, N_RANDOMS / 2, 21, no_shift);
  sif_field_t* b = shape_randoms(&far, N_RANDOMS / 2, 22, no_shift);
  sif_field_t* f = sif_field_alloc(N_RANDOMS);
  CHECK(a && b && f, "field construction failed");
  if (a && b && f) {
    /* The far one mirrored to the other side of the sky. */
    sif_real* x = malloc(N_RANDOMS * sizeof(sif_real));
    sif_real* y = malloc(N_RANDOMS * sizeof(sif_real));
    sif_real* z = malloc(N_RANDOMS * sizeof(sif_real));
    const uint64_t h = N_RANDOMS / 2;
    memcpy(x, a->x, h * sizeof(sif_real));
    memcpy(y, a->y, h * sizeof(sif_real));
    memcpy(z, a->z, h * sizeof(sif_real));
    for (uint64_t i = 0; i < N_RANDOMS - h; i++) {
      x[h + i] = b->x[i];
      y[h + i] = b->y[i];
      z[h + i] = -b->z[i];
    }
    CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
      "assigning positions failed");
    free(x);
    free(y);
    free(z);

    sif_real v = 0.0f;
    char log[4096];
    const int status = volume_logged(f, &v, log, sizeof(log));
    CHECK(status == SIF_OK && v > 0.0f,
      "a non-separable footprint should still be measured: '%s'",
      sif_error_message());
    /* The short half ends at 300; what the warning reports is the farthest
     * random found there, which falls short of it by the gap between
     * randoms -- so how close it comes scales with their density, and the
     * sanitizer-scaled count lands further off than the full one. */
    const char* found = strstr(log, "found ");
    double r_found = 0.0;
    if (found)
      r_found = strtod(found + strlen("found "), NULL);
    CHECK(strstr(log, "radial range differs across the sky") &&
            strstr(log, "end short of r_max") &&
            fabs(r_found - 300.0) < 0.1 * TOLERANCE * 300.0,
      "expected a warning naming the short half, logged '%s'", log);
  }

  sif_field_free(a);
  sif_field_free(b);
  sif_field_free(f);
  printf("  ok\n");
}

static void run_refusals(void) {
  printf("refusals\n");

  sif_real v = 0.0f;
  sif_error_clear();
  CHECK(sif_survey_volume(NULL, NULL, &v) == SIF_ERR_INVALID &&
          error_has("randoms: NULL"),
    "NULL randoms: '%s'", sif_error_message());

  const shape_t s = {0.0, 100.0, -1.0, 0.0, 0.0, 0, 0.0, {{0}}};
  sif_field_t* f = shape_randoms(&s, 1000, 1, no_shift);

  f->units = SIF_COORDINATES_SKY;
  sif_error_clear();
  CHECK(sif_survey_volume(f, NULL, &v) == SIF_ERR_INVALID &&
          error_has("sky coordinates"),
    "sky coordinates: '%s'", sif_error_message());
  f->units = SIF_COORDINATES_CARTESIAN;

  const sif_real keep = f->x[7];
  f->x[7] = (sif_real)NAN;
  sif_error_clear();
  CHECK(sif_survey_volume(f, NULL, &v) == SIF_ERR_INVALID &&
          error_has("1 of 1000 with a position that is not finite"),
    "NaN position: '%s'", sif_error_message());
  f->x[7] = keep;

  const sif_real bad_observer[3] = {0.0f, (sif_real)INFINITY, 0.0f};
  sif_error_clear();
  CHECK(sif_survey_volume(f, bad_observer, &v) == SIF_ERR_INVALID &&
          error_has("observer[1]: inf"),
    "infinite observer: '%s'", sif_error_message());
  sif_field_free(f);

  /* Too few to find any cell wholly inside the mask. */
  f = shape_randoms(&s, 20, 2, no_shift);
  sif_error_clear();
  CHECK(sif_survey_volume(f, NULL, &v) == SIF_ERR_INVALID &&
          error_has("too few to resolve the mask"),
    "20 randoms: '%s'", sif_error_message());
  sif_field_free(f);

  printf("  ok\n");
}

int main(void) {
  sif_config_t cfg = {
    .fft_config = NULL, .omp_config = NULL, .log_level = SIF_LOG_LEVEL_ERROR};
  if (sif_system_init(&cfg) != SIF_OK) {
    printf("FAIL: sif_system_init\n");
    return 1;
  }

  run_shapes();
  run_observer();
  run_thread_independence();
  run_no_false_alarm();
  run_non_separable();
  run_refusals();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");

  sif_system_finalise();
  return failures != 0;
}
