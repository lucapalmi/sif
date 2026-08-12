/* Covers the field/grid rework: Morton permutation coherence, the require_*
 * helpers, CIC mass weighting and the box contract. */
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/structures/octree.h"
#include "sif/utils/align.h"

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

#define N_P (SIF_TEST_SCALE(4096) < 512 ? 512 : SIF_TEST_SCALE(4096))
#define BOX 100.0f

static void make_positions(real_t* x, real_t* y, real_t* z) {
  uint64_t s = 99;
  for (uint64_t i = 0; i < N_P; i++) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x[i] = (real_t)((double)((s >> 33) & 0xFFFF) / 65536.0 * (double)BOX);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    y[i] = (real_t)((double)((s >> 33) & 0xFFFF) / 65536.0 * (double)BOX);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    z[i] = (real_t)((double)((s >> 33) & 0xFFFF) / 65536.0 * (double)BOX);
  }
}

/* The headline coherence bug: assigning velocities AFTER a Morton sort used to
 * pair every particle with someone else's velocity. */
static void test_velocity_permutation(void) {
  printf("velocity assignment after morton sort\n");

  real_t *x = malloc(N_P * sizeof(real_t)), *y = malloc(N_P * sizeof(real_t)),
         *z = malloc(N_P * sizeof(real_t));
  make_positions(x, y, z);

  /* Tag each particle's velocity with its position so a mismatch is visible. */
  real_t *vx = malloc(N_P * sizeof(real_t)), *vy = malloc(N_P * sizeof(real_t)),
         *vz = malloc(N_P * sizeof(real_t));
  real_t* m = malloc(N_P * sizeof(real_t));
  for (uint64_t i = 0; i < N_P; i++) {
    vx[i] = x[i];
    vy[i] = y[i];
    vz[i] = z[i];
    m[i] = x[i];
  }

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  CHECK(sif_field_sort_morton(f) == SIF_OK, "sort_morton failed");
  CHECK((f->state_flags & __FIELD_STATE_MORTON_SORTED) != 0, "flag not set");

  /* Assign in ORIGINAL order, after the sort. */
  CHECK(sif_field_assign_velocities(f, vx, vy, vz) == SIF_OK,
    "assign_velocities failed");
  CHECK(sif_field_assign_masses(f, m) == SIF_OK,
    "assign_masses failed");

  int mismatched = 0;
  for (uint64_t i = 0; i < N_P; i++) {
    if (f->vx[i] != f->x[i] || f->vy[i] != f->y[i] || f->vz[i] != f->z[i])
      mismatched++;
    if (f->masses[i] != f->x[i])
      mismatched++;
  }
  CHECK(mismatched == 0, "%d particles carry the wrong velocity/mass",
    mismatched);

  /* The permutation is applied on the way in, so the caller's arrays are left
   * in their original order. */
  int caller_disturbed = 0;
  for (uint64_t i = 0; i < N_P; i++) {
    if (vx[i] != x[i] || vy[i] != y[i] || vz[i] != z[i] || m[i] != x[i])
      caller_disturbed++;
  }
  CHECK(caller_disturbed == 0, "%d caller entries were modified",
    caller_disturbed);

  sif_field_free(f);
  free(x); free(y); free(z); free(vx); free(vy); free(vz); free(m);
  printf("  ok\n");
}

/* Assigning BEFORE the sort must also stay coherent: the sort permutes
 * everything together. */
static void test_sort_permutes_everything(void) {
  printf("morton sort permutes velocities with positions\n");

  real_t *x = malloc(N_P * sizeof(real_t)), *y = malloc(N_P * sizeof(real_t)),
         *z = malloc(N_P * sizeof(real_t));
  make_positions(x, y, z);

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);
  CHECK(sif_field_assign_velocities(f, x, y, z) == SIF_OK,
    "assign_velocities failed");
  CHECK(sif_field_sort_morton(f) == SIF_OK, "sort failed");

  int mismatched = 0;
  for (uint64_t i = 0; i < N_P; i++)
    if (f->vx[i] != f->x[i] || f->vy[i] != f->y[i] || f->vz[i] != f->z[i])
      mismatched++;
  CHECK(mismatched == 0, "%d particles desynced during the sort", mismatched);

  /* original_indices must be a genuine permutation. */
  uint8_t* seen = calloc(N_P, 1);
  int bad = 0;
  for (uint64_t i = 0; i < N_P; i++) {
    uint64_t o = f->original_indices[i];
    if (o >= N_P || seen[o]) bad++;
    else seen[o] = 1;
  }
  CHECK(bad == 0, "original_indices is not a permutation (%d bad)", bad);
  free(seen);

  sif_field_free(f);
  free(x); free(y); free(z);
  printf("  ok\n");
}

static void test_require_helpers(void) {
  printf("require_morton / require_bounds\n");

  real_t *x = malloc(N_P * sizeof(real_t)), *y = malloc(N_P * sizeof(real_t)),
         *z = malloc(N_P * sizeof(real_t));
  make_positions(x, y, z);

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  CHECK(sif_field_require_bounds(f) == SIF_OK, "require_bounds failed");
  CHECK(sif_field_require_bounds(f) == SIF_OK, "require_bounds not idempotent");
  CHECK(f->max_p[0] > f->min_p[0], "bounds look wrong");

  /* The octree used to hard-fail on an unsorted field; it must now sort it. */
  CHECK((f->state_flags & __FIELD_STATE_MORTON_SORTED) == 0,
    "field should not be sorted yet");

  sif_octree_t* tree = sif_octree_alloc(1024);
  CHECK(tree != NULL, "octree alloc failed");
  if (tree) {
    CHECK(sif_octree_build(tree, f, 16) == 0,
      "octree build should succeed on an unsorted field now");
    CHECK((f->state_flags & __FIELD_STATE_MORTON_SORTED) != 0,
      "octree build should have sorted the field");
    sif_octree_free(tree);
  }

  CHECK(sif_field_require_morton(f) == SIF_OK, "require_morton not idempotent");

  sif_field_free(f);
  free(x); free(y); free(z);
  printf("  ok\n");
}

/* CIC must now honour per-particle masses and conserve total mass. */
static void test_cic_mass(void) {
  printf("cic mass weighting and conservation\n");

  real_t *x = malloc(N_P * sizeof(real_t)), *y = malloc(N_P * sizeof(real_t)),
         *z = malloc(N_P * sizeof(real_t));
  make_positions(x, y, z);

  real_t* m = malloc(N_P * sizeof(real_t));
  for (uint64_t i = 0; i < N_P; i++)
    m[i] = 2.0f; /* uniform but != 1, so ignoring masses is detectable */

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);
  sif_field_assign_masses(f, m);

  const uint32_t n = 16;
  sif_grid_t* g = sif_grid_alloc(n, BOX);
  sif_grid_assign_cic(g, f);

  const uint64_t total = (uint64_t)n * n * n;
  const real_t cell_vol = g->cell_length * g->cell_length * g->cell_length;
  double sum = 0.0;
  for (uint64_t i = 0; i < total; i++)
    sum += (double)g->delta[i];
  sum *= (double)cell_vol; /* delta holds density, undo the 1/V */

  const double expected = 2.0 * (double)N_P;
  CHECK(fabs(sum - expected) / expected < 1e-4,
    "mass not conserved: got %.6g, expected %.6g", sum, expected);
  printf("  deposited mass: %.6g (expected %.6g)\n", sum, expected);

  /* Overdensity must average to ~0. */
  sif_grid_compute_overdensity(g);
  double mean = 0.0;
  for (uint64_t i = 0; i < total; i++)
    mean += (double)g->delta[i];
  mean /= (double)total;
  CHECK(fabs(mean) < 1e-3, "mean overdensity is %.3g, expected ~0", mean);

  sif_grid_free(g);
  sif_field_free(f);
  free(x); free(y); free(z); free(m);
  printf("  ok\n");
}

static void test_cic_rejects_out_of_box(void) {
  printf("cic box contract\n");

  real_t *x = malloc(N_P * sizeof(real_t)), *y = malloc(N_P * sizeof(real_t)),
         *z = malloc(N_P * sizeof(real_t));
  make_positions(x, y, z);
  x[7] = BOX + 5.0f; /* used to underflow the local slab index */

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  sif_grid_t* g = sif_grid_alloc(16, BOX);
  sif_grid_assign_cic(g, f);

  double sum = 0.0;
  for (uint64_t i = 0; i < g->total_cells; i++)
    sum += (double)g->delta[i];
  CHECK(sum == 0.0, "an out-of-box particle should abort the assignment");

  /* Freeing a grid whose buffer was taken must not leak the struct. */
  sif_free_aligned(g->delta);
  g->delta = NULL;
  sif_grid_free(g);

  sif_field_free(f);
  free(x); free(y); free(z);
  printf("  ok\n");
}

/*
 * The wrap has to satisfy the binning contract exactly, not approximately: its
 * whole purpose is to turn a field the validators reject into one they accept.
 */
static void test_wrap_periodic(void) {
  printf("periodic wrap\n");

  real_t *x = malloc(N_P * sizeof(real_t)), *y = malloc(N_P * sizeof(real_t)),
         *z = malloc(N_P * sizeof(real_t));
  make_positions(x, y, z);

  /* The rounding artifact this exists for: stored exactly on the edge. */
  x[3] = BOX;
  y[9] = BOX;

  /* A negative small enough that -eps + BOX is not representable and rounds
   * back to BOX. Without the final clamp the wrap would return it still out of
   * range, which is the trap the implementation guards. */
  z[11] = -1e-6f;

  /* Genuinely outside, in both directions. */
  x[21] = BOX * 2.5f;
  y[33] = -BOX * 1.5f;

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  /* A stale bounds cache must not survive the move. */
  CHECK(sif_field_require_bounds(f) == SIF_OK, "require_bounds failed");

  uint64_t boundary = 0, wrapped = 0;
  CHECK(sif_field_wrap_periodic(f, BOX, &boundary, &wrapped) == SIF_OK,
    "wrap_periodic failed");

  CHECK(boundary == 2, "expected 2 boundary folds, got %llu",
    (unsigned long long)boundary);
  CHECK(wrapped == 3, "expected 3 genuine wraps, got %llu",
    (unsigned long long)wrapped);

  CHECK(f->x[3] == 0.0f, "x on the edge should fold to 0, got %g",
    (double)f->x[3]);
  CHECK(f->y[9] == 0.0f, "y on the edge should fold to 0, got %g",
    (double)f->y[9]);

  /* The clamp case: whatever it lands on, it must be inside the box. */
  CHECK(z[11] + BOX == BOX, "test premise: -1e-6 + BOX should round to BOX");
  CHECK(f->z[11] >= 0.0f && f->z[11] < BOX,
    "a tiny negative must land inside the box, got %g", (double)f->z[11]);

  CHECK((f->state_flags & __FIELD_STATE_BOUNDS_VALID) == 0,
    "the bounds cache should be invalidated by a wrap");

  /* The contract itself: every coordinate now satisfies what the CIC and the
   * chain mesh require. */
  uint64_t bad = 0;
  for (uint64_t i = 0; i < N_P; i++) {
    if (!(f->x[i] >= 0.0f && f->x[i] < BOX) ||
        !(f->y[i] >= 0.0f && f->y[i] < BOX) ||
        !(f->z[i] >= 0.0f && f->z[i] < BOX))
      bad++;
  }
  CHECK(bad == 0, "%llu coordinates still outside the box after wrapping",
    (unsigned long long)bad);

  /* And the field the validators used to reject now assigns. */
  sif_grid_t* g = sif_grid_alloc(16, BOX);
  sif_grid_assign_cic(g, f);
  double sum = 0.0;
  for (uint64_t i = 0; i < g->total_cells; i++)
    sum += (double)g->delta[i];
  CHECK(sum > 0.0, "a wrapped field should assign, got total mass %g", sum);
  sif_grid_free(g);

  /* Idempotent: a field already inside the box is left alone. */
  boundary = wrapped = 1;
  CHECK(sif_field_wrap_periodic(f, BOX, &boundary, &wrapped) == SIF_OK,
    "second wrap failed");
  CHECK(boundary == 0 && wrapped == 0,
    "wrapping an in-box field should move nothing, got %llu/%llu",
    (unsigned long long)boundary, (unsigned long long)wrapped);

  CHECK(sif_field_wrap_periodic(f, 0.0f, NULL, NULL) == SIF_ERR_INVALID,
    "a non-positive box should be rejected");

  sif_field_free(f);
  free(x); free(y); free(z);
  printf("  ok\n");
}

int main(void) {
  test_velocity_permutation();
  test_sort_permutes_everything();
  test_require_helpers();
  test_cic_mass();
  test_cic_rejects_out_of_box();
  test_wrap_periodic();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
