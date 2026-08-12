/* Verifies the octree against brute force. The non-cubic cases are the ones
 * that used to break: the Morton curve and the octree cube disagreed, octants
 * stopped being contiguous, and the tree came out wrong. */
#include "sif/structures/field.h"
#include "sif/structures/octree.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

static uint64_t rs = 0xDEADBEEFCAFEULL;
static double uni(void) {
  rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)((rs >> 11) & 0x1FFFFFFFFFFFFFULL) / (double)0x20000000000000ULL;
}

/*
 * Walks every leaf and confirms the tree actually describes the field:
 * ranges must tile [0, n) exactly, and each particle must lie inside the
 * cube of the leaf that claims it.
 */
static void check_tree_integrity(const sif_octree_t* tree,
  const sif_field_t* field, uint32_t node, real_t cx, real_t cy, real_t cz,
  real_t hs, uint8_t* seen, int* bad_containment) {

  const sif_octree_node_t* nd = &tree->nodes[node];

  if (nd->first_child == UINT32_MAX) {
    for (uint32_t p = nd->p_start; p < nd->p_start + nd->p_counting; p++) {
      if (p >= field->n_particles) {
        (*bad_containment)++;
        continue;
      }
      seen[p]++;
      /* A small tolerance: child centers are recomputed by repeated halving. */
      /* Slack only for the accumulated halving in the recomputed centers;
       * the partition itself is exact, so this must not need to be large. */
      const real_t eps = hs * 1e-3f + 1e-4f;
      if (fabsf(field->x[p] - cx) > hs + eps ||
          fabsf(field->y[p] - cy) > hs + eps ||
          fabsf(field->z[p] - cz) > hs + eps) {
        (*bad_containment)++;
      }
    }
    return;
  }

  const real_t q = hs * 0.5f;
  for (int i = 0; i < 8; i++) {
    check_tree_integrity(tree, field, nd->first_child + i,
      cx + ((i & 1) ? q : -q), cy + ((i & 2) ? q : -q),
      cz + ((i & 4) ? q : -q), q, seen, bad_containment);
  }
}

static void run(const char* label, real_t sx, real_t sy, real_t sz,
  uint64_t n_p, uint32_t max_per_leaf) {

  printf("%s (extents %gx%gx%g, %llu particles)\n", label, (double)sx,
    (double)sy, (double)sz, (unsigned long long)n_p);

  real_t* x = malloc(n_p * sizeof(real_t));
  real_t* y = malloc(n_p * sizeof(real_t));
  real_t* z = malloc(n_p * sizeof(real_t));
  for (uint64_t i = 0; i < n_p; i++) {
    x[i] = (real_t)(uni() * sx);
    y[i] = (real_t)(uni() * sy);
    z[i] = (real_t)(uni() * sz);
  }

  sif_field_t* f = sif_field_alloc(n_p);
  sif_field_assign_positions(f, x, y, z);

  sif_octree_t* tree = sif_octree_alloc(256);
  CHECK(tree != NULL, "octree alloc failed");
  if (!tree) return;

  CHECK(sif_octree_build(tree, f, max_per_leaf) == 0, "build failed");

  /* 1. Structural integrity: every particle in exactly one leaf, and inside
   *    that leaf's cube. */
  uint8_t* seen = calloc(n_p, 1);
  int bad_containment = 0;
  check_tree_integrity(tree, f, 0, tree->root_center[0], tree->root_center[1],
    tree->root_center[2], tree->root_half_span, seen, &bad_containment);

  uint64_t missing = 0, duplicated = 0;
  for (uint64_t i = 0; i < n_p; i++) {
    if (seen[i] == 0) missing++;
    else if (seen[i] > 1) duplicated++;
  }
  CHECK(missing == 0, "%llu particles are in no leaf", (unsigned long long)missing);
  CHECK(duplicated == 0, "%llu particles are in several leaves",
    (unsigned long long)duplicated);
  CHECK(bad_containment == 0, "%d particles sit outside their leaf's cube",
    bad_containment);
  free(seen);

  /* 2. Nearest-neighbour queries must match brute force. The field is sorted
   *    in place, so compare against the sorted arrays. */
  int wrong = 0;
  const int n_queries = SIF_TEST_SCALE(200);
  for (int q = 0; q < n_queries; q++) {
    const real_t px = (real_t)(uni() * sx);
    const real_t py = (real_t)(uni() * sy);
    const real_t pz = (real_t)(uni() * sz);

    double best = 1e300;
    uint64_t best_i = 0;
    for (uint64_t i = 0; i < n_p; i++) {
      const double dx = (double)f->x[i] - px, dy = (double)f->y[i] - py,
                   dz = (double)f->z[i] - pz;
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) { best = d2; best_i = i; }
    }

    const uint64_t got = sif_octree_find_nearest(tree, f, px, py, pz);
    if (got >= n_p) { wrong++; continue; }

    const double gdx = (double)f->x[got] - px, gdy = (double)f->y[got] - py,
                 gdz = (double)f->z[got] - pz;
    const double gd2 = gdx * gdx + gdy * gdy + gdz * gdz;
    /* Ties are fine; only a genuinely farther particle is a failure. */
    if (gd2 > best * (1.0 + 1e-6)) {
      wrong++;
      if (wrong <= 2)
        printf("    query %d: got d=%.6f, brute force d=%.6f (idx %llu vs %llu)\n",
          q, sqrt(gd2), sqrt(best), (unsigned long long)got,
          (unsigned long long)best_i);
    }
  }
  CHECK(wrong == 0, "%d of %d nearest-neighbour queries were wrong", wrong,
    n_queries);

  sif_octree_free(tree);
  sif_field_free(f);
  free(x); free(y); free(z);
  printf("  ok\n");
}

int main(void) {
  run("cubic", 100.0f, 100.0f, 100.0f, (uint64_t)SIF_TEST_SCALE(20000), 16);
  run("flat slab", 100.0f, 100.0f, 5.0f, (uint64_t)SIF_TEST_SCALE(20000), 16);
  run("elongated", 200.0f, 20.0f, 20.0f, (uint64_t)SIF_TEST_SCALE(20000), 16);
  run("extreme aspect", 500.0f, 3.0f, 60.0f, (uint64_t)SIF_TEST_SCALE(20000), 8);

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
