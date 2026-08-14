/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The utility layer: allocation, array constructors and reductions, the ASCII
 * scanner, checksums, the PRNG, the generated sorts and the timer.
 *
 * Small pieces, but everything above them is built on the guarantees they
 * make -- alignment, an exact endpoint, a reproducible draw, a checksum that
 * matches other tools' -- and those guarantees are the thing worth pinning
 * down here. Where a property is what makes the function worth calling at all
 * (calloc's overflow check, sum's independence from the thread count), it gets
 * a test of its own rather than being assumed.
 */

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/utils/align.h"
#include "sif/utils/array.h"
#include "sif/utils/crc32.h"
#include "sif/utils/random.h"
#include "sif/utils/sort.h"
#include "sif/utils/str.h"
#include "sif/utils/timer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: ");                                                      \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

/* Sorts used by the sort tests. Defined at file scope, which is the only
 * place the macro may appear. */
SIF_DEFINE_QUICKSORT(sort_real_asc, sif_real, a < b)
SIF_DEFINE_QUICKSORT(sort_real_desc, sif_real, a > b)

static void test_align(void) {
  printf("aligned allocation\n");

  void* p = sif_malloc_aligned(1);
  CHECK(p != NULL, "one-byte allocation failed");
  CHECK(((uintptr_t)p % SIF_CACHE_LINE) == 0,
    "block is not cache-line aligned (%p)", p);
  sif_free_aligned(p);

  /* Zero is not an error, but there is nothing to hand back either. */
  CHECK(sif_malloc_aligned(0) == NULL, "a zero-byte request should be NULL");

  sif_real* z = sif_calloc_aligned(64, sizeof(sif_real));
  CHECK(z != NULL, "calloc failed");
  if (z) {
    int all_zero = 1;
    for (int i = 0; i < 64; i++)
      if (z[i] != (sif_real)0.0)
        all_zero = 0;
    CHECK(all_zero, "calloc did not zero the block");
    sif_free_aligned(z);
  }

  /* The whole reason to call the calloc form: a product that wraps must be
   * refused, not turned into a small block every later write runs off. */
  CHECK(sif_calloc_aligned(SIZE_MAX / 2 + 1, 4) == NULL,
    "an overflowing element count should be refused");
  CHECK(sif_calloc_aligned(SIZE_MAX, SIZE_MAX) == NULL,
    "an overflowing product should be refused");

  /* Likewise the round-up to a whole cache line, which wraps near SIZE_MAX
   * and would ask the allocator for a handful of bytes. */
  CHECK(sif_malloc_aligned(SIZE_MAX) == NULL,
    "a request that cannot be rounded up should be refused");

  /* realloc keeps the old contents up to the shorter of the two lengths. */
  uint32_t* r = sif_malloc_aligned(4 * sizeof(uint32_t));
  CHECK(r != NULL, "realloc setup allocation failed");
  if (r) {
    for (uint32_t i = 0; i < 4; i++)
      r[i] = i + 1;

    r = sif_realloc_aligned(r, 4 * sizeof(uint32_t), 8 * sizeof(uint32_t));
    CHECK(r != NULL, "grow failed");
    if (r) {
      CHECK(r[0] == 1 && r[3] == 4, "grow lost the contents");
      CHECK(((uintptr_t)r % SIF_CACHE_LINE) == 0, "grown block lost alignment");

      r = sif_realloc_aligned(r, 8 * sizeof(uint32_t), 2 * sizeof(uint32_t));
      CHECK(r != NULL && r[0] == 1 && r[1] == 2, "shrink lost the head");
    }
    CHECK(sif_realloc_aligned(r, 2 * sizeof(uint32_t), 0) == NULL,
      "resizing to zero should free and return NULL");
  }

  sif_free_aligned(NULL); /* must be accepted */
}

static void test_array_construction(void) {
  printf("array constructors\n");

  sif_real* a = sif_array_ones(10);
  CHECK(a != NULL && a[0] == (sif_real)1.0 && a[9] == (sif_real)1.0,
    "ones did not fill");
  sif_free_aligned(a);

  a = sif_array_full(10, (sif_real)-2.5);
  CHECK(a != NULL && a[7] == (sif_real)-2.5, "full did not fill");
  sif_free_aligned(a);

  a = sif_array_zeros(10);
  CHECK(a != NULL && a[0] == (sif_real)0.0 && a[9] == (sif_real)0.0,
    "zeros did not zero");
  sif_free_aligned(a);

  CHECK(sif_array_zeros(0) == NULL, "a zero-length array should be NULL");
  CHECK(sif_array_ones(0) == NULL, "a zero-length array should be NULL");

  /* Both endpoints land exactly. Callers use these as bin edges and compare
   * against the last one, so an accumulated final element would put a value
   * on the wrong side of the top bin. */
  a = sif_array_linspace((sif_real)0.0, (sif_real)1.0, 11);
  CHECK(a != NULL, "linspace failed");
  if (a) {
    CHECK(a[0] == (sif_real)0.0, "linspace start is %g, not 0", (double)a[0]);
    CHECK(a[10] == (sif_real)1.0, "linspace stop is %g, not 1", (double)a[10]);
    CHECK(SIF_REAL_ABS(a[5] - (sif_real)0.5) < (sif_real)1e-6,
      "linspace midpoint is %g, not 0.5", (double)a[5]);
    sif_free_aligned(a);
  }

  a = sif_array_linspace((sif_real)3.0, (sif_real)4.0, 1);
  CHECK(a != NULL && a[0] == (sif_real)3.0, "linspace of 1 should be {start}");
  sif_free_aligned(a);

  /* logspace spaces the exponents, so the endpoints are powers of the base. */
  a = sif_array_logspace((sif_real)0.0, (sif_real)3.0, 4, (sif_real)10.0);
  CHECK(a != NULL, "logspace failed");
  if (a) {
    CHECK(SIF_REAL_ABS(a[0] - (sif_real)1.0) < (sif_real)1e-5,
      "logspace start is %g, not 1", (double)a[0]);
    CHECK(SIF_REAL_ABS(a[3] - (sif_real)1000.0) < (sif_real)1e-2,
      "logspace stop is %g, not 1000", (double)a[3]);
    sif_free_aligned(a);
  }
}

static void test_array_arange(void) {
  printf("arange\n");

  uint64_t n = 12345;
  sif_real* a =
    sif_array_arange((sif_real)0.0, (sif_real)10.0, (sif_real)2.0, &n);
  CHECK(a != NULL, "arange failed");
  CHECK(
    n == 5, "arange produced %llu elements, expected 5", (unsigned long long)n);
  if (a)
    CHECK(a[0] == (sif_real)0.0 && a[4] == (sif_real)8.0,
      "arange values are wrong (%g .. %g)", (double)a[0], (double)a[4]);
  sif_free_aligned(a);

  /* A descending range needs a negative step, and vice versa. */
  n = 12345;
  a = sif_array_arange((sif_real)5.0, (sif_real)0.0, (sif_real)-1.0, &n);
  CHECK(a != NULL && n == 5, "descending arange produced %llu elements",
    (unsigned long long)n);
  if (a)
    CHECK(a[4] == (sif_real)1.0, "descending arange ends at %g, not 1",
      (double)a[4]);
  sif_free_aligned(a);

  /* Every rejection must leave out_size at 0. A caller loops over that count,
   * so a stale value beside a NULL pointer is worse than no value at all. */
  n = 12345;
  CHECK(
    sif_array_arange((sif_real)0.0, (sif_real)1.0, (sif_real)0.0, &n) == NULL,
    "a zero step should be refused");
  CHECK(n == 0, "a zero step left out_size at %llu", (unsigned long long)n);

  n = 12345;
  CHECK(
    sif_array_arange((sif_real)5.0, (sif_real)0.0, (sif_real)1.0, &n) == NULL,
    "an empty range should be refused");
  CHECK(n == 0, "an empty range left out_size at %llu", (unsigned long long)n);

  /* A step this small makes the element count exceed what a uint64_t holds.
   * The cast is undefined, not merely wrong, so the count is bounded first. */
  n = 12345;
  CHECK(sif_array_arange((sif_real)0.0, (sif_real)1e30, (sif_real)1e-30, &n) ==
          NULL,
    "an unrepresentable element count should be refused");
  CHECK(
    n == 0, "a rejected count left out_size at %llu", (unsigned long long)n);
}

static void test_array_reductions(void) {
  printf("array reductions\n");

  const uint64_t n = 1000;
  sif_real* a = sif_malloc_aligned(n * sizeof(sif_real));
  CHECK(a != NULL, "reduction setup allocation failed");
  if (!a)
    return;

  /* 0 + 1 + ... + 999 = 499500, exact in both precisions. */
  for (uint64_t i = 0; i < n; i++)
    a[i] = (sif_real)i;

  CHECK(sif_array_sum(a, n) == (sif_real)499500.0, "sum is %g, not 499500",
    (double)sif_array_sum(a, n));
  CHECK(sif_array_min(a, n) == (sif_real)0.0, "min is %g, not 0",
    (double)sif_array_min(a, n));
  CHECK(sif_array_max(a, n) == (sif_real)999.0, "max is %g, not 999",
    (double)sif_array_max(a, n));

  /* Fewer elements than blocks: most blocks are empty and must contribute
   * their identity rather than a stray zero. */
  CHECK(sif_array_sum(a, 3) == (sif_real)3.0, "sum of the first 3 is %g, not 3",
    (double)sif_array_sum(a, 3));
  CHECK(sif_array_max(a, 3) == (sif_real)2.0, "max of the first 3 is %g, not 2",
    (double)sif_array_max(a, 3));
  CHECK(sif_array_min(a, 1) == (sif_real)0.0, "min of one element is wrong");

  CHECK(sif_array_sum(NULL, 10) == (sif_real)0.0, "sum of NULL should be 0");
  CHECK(sif_array_sum(a, 0) == (sif_real)0.0, "sum of nothing should be 0");
  CHECK(sif_array_min(NULL, 10) == (sif_real)0.0, "min of NULL should be 0");
  CHECK(sif_array_max(a, 0) == (sif_real)0.0, "max of nothing should be 0");

  sif_free_aligned(a);
}

static void test_array_sum_is_stable(void) {
  printf("sum does not depend on the thread count\n");

  /* One large term followed by many small ones. A sif_real accumulator in a
   * single-precision build stops moving here -- one ulp at 1e8 is 8.0, so
   * adding 1.0 changes nothing -- and returns 1e8 instead of 1.01e8. The
   * block sums are accumulated in double precisely so this does not happen. */
  const uint64_t n = 1000001;
  sif_real* a = sif_malloc_aligned(n * sizeof(sif_real));
  CHECK(a != NULL, "stability setup allocation failed");
  if (!a)
    return;

  a[0] = (sif_real)1e8;
  for (uint64_t i = 1; i < n; i++)
    a[i] = (sif_real)1.0;

  const double expected = 1e8 + 1e6;
  const double got = (double)sif_array_sum(a, n);
  CHECK(SIF_REAL_ABS(got - expected) < 1.0,
    "sum lost the small terms: %.1f, expected %.1f", got, expected);

#ifdef _OPENMP
  /* The partition is a property of the array, not of the machine, so the same
   * input must give the same bits however many threads run over it. */
  const int saved = omp_get_max_threads();

  omp_set_num_threads(1);
  const sif_real one_thread = sif_array_sum(a, n);

  omp_set_num_threads(7);
  const sif_real seven_threads = sif_array_sum(a, n);

  omp_set_num_threads(saved);
  const sif_real restored = sif_array_sum(a, n);

  CHECK(one_thread == seven_threads, "1 thread gives %.1f, 7 threads give %.1f",
    (double)one_thread, (double)seven_threads);
  CHECK(one_thread == restored, "the sum moved again at %d threads", saved);
#endif

  sif_free_aligned(a);
}

static void test_str(void) {
  printf("format decoding and field scanning\n");

  sif_col_target_t targets[8];

  int n = sif_str_decode_format("xyz*m", targets, 8);
  CHECK(n == 5, "decoded %d columns from \"xyz*m\", expected 5", n);
  CHECK(targets[0] == SIF_COL_X && targets[2] == SIF_COL_Z &&
          targets[3] == SIF_COL_IGNORE && targets[4] == SIF_COL_M,
    "\"xyz*m\" decoded to the wrong targets");

  CHECK(
    sif_str_decode_format("XYZ", targets, 8) == 3, "upper case not decoded");
  CHECK(sif_str_decode_format("uvw", targets, 8) == 3,
    "velocity columns not decoded");

  /* Unknown characters are skipped, not rejected -- which is what lets a
   * format be written with separators, and why a typo shortens the layout
   * instead of failing. */
  CHECK(sif_str_decode_format("x y z", targets, 8) == 3,
    "spaces in a format should be ignored");
  CHECK(sif_str_decode_format("?!", targets, 8) == 0,
    "an unrecognized format should decode to nothing");

  /* Decoding stops at the caller's capacity rather than writing past it. */
  CHECK(sif_str_decode_format("xyzuvwm", targets, 3) == 3,
    "decoding did not stop at max_cols");

  CHECK(sif_str_decode_format(NULL, targets, 8) == 0, "NULL fmt should be 0");
  CHECK(sif_str_decode_format("xyz", NULL, 8) == 0, "NULL targets should be 0");

  /* Whitespace-separated, with the cursor left ready for the next field. */
  char line[] = "1.5 -2 3e2\n";
  char* cursor = line;
  sif_real v = (sif_real)0.0;

  CHECK(sif_str_extract_next_real(&cursor, ' ', &v) == 1 && v == (sif_real)1.5,
    "first field read as %g, not 1.5", (double)v);
  CHECK(sif_str_extract_next_real(&cursor, ' ', &v) == 1 && v == (sif_real)-2.0,
    "second field read as %g, not -2", (double)v);
  CHECK(
    sif_str_extract_next_real(&cursor, ' ', &v) == 1 && v == (sif_real)300.0,
    "third field read as %g, not 300", (double)v);

  /* End of line is reported as "no field", which is how a caller tells a
   * short row from one whose columns merely failed to parse. */
  CHECK(sif_str_extract_next_real(&cursor, ' ', &v) == 0,
    "the end of the line should report no field");

  char csv[] = "4,5,6";
  cursor = csv;
  CHECK(sif_str_extract_next_real(&cursor, ',', &v) == 1 && v == (sif_real)4.0,
    "comma-delimited field read as %g, not 4", (double)v);

  /* A text column is consumed and reads as zero, so one bad column does not
   * derail the columns behind it. */
  char mixed[] = "abc 7";
  cursor = mixed;
  CHECK(sif_str_extract_next_real(&cursor, ' ', &v) == 1 && v == (sif_real)0.0,
    "a non-numeric field should read as 0, got %g", (double)v);
  CHECK(sif_str_extract_next_real(&cursor, ' ', &v) == 1 && v == (sif_real)7.0,
    "the field after a non-numeric one read as %g, not 7", (double)v);

  char blank[] = "\n";
  cursor = blank;
  CHECK(sif_str_extract_next_real(&cursor, ' ', &v) == 0,
    "a blank line should report no field");
}

static void test_crc32(void) {
  printf("crc32\n");

  /* The check value every CRC-32 implementation publishes for this input. It
   * is here so a file sif writes can be verified with zlib, gzip or python's
   * binascii and agree. */
  const char* check = "123456789";
  CHECK(sif_crc32(check, 9) == 0xCBF43926u,
    "crc32(\"123456789\") is %08x, expected cbf43926", sif_crc32(check, 9));

  /* Folding two buffers must equal checksumming their concatenation: the
   * binary writers rely on it, since a field's blocks are separate
   * allocations but one payload. */
  uint32_t running = SIF_CRC32_INIT;
  running = sif_crc32_update(running, "12345", 5);
  running = sif_crc32_update(running, "6789", 4);
  CHECK(sif_crc32_final(running) == 0xCBF43926u,
    "a chained checksum differs from the one-shot one");

  /* The initial and final complements are what make length matter; without
   * them a run of zeros would checksum to zero however long it was. */
  const uint8_t zeros[8] = {0};
  CHECK(sif_crc32(zeros, 4) != sif_crc32(zeros, 8),
    "four and eight zero bytes checksum the same");

  CHECK(sif_crc32(NULL, 0) == 0, "an empty buffer should checksum to 0");
}

static void test_random(void) {
  printf("prng\n");

  sif_prng_state_t a, b;
  sif_prng_init(&a, 12345);
  sif_prng_init(&b, 12345);

  /* Reproducible from the seed alone: a run that reports a seed has to be
   * repeatable from it. */
  int identical = 1;
  for (int i = 0; i < 64; i++)
    if (sif_prng_next_u64(&a) != sif_prng_next_u64(&b))
      identical = 0;
  CHECK(identical, "two states seeded alike diverged");

  sif_prng_init(&b, 12346);
  CHECK(sif_prng_next_u64(&a) != sif_prng_next_u64(&b),
    "adjacent seeds produced the same first draw");

  /* [0, 1), both ends meant literally. */
  sif_prng_init(&a, 7);
  int in_range = 1;
  for (int i = 0; i < 10000; i++) {
    const double u = sif_prng_next_double(&a);
    if (!(u >= 0.0 && u < 1.0))
      in_range = 0;
  }
  CHECK(in_range, "a double draw fell outside [0, 1)");

  sif_prng_init(&a, 7);
  in_range = 1;
  for (int i = 0; i < 10000; i++) {
    const sif_real u = sif_prng_next_real(&a);
    if (!(u >= (sif_real)0.0 && u < (sif_real)1.0))
      in_range = 0;
  }
  CHECK(in_range, "a sif_real draw fell outside [0, 1)");

  /* Box-Muller over enough pairs to place the mean and variance loosely. This
   * is a smoke test for the transform, not a test of the generator. */
  sif_prng_init(&a, 99);
  double mean = 0.0, mean_sq = 0.0;
  const int n_pairs = 50000;
  for (int i = 0; i < n_pairs; i++) {
    double z0, z1;
    sif_prng_next_gaussian_pair(&a, &z0, &z1);
    mean += z0 + z1;
    mean_sq += z0 * z0 + z1 * z1;
  }
  mean /= (2 * n_pairs);
  mean_sq /= (2 * n_pairs);

  CHECK(mean > -0.02 && mean < 0.02, "gaussian mean is %.4f, not ~0", mean);
  CHECK(mean_sq > 0.97 && mean_sq < 1.03, "gaussian variance is %.4f, not ~1",
    mean_sq);
}

static void test_sort(void) {
  printf("generated quicksort\n");

  /* Ties on purpose: the partition is three-way so that runs of equal keys
   * are placed in one pass, and radius arrays are full of them. */
  sif_real values[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 5, 5, 0, 5};
  const uint64_t n = sizeof(values) / sizeof(values[0]);

  sif_real work[15];
  memcpy(work, values, sizeof(values));
  sort_real_asc(work, n);

  int ordered = 1;
  for (uint64_t i = 1; i < n; i++)
    if (work[i] < work[i - 1])
      ordered = 0;
  CHECK(ordered, "ascending sort left the array out of order");
  CHECK(work[0] == (sif_real)0.0 && work[n - 1] == (sif_real)9.0,
    "ascending sort ends are %g .. %g, expected 0 .. 9", (double)work[0],
    (double)work[n - 1]);

  /* Already sorted, which the middle-element pivot makes the good case. */
  sort_real_asc(work, n);
  CHECK(work[0] == (sif_real)0.0 && work[n - 1] == (sif_real)9.0,
    "re-sorting a sorted array changed it");

  memcpy(work, values, sizeof(values));
  sort_real_desc(work, n);
  ordered = 1;
  for (uint64_t i = 1; i < n; i++)
    if (work[i] > work[i - 1])
      ordered = 0;
  CHECK(ordered, "descending sort left the array out of order");
  CHECK(work[0] == (sif_real)9.0, "descending sort starts at %g, not 9",
    (double)work[0]);

  /* Degenerate lengths must be no-ops rather than reading arr[-1]. */
  sif_real one = (sif_real)42.0;
  sort_real_asc(&one, 1);
  sort_real_asc(&one, 0);
  CHECK(one == (sif_real)42.0, "sorting one element changed it");
}

static void test_timer(void) {
  printf("timer\n");

  sif_timer_t t;
  sif_timer_start(&t);

  /* Something the optimizer cannot discard, so the interval is real. */
  volatile double spin = 0.0;
  for (int i = 0; i < 2000000; i++)
    spin += (double)i;

  sif_timer_stop(&t);

  const double ms = sif_timer_elapsed_ms(&t);
  CHECK(ms >= 0.0, "elapsed time came back negative (%g ms)", ms);
  CHECK(ms < 60000.0, "elapsed time is implausible (%g ms)", ms);

  CHECK(sif_timer_elapsed_ms(NULL) == -1.0, "a NULL timer should report -1.0");
}

int main(void) {
  sif_init(SIF_CONFIG_QUIET);

  test_align();
  test_array_construction();
  test_array_arange();
  test_array_reductions();
  test_array_sum_is_stable();
  test_str();
  test_crc32();
  test_random();
  test_sort();
  test_timer();

  sif_finalize();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
