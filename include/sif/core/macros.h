/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file macros.h
 * @brief Library-wide types, option flags, status codes and compiler hints.
 *
 * Every translation unit in sif includes this header, directly or otherwise.
 * It is organised in sections:
 *
 *   1. option flags     the bit fields passed as sif_option
 *   2. scalar type      sif_real and its math wrappers
 *   3. compiler hints   attribute wrappers with neutral fallbacks
 *   4. status codes     SIF_OK and the SIF_ERR_* family
 *   5. assertions       debug-only invariant checks
 *   6. atomics          relaxed read-modify-write on a uint64_t
 *   7. log levels
 */

#ifndef SIF_CORE_MACROS_H
#define SIF_CORE_MACROS_H

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* 1. option flags                                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief Bit field of behaviour flags accepted by most entry points.
 *
 * Flags from different families occupy different bit ranges and are combined
 * with `|`. Passing SIF_DEFAULT selects every family's default, which is
 * always the zero-valued member.
 */
typedef uint32_t sif_option;

/** @brief Every option at its default. */
#define SIF_DEFAULT 0u

/**
 * @defgroup opt_pbc Boundary conditions
 * @brief Whether the box wraps at its faces.
 * @{
 */
#define SIF_PBC_PERIODIC (0u << 0)
#define SIF_PBC_OPEN     (1u << 0)
#define SIF__PBC_MASK    (1u << 0)
/** @} */

/**
 * @defgroup opt_finder Finder options
 * @brief Bits 12-15. Bits 8-11 are free.
 * @{
 */
/**
 * Leave the grid holding the field smoothed at the last radius, instead of
 * restoring the input.
 *
 * A finder smooths the grid in place at every radius, so the input is gone by
 * the time it finishes. By default it is transformed back at the end; this flag
 * skips that, saving one inverse FFT for a caller that is going to discard the
 * grid anyway.
 */
#define SIF_FINDER_CONSUME_GRID (1u << 15)

/**
 * @name Rescaling search radius
 *
 * How far past a rung the rescaling looks for the crossing, as a multiple of
 * the rung: r_search = factor * radius.
 *
 * This is the finder's dominant cost and almost none of it is usually needed.
 * The rescaling histograms every tracer in the shell [radius, r_search] --
 * everything inside the rung is counted whole, cell by cell, and costs
 * nothing -- so the work goes as factor^3 - 1. At 2.0 that is 7 r^3 to place a
 * crossing that, on a 1.5% ladder, is measured at a mean of 1.03 r.
 *
 * What the reach actually has to be is set by the ladder, not by the void: a
 * rung returns crossings in [radius, r_search], so consecutive rungs must
 * overlap or sizes between them are reachable at no rung at all. Reaching to
 * the previous, larger rung is exactly enough, and sif_finder_exodus() raises
 * r_search to it when the factor here falls short -- so a sparse ladder stays
 * correct whatever is chosen, and a fine one stops paying for reach it has
 * already covered. A void is found at the largest rung at or below its
 * crossing radius either way; a lower factor only stops the rungs below that
 * one from re-deriving an answer their elder already returned.
 *
 * 1.5 is the default because it is comfortably above any ladder anyone runs
 * (a 50% step) while costing 2.4 r^3 against 7. Raise it for a coarse ladder,
 * or to 2.0 to reproduce older catalogs bit for bit.
 * @{
 */
#define SIF_FINDER_SEARCH_1_50 (0u << 12) /**< Default. */
#define SIF_FINDER_SEARCH_1_25 (1u << 12)
#define SIF_FINDER_SEARCH_1_75 (2u << 12)
#define SIF_FINDER_SEARCH_2_00 (3u << 12)
#define SIF__FINDER_SEARCH_MASK (3u << 12)

/**
 * @brief The multiple of the rung a #SIF__FINDER_SEARCH_MASK selection asks
 * for. Order follows the encoding, not the value: the default has to be the
 * zero-valued member.
 */
static inline float sif__finder_search_factor(uint32_t opt) {
  static const float f[4] = {1.5f, 1.25f, 1.75f, 2.0f};
  return f[(opt & SIF__FINDER_SEARCH_MASK) >> 12];
}
/** @} */

/**
 * Skip deconvolution of the CIC assignment window.
 *
 * sif_grid_assign_cic() convolves the field with its own window before the
 * finder ever sees it, so a top-hat applied on top of that smooths twice: the
 * effective window is wider than the radius asked for, and it has soft edges
 * about a cell thick. Both push the measured contrast in a void upward, which
 * for the exodus finder shows up directly as rescaled radii biased large.
 * Deconvolution is therefore the default; this flag disables it for grids that
 * were not built by CIC -- one filled by hand, or from a tessellation, or read
 * from a file whose assignment is unknown.
 *
 * @note The correction grows towards the Nyquist corner, so it is only safe
 * while the top-hat that follows suppresses those modes. The finders check the
 * smallest radius against the cell size and warn when it does not.
 */
#define SIF_FINDER_KEEP_CIC_WINDOW (1u << 14)
/** @} */

/**
 * @defgroup opt_mesh Chain mesh options
 * @brief Bits 16-17, deliberately outside the 8-14 range the per-entry-point
 * families reuse: a mesh is built from inside several of those calls, and a
 * flag that collided with one of them would be read as the other's.
 * @{
 */
/**
 * Release sif_chain_mesh_t::original_indices once the mesh is in canonical
 * order, instead of keeping it.
 *
 * The map back to field order costs 8 bytes per particle -- 25 GiB at 3.4e9
 * tracers -- and the finders never read it. It is still built and still sorted
 * on, so the mesh is byte-for-byte the one a default build produces; only the
 * key is dropped afterwards. The cost is that
 * sif_chain_mesh_find_nearest_open() and sif_chain_mesh_find_nearest_pbc()
 * can no longer name their answer and refuse.
 */
#define SIF_MESH_DROP_INDICES (1u << 16)

/**
 * Skip the canonical ordering pass, leaving each cell in whatever order the
 * parallel scatter produced.
 *
 * The pass exists so that two identical runs give byte-identical meshes: the
 * scatter claims slots with an atomic, so which chunk reaches a cell first is
 * a race, and everything that walks a cell inherits it. Sorting each cell on
 * the field index replaces that with an order fixed by the input alone.
 *
 * What that buys is reproducibility of the last bits, and nothing else -- the
 * set of particles in a cell is identical either way, so any measurement that
 * does not depend on summation order is unaffected. A finder that only counts
 * tracers inside a sphere is in that class; stacked profiles, anything that
 * sums weights (the exodus finder included, on a weighted mesh) and
 * nearest-neighbour queries with exact distance ties are not.
 *
 * Worth setting only where the ordering genuinely does not matter, because the
 * pass is no longer the bottleneck it once was: it sorts a compact key array
 * and applies the permutation once, rather than swapping whole payload rows
 * per inversion.
 */
#define SIF_MESH_NO_CANONICAL (1u << 17)
#define SIF__MESH_MASK        (3u << 16)
/** @} */

/**
 * @defgroup opt_profiles Radial density binning
 * @brief What each bin of a density profile holds.
 *
 * CUMULATIVE is the contrast enclosed within a bin's outer edge, which is what
 * the spherical-evolution mapping takes. DIFFERENTIAL is the contrast of that
 * shell alone, which is the one worth measuring from a tessellation: a shell
 * is all boundary, so it is where resolving a cell's overlap with the sphere
 * rather than counting whole tracers makes the most difference.
 *
 * Velocity profiles are differential either way; a mean infall over everything
 * inside a radius is not a quantity anyone wants.
 * @{
 */
#define SIF_PROFILES_CUMULATIVE   (0u << 8)
#define SIF_PROFILES_DIFFERENTIAL (1u << 8)
#define SIF__PROFILES_BIN_MASK    (1u << 8)
/** @} */

/**
 * @defgroup opt_profiles_vel Radial velocity averaging
 * @brief What a shell's mean radial velocity is averaged over.
 *
 * NUMBER is the plain mean over the tracers in the shell. WEIGHTED is the mean
 * over their weights, sum(w v) / sum(w), which for a mass-weighted field is the
 * momentum of the shell over its mass. On a mesh without weights the two are
 * the same thing.
 *
 * NUMBER is the default because of what it means on a tessellation's samples:
 * every sample stands for the same volume, so counting them gives the
 * volume-weighted velocity, which is what the samples are for. WEIGHTED on the
 * same mesh gives the tracer-weighted one instead.
 * @{
 */
#define SIF_PROFILES_VELOCITY_NUMBER   (0u << 9)
#define SIF_PROFILES_VELOCITY_WEIGHTED (1u << 9)
#define SIF__PROFILES_VELOCITY_MASK    (1u << 9)
/** @} */

/**
 * @defgroup opt_spherical Spherical-evolution mapping
 * @brief How the linear and non-linear density contrasts of a void are mapped
 * onto each other.
 *
 * B94 is the Bernardeau (1994) fit, closed form and accurate to 0.2%; EXACT
 * root-finds the Einstein-de Sitter expansion solution.
 *
 * @note Bit 9 is free. Bits 8-9 used to select between a histogram, a KDE and
 * an auto-binned histogram; the VSF is now always a plain histogram, and bit 8
 * has since been taken by this mapping.
 * @{
 */
#define SIF_SPHERICAL_B94   (0u << 8)
#define SIF_SPHERICAL_EXACT (1u << 8)
#define SIF__SPHERICAL_MASK (1u << 8)
/** @} */

/**
 * @defgroup opt_vsf_bin Void size function binning
 * @brief Spacing of the radius bins.
 * @{
 */
#define SIF_VSF_BIN_LN     (0u << 10)
#define SIF_VSF_BIN_LINEAR (1u << 10)
#define SIF__VSF_BIN_MASK  (1u << 10)
/** @} */

/**
 * @defgroup opt_vsf_merge Void size function combination
 * @brief How several size functions are merged into one.
 * @{
 */
#define SIF_VSF_MERGE_MEAN   (0u << 12)
#define SIF_VSF_MERGE_MEDIAN (1u << 12)
#define SIF_VSF_MERGE_STITCH (2u << 12)
#define SIF__VSF_MERGE_MASK  (3u << 12)
/** @} */

/**
 * @defgroup opt_delta_shuffle Surrogate field generation
 * @brief How the surrogate field for a PDF comparison is generated.
 *
 * PHASES keeps every `|delta_k|` and randomizes only the phase, so the realized
 * P(k) is bit-for-bit the input's and any change in the PDF is attributable to
 * phase information alone. GAUSSIAN additionally resamples the amplitudes from
 * the Rayleigh distribution implied by `|delta_k|`, which is the
 * correct
 * surrogate when the comparison is against a Gaussian random field *ensemble*
 * rather than against this one realization.
 * @{
 */
#define SIF_DELTA_SHUFFLE_NONE     (0u << 8)
#define SIF_DELTA_SHUFFLE_PHASES   (1u << 8)
#define SIF_DELTA_SHUFFLE_GAUSSIAN (2u << 8)
#define SIF__DELTA_SHUFFLE_MASK    (3u << 8)
/** @} */

/**
 * @brief Skip deconvolution of the CIC assignment window.
 *
 * The CIC assignment in sif_grid_assign_cic convolves the field with its own
 * window, which has to come back out before the top-hat is applied or the
 * field is smoothed twice. Deconvolution is therefore the default; this flag
 * disables it for grids that were not built by CIC.
 */
#define SIF_DELTA_KEEP_CIC_WINDOW (1u << 10)

/**
 * @defgroup opt_delta_filter Smoothing window
 * @brief Window applied to both the PDF and the spectral moments.
 *
 * They have to share one window: gamma and R_star describe the field whose PDF
 * is being measured, and mixing a top-hat histogram with Gaussian-window
 * moments would describe two different fields.
 *
 * The top-hat is the natural window for counts in spheres, but its W^2 decays
 * only as k^-4, so the k^4-weighted sigma_2 sum does not converge and is cut
 * off by the grid instead of by the field. Use the Gaussian whenever sigma_2
 * has to carry meaning.
 * @{
 */
#define SIF_DELTA_FILTER_TOP_HAT  (0u << 11)
#define SIF_DELTA_FILTER_GAUSSIAN (1u << 11)
#define SIF__DELTA_FILTER_MASK    (1u << 11)
/** @} */

/**
 * @defgroup opt_bbks_g BBKS curvature integral
 * @brief Which G(gamma, w) to evaluate.
 *
 * FITTED is the BBKS analytic approximation, fast and closed-form but
 * calibrated for a limited gamma band. EXACT quadratures the defining integral,
 * which costs a few hundred evaluations of the curvature weight per point and
 * is correct everywhere.
 * @{
 */
#define SIF_BBKS_G_FITTED (0u << 8)
#define SIF_BBKS_G_EXACT  (1u << 8)
#define SIF__BBKS_G_MASK  (1u << 8)
/** @} */

/* ------------------------------------------------------------------ */
/* 2. scalar type                                                      */
/* ------------------------------------------------------------------ */

/**
 * @brief The library's floating-point type: double if SIF_USE_DOUBLE is
 * defined at configure time, float otherwise.
 *
 * Use sif_real for every physical quantity, and the SIF_REAL_* wrappers below
 * rather than the libm functions directly: calling cos() on a float build
 * promotes to double and back on every evaluation, and calling cosf() on a
 * double build silently discards precision.
 */
#ifdef SIF_USE_DOUBLE

typedef double sif_real;
#  define SIF_REAL_MAX_VAL          DBL_MAX
#  define SIF_REAL_MIN_VAL          DBL_MIN
#  define SIF_REAL_ABS(x)           fabs(x)
#  define SIF_REAL_CEIL(x)          ceil(x)
#  define SIF_REAL_FLOOR(x)         floor(x)
#  define SIF_REAL_ROUND(x)         round(x)
#  define SIF_REAL_FMOD(x, y)       fmod(x, y)
#  define SIF_REAL_MIN(x, y)        fmin(x, y)
#  define SIF_REAL_MAX(x, y)        fmax(x, y)
#  define SIF_REAL_COS(x)           cos(x)
#  define SIF_REAL_SIN(x)           sin(x)
#  define SIF_REAL_TAN(x)           tan(x)
#  define SIF_REAL_ACOS(x)          acos(x)
#  define SIF_REAL_ASIN(x)          asin(x)
#  define SIF_REAL_ATAN(x)          atan(x)
#  define SIF_REAL_ATAN2(y, x)      atan2(y, x)
#  define SIF_REAL_SQRT(x)          sqrt(x)
#  define SIF_REAL_POW(x, y)        pow(x, y)
#  define SIF_REAL_EXP(x)           exp(x)
#  define SIF_REAL_LOG(x)           log(x)
#  define SIF_REAL_LOG10(x)         log10(x)
#  define SIF_REAL_COPYSIGN(x, y)   copysign(x, y)
#  define SIF_REAL_NEXT_AFTER(x, y) nextafter(x, y)

/** printf conversion that round-trips a sif_real exactly: 17 significant
 * digits are enough to recover any double, 9 any float.
 */
#  define SIF_PRI_REAL "%.17g"
/** scanf conversion matching sif_real. Getting this wrong is silent: scanf
 * writes through a pointer whose type it cannot check.
 */
#  define SIF_SCN_REAL "%lf"

#else

typedef float sif_real;
#  define SIF_REAL_MAX_VAL          FLT_MAX
#  define SIF_REAL_MIN_VAL          FLT_MIN /* smallest positive normalized */
#  define SIF_REAL_ABS(x)           fabsf(x)
#  define SIF_REAL_CEIL(x)          ceilf(x)
#  define SIF_REAL_FLOOR(x)         floorf(x)
#  define SIF_REAL_ROUND(x)         roundf(x)
#  define SIF_REAL_FMOD(x, y)       fmodf(x, y)
#  define SIF_REAL_MIN(x, y)        fminf(x, y)
#  define SIF_REAL_MAX(x, y)        fmaxf(x, y)
#  define SIF_REAL_COS(x)           cosf(x)
#  define SIF_REAL_SIN(x)           sinf(x)
#  define SIF_REAL_TAN(x)           tanf(x)
#  define SIF_REAL_ACOS(x)          acosf(x)
#  define SIF_REAL_ASIN(x)          asinf(x)
#  define SIF_REAL_ATAN(x)          atanf(x)
#  define SIF_REAL_ATAN2(y, x)      atan2f(y, x)
#  define SIF_REAL_SQRT(x)          sqrtf(x)
#  define SIF_REAL_POW(x, y)        powf(x, y)
#  define SIF_REAL_EXP(x)           expf(x)
#  define SIF_REAL_LOG(x)           logf(x)
#  define SIF_REAL_LOG10(x)         log10f(x)
#  define SIF_REAL_COPYSIGN(x, y)   copysignf(x, y)
#  define SIF_REAL_NEXT_AFTER(x, y) nextafterf(x, y)

#  define SIF_PRI_REAL "%.9g"
#  define SIF_SCN_REAL "%f"

/* Selects the float entry points in the vendored predicates. The leading
 * underscores break this guide's naming rules deliberately: the name is read
 * by vendor/predicates/predicates.c, and upstream's spelling wins there. */
#  define __PREDICATES_USE_FLOAT

#endif

/** @brief Pi at sif_real precision.
 *
 * Defined here rather than taken from M_PI, which is not in C99 and whose
 * fallbacks are easy to get wrong: a float literal used in a double build
 * silently costs eight digits.
 */
#define SIF_PI ((sif_real)3.14159265358979323846)

/** @brief Assumed cache-line size, in bytes. Override at configure time.
 *
 * Must be a power of two and at least `sizeof(void*)`.
 */
#ifndef SIF_CACHE_LINE
#  define SIF_CACHE_LINE 64
#endif

/* The allocator rounds sizes with `(n + SIF_CACHE_LINE - 1) & ~(SIF_CACHE_LINE
 * - 1)`, and posix_memalign() demands a power-of-two multiple of sizeof(void*).
 * A value that is neither would not fail anywhere visible -- every allocation
 * in the library would just return NULL -- so it is rejected here instead. C99
 * has no _Static_assert, and a negative array bound is the portable way to fail
 * at compile time. */
#define SIF__CACHE_LINE_IS_VALID                                               \
  (SIF_CACHE_LINE >= (int)sizeof(void*) &&                                     \
    (SIF_CACHE_LINE & (SIF_CACHE_LINE - 1)) == 0)

typedef char sif__cache_line_check[SIF__CACHE_LINE_IS_VALID ? 1 : -1];

/* ------------------------------------------------------------------ */
/* 3. compiler hints                                                   */
/* ------------------------------------------------------------------ */

/** @brief Emits a _Pragma from a macro argument. */
#define SIF_PRAGMA_HELPER(x) _Pragma(#x)

#if defined(__GNUC__) || defined(__clang__)

/** @brief Warn if the return value is discarded. */
#  define SIF_NODISCARD __attribute__((warn_unused_result))
/** @brief No side effects; result depends only on the arguments and memory. */
#  define SIF_PURE_FUNCTION __attribute__((pure))
/** @brief Hint that a function sits on a hot path. */
#  define SIF_HOT_LOOP __attribute__((hot))
/** @brief Inline even where the compiler would rather not: for a body that
 * takes a compile-time flag and has to be specialized on it at every call. */
#  define SIF_ALWAYS_INLINE inline __attribute__((always_inline))
/** @brief Align an object on a cache line. */
#  define SIF_ALIGN_T __attribute__((aligned(SIF_CACHE_LINE)))

#else

#  define SIF_NODISCARD
#  define SIF_PURE_FUNCTION
#  define SIF_HOT_LOOP
#  define SIF_ALWAYS_INLINE inline
#  define SIF_ALIGN_T

#endif

/**
 * @defgroup bit_intrinsics Bit-counting intrinsics
 * @brief Population count and trailing-zero count, with portable fallbacks.
 *
 * Wrapped here rather than called directly, for the same reason as everything
 * else in this section: a raw `__builtin_` in library code compiles nowhere
 * but GCC and Clang, and the failure is a build error in a file that has
 * nothing to do with portability. The fallbacks are the textbook loops --
 * slower, but only reached on a compiler that has no intrinsic to offer.
 * @{
 */
#if defined(__GNUC__) || defined(__clang__)

#  define SIF_POPCOUNT_U64(x) ((uint32_t)__builtin_popcountll((uint64_t)(x)))
#  define SIF_CTZ_U32(x)      ((uint32_t)__builtin_ctz((uint32_t)(x)))

#else

/** @brief Set bits in a 64-bit word. Undefined for no compiler at all. */
static inline uint32_t sif__popcount_u64(uint64_t x) {
  uint32_t n = 0;
  while (x) {
    x &= x - 1; /* clears the lowest set bit */
    n++;
  }
  return n;
}

/** @brief Trailing zeros in a 32-bit word. Undefined for x == 0, matching
 * __builtin_ctz.
 */
static inline uint32_t sif__ctz_u32(uint32_t x) {
  uint32_t n = 0;
  while ((x & 1u) == 0u) {
    x >>= 1;
    n++;
  }
  return n;
}

#  define SIF_POPCOUNT_U64(x) sif__popcount_u64((uint64_t)(x))
#  define SIF_CTZ_U32(x)      sif__ctz_u32((uint32_t)(x))

#endif
/** @} */

/* ------------------------------------------------------------------ */
/* 4. status codes                                                     */
/* ------------------------------------------------------------------ */

/**
 * @defgroup status Status codes
 * @brief Returned by functions that can fail for a distinguishable reason.
 *
 * Such functions return an int: SIF_OK on success, a negative SIF_ERR_*
 * otherwise. Allocators keep returning a pointer, NULL on failure.
 * @{
 */
#define SIF_OK          0
#define SIF_ERR_ALLOC   (-1)
#define SIF_ERR_INVALID (-2)
#define SIF_ERR_RANGE   (-3)
/** A file could not be opened, read, written, or failed validation. */
#define SIF_ERR_IO (-4)
/** @} */

/* ------------------------------------------------------------------ */
/* 5. assertions                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Debug-only invariant check. Compiles to nothing by default.
 *
 * Enabled by configuring with -DSIF_DEBUG_CHECKS. Deliberately NOT keyed off
 * NDEBUG: the release preset never defines it, and these guards sit in the
 * hottest loops in the library.
 *
 * @warning The condition is not evaluated in a normal build. It must never
 * carry a side effect.
 */
#ifdef SIF_DEBUG_CHECKS
#  include <assert.h>
#  define SIF_ASSERT(cond) assert(cond)
#else
#  define SIF_ASSERT(cond) ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* 6. atomics                                                          */
/* ------------------------------------------------------------------ */

/**
 * @defgroup atomics Relaxed atomics
 * @brief Read-modify-write on a uint64_t word, relaxed ordering.
 *
 * Relaxed is sufficient wherever these are used: the words are bitmask lanes
 * whose bits are set independently, so only atomicity of the individual
 * update matters, not its order relative to any other memory operation.
 *
 * SIF_HAS_ATOMIC_BUILTINS is 0 where the compiler provides no such builtins,
 * and callers must serialize the update themselves.
 * @{
 */
#if defined(__GNUC__) || defined(__clang__)
#  define SIF_ATOMIC_LOAD_U64(p)  __atomic_load_n((p), __ATOMIC_RELAXED)
#  define SIF_ATOMIC_OR_U64(p, v) __atomic_fetch_or((p), (v), __ATOMIC_RELAXED)
#  define SIF_ATOMIC_AND_U64(p, v)                                             \
    __atomic_fetch_and((p), (v), __ATOMIC_RELAXED)
#  define SIF_HAS_ATOMIC_BUILTINS 1
#else
#  define SIF_HAS_ATOMIC_BUILTINS 0
#endif
/** @} */

/* ------------------------------------------------------------------ */
/* 7. log levels                                                       */
/* ------------------------------------------------------------------ */

/**
 * @defgroup log_levels Log levels
 * @brief Verbosity threshold, set through sif_config_t::log_level.
 * @{
 */
#define SIF_LOG_LEVEL_TRACE   0
#define SIF_LOG_LEVEL_DEBUG   1
#define SIF_LOG_LEVEL_INFO    2
#define SIF_LOG_LEVEL_WARNING 3
#define SIF_LOG_LEVEL_ERROR   4
#define SIF_LOG_LEVEL_NONE    5
/** @} */

#endif /* SIF_CORE_MACROS_H */
