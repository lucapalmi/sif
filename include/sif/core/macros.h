#ifndef __SIF_MACROS_H__
#define __SIF_MACROS_H__

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>

/* configuration */

typedef uint32_t sif_option_t;

#define SIF_DEFAULT 0u

#define SIF_PBC_PERIODIC (0u << 0)
#define SIF_PBC_OPEN     (1u << 0)
#define __SIF_PBC_MASK   (1u << 0)

/* --- FINDER OPTIONS --- */

#define SIF_FINDER_CENTER_IS_MINIMUM     (1u << 8)
#define SIF_FINDER_REFINE_CENTER_HESSIAN (1u << 9)
#define SIF_FINDER_PRESERVE_GRID         (1u << 15)

/* Bits 10-14 are free. They used to carry SIF_FINDER_MARKING_FAST and the
 * RMIN_FACTOR / NOISE_TOLERANCE families; all three were removed because
 * their "default" variant encoded to 0 and therefore never reached the
 * dispatch, so the documented defaults were never the ones in effect. The
 * values that were actually running are now the fixed constants in
 * rescaled_spherical_finder.c. */

/* --- PROFILES OPTIONS --- */

#define SIF_PROFILES_ALGO_MESH    (0u << 8)
#define SIF_PROFILES_ALGO_VORONOI (1u << 8)
#define __SIF_PROFILES_ALGO_MASK  (1u << 8)

/* --- SIZE FUNCTION OPTIONS --- */

/* Bit 9 is free. Bits 8-9 used to select between a histogram, a KDE and an
 * auto-binned histogram; the VSF is now always a plain histogram, and bit 8
 * has since been taken by the spherical-evolution mapping below. */

/* How the linear and non-linear density contrasts of a void are mapped onto
 * each other. B94 is the Bernardeau (1994) fit, closed form and accurate to
 * 0.2%; EXACT root-finds the Einstein-de Sitter expansion solution. */
#define SIF_SPHERICAL_B94    (0u << 8)
#define SIF_SPHERICAL_EXACT  (1u << 8)
#define __SIF_SPHERICAL_MASK (1u << 8)

#define SIF_VSF_BIN_LN     (0u << 10)
#define SIF_VSF_BIN_LINEAR (1u << 10)
#define __SIF_VSF_BIN_MASK (1u << 10)

/* --- VSF COMBINATION OPTIONS --- */

#define SIF_VSF_MERGE_MEAN   (0u << 12)
#define SIF_VSF_MERGE_MEDIAN (1u << 12)
#define SIF_VSF_MERGE_STITCH (2u << 12)
#define __SIF_VSF_MERGE_MASK (3u << 12)

/* --- DELTA STATISTICS OPTIONS --- */

/* Surrogate field generation. PHASES keeps every |delta_k| and randomizes only
 * the phase, so the realized P(k) is bit-for-bit the input's and any change in
 * the PDF is attributable to phase information alone. GAUSSIAN additionally
 * resamples the amplitudes from the Rayleigh distribution implied by |delta_k|,
 * which is the correct surrogate when the comparison is against a Gaussian
 * random field *ensemble* rather than against this one realization. */
#define SIF_DELTA_SHUFFLE_NONE     (0u << 8)
#define SIF_DELTA_SHUFFLE_PHASES   (1u << 8)
#define SIF_DELTA_SHUFFLE_GAUSSIAN (2u << 8)
#define __SIF_DELTA_SHUFFLE_MASK   (3u << 8)

/* The CIC assignment in sif_grid_assign_cic convolves the field with its own
 * window, which has to come back out before the top-hat is applied or the
 * field is smoothed twice. Deconvolution is therefore the default; the flag
 * disables it for grids that were not built by CIC. */
#define SIF_DELTA_KEEP_CIC_WINDOW (1u << 10)

/* Smoothing window, applied to both the PDF and the spectral moments. They
 * have to share one window: gamma and R_star describe the field whose PDF is
 * being measured, and mixing a top-hat histogram with Gaussian-window moments
 * would describe two different fields.
 *
 * The top-hat is the natural window for counts in spheres, but its W^2 decays
 * only as k^-4, so the k^4-weighted sigma_2 sum does not converge and is cut
 * off by the grid instead of by the field. Use the Gaussian whenever sigma_2
 * has to carry meaning. */
#define SIF_DELTA_FILTER_TOP_HAT  (0u << 11)
#define SIF_DELTA_FILTER_GAUSSIAN (1u << 11)
#define __SIF_DELTA_FILTER_MASK   (1u << 11)

/* --- BBKS PEAK STATISTICS OPTIONS --- */

/* Which G(gamma, w) to evaluate. FITTED is the BBKS analytic approximation,
 * fast and closed-form but calibrated for a limited gamma band. EXACT
 * quadratures the defining integral, which costs a few hundred evaluations of
 * the curvature weight per point and is correct everywhere. */
#define SIF_BBKS_G_FITTED (0u << 8)
#define SIF_BBKS_G_EXACT  (1u << 8)
#define __SIF_BBKS_G_MASK (1u << 8)

/* --- TESSELLATION OPTIONS --- */

#define SIF_TESS_METHOD_RANDOM (0u << 8)
#define SIF_TESS_METHOD_VOXEL  (1u << 8)
#define __SIF_TESS_METHOD_MASK (1u << 8)

/* real_t type */

#ifdef __SIF_USE_DOUBLE

typedef double real_t;
#  define REAL_MAX_VAL        DBL_MAX
#  define REAL_MIN_VAL        DBL_MIN
#  define REAL_ABS(x)         fabs(x)
#  define REAL_CEIL(x)        ceil(x)
#  define REAL_FLOOR(x)       floor(x)
#  define REAL_ROUND(x)       round(x)
#  define REAL_FMOD(x, y)     fmod(x, y)
#  define REAL_MIN(x, y)      fmin(x, y)
#  define REAL_MAX(x, y)      fmax(x, y)
#  define REAL_COS(x)         cos(x)
#  define REAL_SIN(x)         sin(x)
#  define REAL_TAN(x)         tan(x)
#  define REAL_ACOS(x)        acos(x)
#  define REAL_ASIN(x)        asin(x)
#  define REAL_ATAN(x)        atan(x)
#  define REAL_ATAN2(y, x)    atan2(y, x)
#  define REAL_SQRT(x)        sqrt(x)
#  define REAL_POW(x, y)      pow(x, y)
#  define REAL_EXP(x)         exp(x)
#  define REAL_LOG(x)         log(x)
#  define REAL_LOG10(x)       log10(x)
#  define REAL_COPYSIGN(x, y) copysign(x, y)

#else

typedef float real_t;
#  define REAL_MAX_VAL        FLT_MAX
#  define REAL_MIN_VAL        FLT_MIN /* Smallest positive normalized value */
#  define REAL_ABS(x)         fabsf(x)
#  define REAL_CEIL(x)        ceilf(x)
#  define REAL_FLOOR(x)       floorf(x)
#  define REAL_ROUND(x)       roundf(x)
#  define REAL_FMOD(x, y)     fmodf(x, y)
#  define REAL_MIN(x, y)      fminf(x, y)
#  define REAL_MAX(x, y)      fmaxf(x, y)
#  define REAL_COS(x)         cosf(x)
#  define REAL_SIN(x)         sinf(x)
#  define REAL_TAN(x)         tanf(x)
#  define REAL_ACOS(x)        acosf(x)
#  define REAL_ASIN(x)        asinf(x)
#  define REAL_ATAN(x)        atanf(x)
#  define REAL_ATAN2(y, x)    atan2f(y, x)
#  define REAL_SQRT(x)        sqrtf(x)
#  define REAL_POW(x, y)      powf(x, y)
#  define REAL_EXP(x)         expf(x)
#  define REAL_LOG(x)         logf(x)
#  define REAL_LOG10(x)       log10f(x)
#  define REAL_COPYSIGN(x, y) copysignf(x, y)
#  define __PREDICATES_USE_FLOAT

#endif

#ifndef __SIF_CACHE_LINE
#  define __SIF_CACHE_LINE 64
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846f
#endif

/* compiler hints */

#define PRAGMA_HELPER(x) _Pragma(#x)

#if defined(__GNUC__) || defined(__clang__)

#  define NODISCARD     __attribute__((warn_unused_result))
#  define PURE_FUNCTION __attribute__((pure))
#  define HOT_LOOP      __attribute__((hot))
#  define ALIGN_T       __attribute__((aligned(__SIF_CACHE_LINE)))

#else

#  define NODISCARD
#  define PURE_FUNCTION
#  define HOT_LOOP
#  define ALIGN_64
#  define ALIGN_T

#endif

/* status codes
 *
 * Functions that can fail for a reason the caller may want to distinguish
 * return an int: SIF_OK on success, a negative SIF_ERR_* otherwise.
 * Allocators keep returning a pointer (NULL on failure). */

#define SIF_OK           0
#define SIF_ERR_ALLOC   (-1)
#define SIF_ERR_INVALID (-2)
#define SIF_ERR_RANGE   (-3)

/* debug-only invariant checks
 *
 * Enabled by configuring with -DSIF_DEBUG_CHECKS. Deliberately NOT keyed off
 * NDEBUG: the release preset never defines it, and these guards sit in the
 * hottest loops in the library. */

#ifdef SIF_DEBUG_CHECKS
#  include <assert.h>
#  define SIF_ASSERT(cond) assert(cond)
#else
#  define SIF_ASSERT(cond) ((void)0)
#endif

/* relaxed atomic read-modify-write on a uint64_t word */

#if defined(__GNUC__) || defined(__clang__)
#  define SIF_ATOMIC_LOAD_U64(p)      __atomic_load_n((p), __ATOMIC_RELAXED)
#  define SIF_ATOMIC_OR_U64(p, v)     __atomic_fetch_or((p), (v), __ATOMIC_RELAXED)
#  define SIF_ATOMIC_AND_U64(p, v)    __atomic_fetch_and((p), (v), __ATOMIC_RELAXED)
#  define SIF_HAS_ATOMIC_BUILTINS     1
#else
#  define SIF_HAS_ATOMIC_BUILTINS     0
#endif

#define __SIF_RESCALED_SPHERICAL_FINDER_RADIAL_BIN_SIZE 0.1f
#define __SIF_RESCALED_SPHERICAL_FINDER_MAX_BINS        2048

#define __SIF_PROFILE_GRID_DIM 100

#define SIF_LOG_LEVEL_TRACE   0
#define SIF_LOG_LEVEL_DEBUG   1
#define SIF_LOG_LEVEL_INFO    2
#define SIF_LOG_LEVEL_WARNING 3
#define SIF_LOG_LEVEL_ERROR   4
#define SIF_LOG_LEVEL_NONE    5

#endif /* __SIF_MACROS_H__ */
