#include "sif/structures/field.h"

#include "core/get_system.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/stringy.h"

#include <stdlib.h>
#include <string.h>

sif_field_t* sif_field_alloc(uint64_t n_particles) {
  sif_field_t* field = malloc(sizeof(sif_field_t));
  if (!field) {
    SIF_LOG_ERROR(
      "field", "failed to allocate field (%zu bytes)", sizeof(sif_field_t));
    return NULL;
  }

  field->n_particles = n_particles;
  field->state_flags = 0;

  field->_position_block = NULL;
  field->_velocity_block = NULL;

  field->x = NULL;
  field->y = NULL;
  field->z = NULL;
  field->vx = NULL;
  field->vy = NULL;
  field->vz = NULL;

  field->masses = NULL;
  field->original_indices = NULL;

  for (int i = 0; i < 3; i++) {
    field->min_p[i] = 0.0f;
    field->max_p[i] = 0.0f;
    field->center[i] = 0.0f;
  }
  field->half_span = 0.0f;

  return field;
}

/* Number of real_t per cache line, used to pad each sub-array of a block. */
PURE_FUNCTION uint64_t sif_field_padded_n(uint64_t n_particles) {
  const uint64_t align_elements = __SIF_CACHE_LINE / sizeof(real_t);
  return (n_particles + align_elements - 1) & ~(align_elements - 1);
}

void sif_field_free(sif_field_t* field) {
  if (!field) {
    SIF_LOG_WARNING("field", "cannot free a NULL field");
    return;
  }

  sif_free_aligned(field->_position_block);
  sif_free_aligned(field->_velocity_block);
  sif_free_aligned(field->masses);
  sif_free_aligned(field->original_indices);

  free(field);
}

/*
 * Carves a 3 x padded_n block into three cache-line aligned views.
 */
static int __field_reserve_block(uint64_t n_particles, real_t** block,
  real_t** a, real_t** b, real_t** c, const char* what) {

  const uint64_t padded_n = sif_field_padded_n(n_particles);

  real_t* fresh = sif_malloc_aligned(3 * padded_n * sizeof(real_t));
  if (!fresh) {
    SIF_LOG_ERROR("field", "failed to allocate the %s block", what);
    return SIF_ERR_ALLOC;
  }

  *block = fresh;
  *a = fresh;
  *b = fresh + padded_n;
  *c = fresh + (2 * padded_n);

  return SIF_OK;
}

int sif_field_reserve_positions(sif_field_t* field) {
  if (!field || field->n_particles == 0) {
    SIF_LOG_ERROR("field", "invalid or empty field");
    return SIF_ERR_INVALID;
  }

  if (field->_position_block)
    return SIF_OK;

  return __field_reserve_block(field->n_particles, &field->_position_block,
    &field->x, &field->y, &field->z, "position");
}

int sif_field_reserve_velocities(sif_field_t* field) {
  if (!field || field->n_particles == 0) {
    SIF_LOG_ERROR("field", "invalid or empty field");
    return SIF_ERR_INVALID;
  }

  if (field->_velocity_block)
    return SIF_OK;

  return __field_reserve_block(field->n_particles, &field->_velocity_block,
    &field->vx, &field->vy, &field->vz, "velocity");
}

int sif_field_reserve_masses(sif_field_t* field) {
  if (!field || field->n_particles == 0) {
    SIF_LOG_ERROR("field", "invalid or empty field");
    return SIF_ERR_INVALID;
  }

  if (field->masses)
    return SIF_OK;

  field->masses = sif_malloc_aligned(field->n_particles * sizeof(real_t));
  if (!field->masses) {
    SIF_LOG_ERROR("field", "failed to allocate the mass array");
    return SIF_ERR_ALLOC;
  }

  return SIF_OK;
}

int sif_field_assign_positions(
  sif_field_t* field, const real_t* x, const real_t* y, const real_t* z) {

  if (!field || !x || !y || !z) {
    SIF_LOG_ERROR("field", "invalid positions");
    return SIF_ERR_INVALID;
  }

  if (field->n_particles == 0) {
    SIF_LOG_ERROR("field", "cannot assign positions to an empty field");
    return SIF_ERR_INVALID;
  }

  real_t *new_block = NULL, *new_x = NULL, *new_y = NULL, *new_z = NULL;
  if (__field_reserve_block(field->n_particles, &new_block, &new_x, &new_y,
        &new_z, "position") != SIF_OK)
    return SIF_ERR_ALLOC;

  memcpy(new_x, x, field->n_particles * sizeof(real_t));
  memcpy(new_y, y, field->n_particles * sizeof(real_t));
  memcpy(new_z, z, field->n_particles * sizeof(real_t));

  /* Only release the old block once the replacement is secured. */
  sif_free_aligned(field->_position_block);

  field->_position_block = new_block;
  field->x = new_x;
  field->y = new_y;
  field->z = new_z;

  /* The stored permutation described the previous positions, so it is now
   * meaningless; leaving it in place would let a later assign_masses gather
   * through a permutation that no longer belongs to this data. */
  sif_free_aligned(field->original_indices);
  field->original_indices = NULL;

  field->state_flags &= ~__FIELD_STATE_BOUNDS_VALID;
  field->state_flags &= ~__FIELD_STATE_MORTON_SORTED;

  return SIF_OK;
}

/*
 * True when incoming per-particle data, which the caller indexes in the
 * original particle order, has to be permuted into the field's current order.
 */
static inline int __field_needs_permutation(const sif_field_t* field) {
  return (field->state_flags & __FIELD_STATE_MORTON_SORTED) &&
         field->original_indices != NULL;
}

int sif_field_assign_velocities(
  sif_field_t* field, const real_t* vx, const real_t* vy, const real_t* vz) {

  if (!field || !vx || !vy || !vz) {
    SIF_LOG_ERROR("field", "invalid velocities");
    return SIF_ERR_INVALID;
  }

  if (field->n_particles == 0) {
    SIF_LOG_ERROR("field", "cannot assign velocities to an empty field");
    return SIF_ERR_INVALID;
  }

  /* Velocities never invalidate the Morton order or the bounds, but if the
   * positions have already been permuted the incoming arrays are in the wrong
   * order and must be gathered through the stored permutation. Copying them
   * straight in would silently pair every particle with someone else's
   * velocity. */
  const int permute = __field_needs_permutation(field);

  real_t *new_block = NULL, *new_vx = NULL, *new_vy = NULL, *new_vz = NULL;
  if (__field_reserve_block(field->n_particles, &new_block, &new_vx, &new_vy,
        &new_vz, "velocity") != SIF_OK)
    return SIF_ERR_ALLOC;

  if (permute) {
    const uint64_t* perm = field->original_indices;
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < field->n_particles; i++) {
      const uint64_t src = perm[i];
      new_vx[i] = vx[src];
      new_vy[i] = vy[src];
      new_vz[i] = vz[src];
    }
  } else {
    memcpy(new_vx, vx, field->n_particles * sizeof(real_t));
    memcpy(new_vy, vy, field->n_particles * sizeof(real_t));
    memcpy(new_vz, vz, field->n_particles * sizeof(real_t));
  }

  /* Only release the old block once the replacement is secured. */
  sif_free_aligned(field->_velocity_block);

  field->_velocity_block = new_block;
  field->vx = new_vx;
  field->vy = new_vy;
  field->vz = new_vz;

  return SIF_OK;
}

int sif_field_assign_masses(sif_field_t* field, const real_t* masses) {
  if (!field || !masses) {
    SIF_LOG_ERROR("field", "invalid masses");
    return SIF_ERR_INVALID;
  }

  if (field->n_particles == 0) {
    SIF_LOG_ERROR("field", "cannot assign masses to an empty field");
    return SIF_ERR_INVALID;
  }

  const int permute = __field_needs_permutation(field);

  real_t* new_masses =
    sif_malloc_aligned(field->n_particles * sizeof(real_t));
  if (!new_masses) {
    SIF_LOG_ERROR("field", "failed to allocate the mass array");
    return SIF_ERR_ALLOC;
  }

  if (permute) {
    const uint64_t* perm = field->original_indices;
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < field->n_particles; i++)
      new_masses[i] = masses[perm[i]];
  } else {
    memcpy(new_masses, masses, field->n_particles * sizeof(real_t));
  }

  sif_free_aligned(field->masses);
  field->masses = new_masses;

  return SIF_OK;
}

/*
 * Folds one coordinate into [0, box_length).
 *
 * The final clamp is not redundant. For a small negative coordinate the
 * correction x + box_length is not representable in single precision -- near a
 * box of 2000 the neighbouring floats are ~1e-4 apart -- so it rounds back up
 * to exactly box_length and the value would leave this function still out of
 * range, which is the bug it was called to fix.
 */
static inline real_t __wrap_coordinate(real_t v, real_t box_length,
  uint64_t* boundary, uint64_t* wrapped) {

  if (v >= 0.0f && v < box_length)
    return v;

  /* Leaves NaN untouched and uncounted: it fails both comparisons above and
   * every one below, and no fold makes it a position. */
  if (!(v >= 0.0f) && !(v < box_length))
    return v;

  if (v == box_length) {
    (*boundary)++;
    return 0.0f;
  }

  (*wrapped)++;

  real_t w = REAL_FMOD(v, box_length);
  if (w < 0.0f)
    w += box_length;
  if (w >= box_length)
    w = 0.0f;

  return w;
}

int sif_field_wrap_periodic(sif_field_t* field, real_t box_length,
  uint64_t* n_boundary, uint64_t* n_wrapped) {

  if (!field || !field->x || !field->y || !field->z ||
      field->n_particles == 0) {
    SIF_LOG_ERROR("field", "invalid or empty field");
    return SIF_ERR_INVALID;
  }

  if (!(box_length > 0.0f)) {
    SIF_LOG_ERROR("field", "invalid box length %g", (double)box_length);
    return SIF_ERR_INVALID;
  }

  uint64_t boundary = 0;
  uint64_t wrapped = 0;

#pragma omp parallel for schedule(static) reduction(+ : boundary, wrapped)
  for (uint64_t i = 0; i < field->n_particles; i++) {
    field->x[i] = __wrap_coordinate(field->x[i], box_length, &boundary, &wrapped);
    field->y[i] = __wrap_coordinate(field->y[i], box_length, &boundary, &wrapped);
    field->z[i] = __wrap_coordinate(field->z[i], box_length, &boundary, &wrapped);
  }

  /* Positions moved, so anything derived from them is stale. The Morton order
   * is dropped for the same reason: a wrapped particle jumps to the opposite
   * corner, which is exactly the case the curve orders by. */
  if (boundary > 0 || wrapped > 0) {
    field->state_flags &= ~__FIELD_STATE_BOUNDS_VALID;
    field->state_flags &= ~__FIELD_STATE_MORTON_SORTED;
  }

  if (wrapped > 0) {
    SIF_LOG_WARNING("field",
      "wrapped %" PRIu64 " coordinates that were genuinely outside [0, %g); "
      "if this is not a small number, the box length is probably wrong and the "
      "folded field is meaningless",
      wrapped, (double)box_length);
  }

  if (boundary > 0) {
    SIF_LOG_INFO("field",
      "folded %" PRIu64 " coordinates sitting exactly on the box edge to 0",
      boundary);
  }

  if (n_boundary)
    *n_boundary = boundary;
  if (n_wrapped)
    *n_wrapped = wrapped;

  return SIF_OK;
}

int sif_field_compute_bounds(sif_field_t* field) {
  if (!field || !field->x || field->n_particles == 0) {
    SIF_LOG_ERROR("field", "invalid or empty field");
    return SIF_ERR_INVALID;
  }

  real_t min_x = REAL_MAX_VAL, min_y = REAL_MAX_VAL, min_z = REAL_MAX_VAL;
  real_t max_x = -REAL_MAX_VAL, max_y = -REAL_MAX_VAL, max_z = -REAL_MAX_VAL;

#pragma omp parallel for simd reduction(min : min_x, min_y, min_z)             \
  reduction(max : max_x, max_y, max_z)
  for (uint64_t i = 0; i < field->n_particles; i++) {
    min_x = REAL_MIN(min_x, field->x[i]);
    max_x = REAL_MAX(max_x, field->x[i]);

    min_y = REAL_MIN(min_y, field->y[i]);
    max_y = REAL_MAX(max_y, field->y[i]);

    min_z = REAL_MIN(min_z, field->z[i]);
    max_z = REAL_MAX(max_z, field->z[i]);
  }

  field->min_p[0] = min_x;
  field->min_p[1] = min_y;
  field->min_p[2] = min_z;
  field->max_p[0] = max_x;
  field->max_p[1] = max_y;
  field->max_p[2] = max_z;

  field->center[0] = (max_x + min_x) * 0.5f;
  field->center[1] = (max_y + min_y) * 0.5f;
  field->center[2] = (max_z + min_z) * 0.5f;

  real_t dx = max_x - min_x;
  real_t dy = max_y - min_y;
  real_t dz = max_z - min_z;
  real_t max_dim = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);

  field->half_span = (max_dim * 0.5f) * 1.001f;
  field->state_flags |= __FIELD_STATE_BOUNDS_VALID;

  return SIF_OK;
}

int sif_field_require_bounds(sif_field_t* field) {
  if (!field)
    return SIF_ERR_INVALID;

  if (field->state_flags & __FIELD_STATE_BOUNDS_VALID)
    return SIF_OK;

  return sif_field_compute_bounds(field);
}

/* --- Morton Sorting Utilities --- */

static inline uint64_t __spread_bits_3(uint32_t v) {
  uint64_t x = v & 0x1FFFFF; /* 21 bits */
  x = (x | (x << 32)) & 0x1F00000000FFFFULL;
  x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
  x = (x | (x << 8)) & 0x100F00F00F00F00FULL;
  x = (x | (x << 4)) & 0x10C30C30C30C30C3ULL;
  x = (x | (x << 2)) & 0x1249249249249249ULL;
  return x;
}

static inline uint64_t __morton_3d(uint32_t x, uint32_t y, uint32_t z) {
  return __spread_bits_3(x) | (__spread_bits_3(y) << 1) |
         (__spread_bits_3(z) << 2);
}

typedef struct {
  uint64_t original_index;
  uint64_t morton_code;
} particle_sort_t;

static int __radix_sort_morton_parallel(particle_sort_t* array, uint64_t n) {
  if (n == 0)
    return SIF_OK;

  particle_sort_t* temp = sif_malloc_aligned(n * sizeof(particle_sort_t));
  if (!temp) {
    SIF_LOG_ERROR(
      "field", "Failed to allocate temporary array for parallel radix sort");
    return SIF_ERR_ALLOC;
  }

  particle_sort_t* src = array;
  particle_sort_t* dst = temp;

  /* The per-thread histograms are indexed by omp_get_thread_num(), so they
   * must be sized by the team the parallel regions below will actually get,
   * not by the ceiling recorded at init time. Pinning the team size makes the
   * two agree by construction. */
  int n_threads = sif_system_get_max_threads();
  if (n_threads < 1)
    n_threads = 1;

  uint64_t* global_counts = calloc((size_t)n_threads * 256, sizeof(uint64_t));
  uint64_t* global_offsets = calloc((size_t)n_threads * 256, sizeof(uint64_t));

  if (!global_counts || !global_offsets) {
    SIF_LOG_ERROR("field", "failed to allocate the radix histograms");
    free(global_counts);
    free(global_offsets);
    sif_free_aligned(temp);
    return SIF_ERR_ALLOC;
  }

  for (int byte_idx = 0; byte_idx < 8; byte_idx++) {
    memset(global_counts, 0, (size_t)n_threads * 256 * sizeof(uint64_t));

#pragma omp parallel num_threads(n_threads)
    {
      int tid = sif_system_get_thread_num();
      int num_t = sif_system_get_num_threads();
      uint64_t chunk = n / num_t;
      uint64_t start = tid * chunk;
      uint64_t end = (tid == num_t - 1) ? n : start + chunk;

      for (uint64_t i = start; i < end; i++) {
        uint8_t byte_val = (src[i].morton_code >> (byte_idx * 8)) & 0xFF;
        global_counts[tid * 256 + byte_val]++;
      }
    }

    uint64_t current_offset = 0;
    for (int val = 0; val < 256; val++) {
      for (int tid = 0; tid < n_threads; tid++) {
        global_offsets[tid * 256 + val] = current_offset;
        current_offset += global_counts[tid * 256 + val];
      }
    }

#pragma omp parallel num_threads(n_threads)
    {
      int tid = sif_system_get_thread_num();
      int num_t = sif_system_get_num_threads();
      uint64_t chunk = n / num_t;
      uint64_t start = tid * chunk;
      uint64_t end = (tid == num_t - 1) ? n : start + chunk;

      uint64_t local_offsets[256];
      for (int i = 0; i < 256; i++) {
        local_offsets[i] = global_offsets[tid * 256 + i];
      }

      for (uint64_t i = start; i < end; i++) {
        uint8_t byte_val = (src[i].morton_code >> (byte_idx * 8)) & 0xFF;
        uint64_t dest_idx = local_offsets[byte_val]++;
        dst[dest_idx] = src[i];
      }
    }

    particle_sort_t* swap = src;
    src = dst;
    dst = swap;
  }

  free(global_counts);
  free(global_offsets);
  sif_free_aligned(temp);

  /* Eight byte-passes is an even number of buffer swaps, so the sorted result
   * is back in the caller's array. */
  return SIF_OK;
}

int sif_field_sort_morton(sif_field_t* field) {
  if (!field || !field->x || field->n_particles == 0) {
    SIF_LOG_ERROR("field", "invalid or empty field");
    return SIF_ERR_INVALID;
  }
  if (field->state_flags & __FIELD_STATE_MORTON_SORTED) {
    return SIF_OK; /* Already sorted! */
  }

  /* Quantization needs bounds that describe the data as it is now. */
  field->state_flags &= ~__FIELD_STATE_BOUNDS_VALID;
  if (sif_field_compute_bounds(field) != SIF_OK)
    return SIF_ERR_INVALID;

  /* Quantize against the field's bounding CUBE, not each axis independently.
   *
   * A per-axis normalization makes the Morton curve split space differently
   * from the octree, which subdivides a single cube of side 2 * half_span
   * around center. When the two disagree, the octants stop being contiguous
   * along the sorted array and sif_octree_build cannot partition. Using the
   * same cube here makes Morton order exactly depth-first octree order. */
  real_t cube_side = 2.0f * field->half_span;
  if (!(cube_side > 0.0f))
    cube_side = 1.0f; /* every particle coincident: any order will do */

  const real_t origin_x = field->center[0] - field->half_span;
  const real_t origin_y = field->center[1] - field->half_span;
  const real_t origin_z = field->center[2] - field->half_span;
  const real_t inv_side = 1.0f / cube_side;

  particle_sort_t* sort_array =
    sif_malloc_aligned(field->n_particles * sizeof(particle_sort_t));
  if (!sort_array) {
    SIF_LOG_ERROR("field", "Failed to allocate sorting array. Out of memory!");
    return SIF_ERR_ALLOC;
  }

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < field->n_particles; i++) {
    const uint32_t qx = sif_field_quantize(field->x[i], origin_x, inv_side);
    const uint32_t qy = sif_field_quantize(field->y[i], origin_y, inv_side);
    const uint32_t qz = sif_field_quantize(field->z[i], origin_z, inv_side);

    sort_array[i].original_index = i;
    sort_array[i].morton_code = __morton_3d(qx, qy, qz);
  }

  if (__radix_sort_morton_parallel(sort_array, field->n_particles) != SIF_OK) {
    sif_free_aligned(sort_array);
    return SIF_ERR_ALLOC;
  }

  const uint64_t padded_n = sif_field_padded_n(field->n_particles);

  /* Allocate new blocks */
  real_t* new_pos_block = sif_malloc_aligned(3 * padded_n * sizeof(real_t));

  real_t* new_vel_block = NULL;
  if (field->vx) {
    new_vel_block = sif_malloc_aligned(3 * padded_n * sizeof(real_t));
  }

  real_t* new_masses = NULL;
  if (field->masses) {
    new_masses = sif_malloc_aligned(field->n_particles * sizeof(real_t));
  }

  uint64_t* new_indices =
    sif_malloc_aligned(field->n_particles * sizeof(uint64_t));

  if (!new_pos_block || !new_indices || (field->masses && !new_masses) ||
      (field->vx && !new_vel_block)) {
    SIF_LOG_ERROR("field", "OOM during morton permutation reallocation.");
    /* Nothing has been swapped in yet, so releasing the partial allocations
     * leaves the field exactly as it was. */
    sif_free_aligned(new_pos_block);
    sif_free_aligned(new_vel_block);
    sif_free_aligned(new_masses);
    sif_free_aligned(new_indices);
    sif_free_aligned(sort_array);
    return SIF_ERR_ALLOC;
  }

  real_t* new_x = new_pos_block;
  real_t* new_y = new_pos_block + padded_n;
  real_t* new_z = new_pos_block + (2 * padded_n);

  real_t* new_vx = new_vel_block ? new_vel_block : NULL;
  real_t* new_vy = new_vel_block ? new_vel_block + padded_n : NULL;
  real_t* new_vz = new_vel_block ? new_vel_block + (2 * padded_n) : NULL;

  /* Apply the sorted permutation to ALL arrays in one parallel pass. This is a
   * gather through an index array, so it does not vectorize: plain
   * parallel for. */
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < field->n_particles; i++) {
    uint64_t old_idx = sort_array[i].original_index;

    new_x[i] = field->x[old_idx];
    new_y[i] = field->y[old_idx];
    new_z[i] = field->z[old_idx];

    if (new_vx) {
      new_vx[i] = field->vx[old_idx];
      new_vy[i] = field->vy[old_idx];
      new_vz[i] = field->vz[old_idx];
    }

    if (new_masses)
      new_masses[i] = field->masses[old_idx];

    new_indices[i] =
      (field->original_indices) ? field->original_indices[old_idx] : old_idx;
  }

  sif_free_aligned(sort_array);

  sif_free_aligned(field->_position_block);
  sif_free_aligned(field->_velocity_block);
  sif_free_aligned(field->masses);
  sif_free_aligned(field->original_indices);

  field->_position_block = new_pos_block;
  field->x = new_x;
  field->y = new_y;
  field->z = new_z;

  field->_velocity_block = new_vel_block;
  field->vx = new_vx;
  field->vy = new_vy;
  field->vz = new_vz;

  field->masses = new_masses;
  field->original_indices = new_indices;

  field->state_flags |= __FIELD_STATE_MORTON_SORTED;

  /* Reordering does not move the bounding box, so the cached bounds stay
   * valid. */

  return SIF_OK;
}

int sif_field_require_morton(sif_field_t* field) {
  if (!field)
    return SIF_ERR_INVALID;

  if (field->state_flags & __FIELD_STATE_MORTON_SORTED)
    return SIF_OK;

  return sif_field_sort_morton(field);
}
