#include "sif/structures/tessellation.h"
#include "sif/core/macros.h"

#include "predicates/predicates.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* */
/* BOWYER - WATSON */
/* */

#define __MAX_CAVITY_SIZE    16348
#define __DEAD_MARKER_OFFSET 0x8000000000000000ULL

static real_t _super_center[3] = {0, 0, 0};
static real_t _super_span = 1.0f;
#pragma omp threadprivate(_super_center, _super_span)

static void __init_super_tetrahedron_bounds(const sif_field_t* field) {
  real_t min_x = REAL_MAX_VAL, min_y = REAL_MAX_VAL, min_z = REAL_MAX_VAL;
  real_t max_x = -REAL_MAX_VAL, max_y = -REAL_MAX_VAL, max_z = -REAL_MAX_VAL;

  for (uint64_t i = 0; i < field->n_particles; i++) {
    if (field->x[i] < min_x)
      min_x = field->x[i];
    if (field->y[i] < min_y)
      min_y = field->y[i];
    if (field->z[i] < min_z)
      min_z = field->z[i];
    if (field->x[i] > max_x)
      max_x = field->x[i];
    if (field->y[i] > max_y)
      max_y = field->y[i];
    if (field->z[i] > max_z)
      max_z = field->z[i];
  }

  _super_center[0] = (max_x + min_x) * 0.5f;
  _super_center[1] = (max_y + min_y) * 0.5f;
  _super_center[2] = (max_z + min_z) * 0.5f;

  real_t dx = max_x - min_x;
  real_t dy = max_y - min_y;
  real_t dz = max_z - min_z;
  _super_span = (dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz)) * 1000.0f;
}

HOT_LOOP static inline void __get_vertex_coords(
  const sif_field_t* field, uint64_t v_id, real_t* out_p) {
  if (v_id < field->n_particles) {
    out_p[0] = field->x[v_id];
    out_p[1] = field->y[v_id];
    out_p[2] = field->z[v_id];
  } else {
    uint64_t virtual_idx = v_id - field->n_particles;
    real_t S = _super_span;
    real_t cx = _super_center[0];
    real_t cy = _super_center[1];
    real_t cz = _super_center[2];

    if (virtual_idx == 0) {
      out_p[0] = cx;
      out_p[1] = cy;
      out_p[2] = cz + 3.0f * S;
    } else if (virtual_idx == 1) {
      out_p[0] = cx - 3.0f * S;
      out_p[1] = cy + 3.0f * S;
      out_p[2] = cz - 3.0f * S;
    } else if (virtual_idx == 2) {
      out_p[0] = cx;
      out_p[1] = cy - 3.0f * S;
      out_p[2] = cz - 3.0f * S;
    } else if (virtual_idx == 3) {
      out_p[0] = cx + 3.0f * S;
      out_p[1] = cy + 3.0f * S;
      out_p[2] = cz - 3.0f * S;
    }
  }
}

/* --- 3. Memory Management --- */

static uint64_t __alloc_tetrahedron(
  sif_delaunay_tessellation_t* tess, uint64_t** visited_array) {
  if (tess->free_head != UINT64_MAX) {
    uint64_t id = tess->free_head;
    tess->free_head = tess->tetrahedrons[id].neighbors[0];
    return id;
  }

  if (tess->n_tetrahedrons == tess->capacity) {
    uint64_t new_cap = tess->capacity * 2;
    tess->tetrahedrons =
      realloc(tess->tetrahedrons, new_cap * sizeof(sif_tetrahedron_t));
    *visited_array = realloc(*visited_array, new_cap * sizeof(uint64_t));
    memset((*visited_array) + tess->capacity, 0,
      (new_cap - tess->capacity) * sizeof(uint64_t));
    tess->capacity = new_cap;
  }
  return tess->n_tetrahedrons++;
}

static void __delete_tetrahedron(
  uint64_t tet_id, sif_delaunay_tessellation_t* tess) {
  sif_tetrahedron_t* tet = &tess->tetrahedrons[tet_id];
  tet->vertices[0] = UINT64_MAX;
  tet->neighbors[0] = tess->free_head;
  tess->free_head = tet_id;
}

typedef struct {
  uint64_t v[3];
  uint64_t outside_tet;
  uint8_t outside_face;
  uint64_t new_tet_id;
} cavity_face_t;

HOT_LOOP static uint64_t __find_containing_tet(
  sif_delaunay_tessellation_t* tess, const sif_field_t* field,
  uint64_t target_p_id, uint64_t start_tet_id) {

  real_t p_target[3];
  __get_vertex_coords(field, target_p_id, p_target);

  uint64_t curr_tet_id = start_tet_id;
  uint32_t failsafe = 0;

  while (failsafe++ < 50000) {
    sif_tetrahedron_t* curr = &tess->tetrahedrons[curr_tet_id];

    real_t p[4][3];
    for (int i = 0; i < 4; i++)
      __get_vertex_coords(field, curr->vertices[i], p[i]);

    int crossed_face = -1;
    real_t o;

    o = orient3dfast(p[1], p[2], p[3], p_target);
    if (o > 0.0f)
      crossed_face = 0;
    else if (o == 0.0f && orient3d(p[1], p[2], p[3], p_target) > 0.0f)
      crossed_face = 0;

    if (crossed_face == -1) {
      o = orient3dfast(p[0], p[3], p[2], p_target);
      if (o > 0.0f)
        crossed_face = 1;
      else if (o == 0.0f && orient3d(p[0], p[3], p[2], p_target) > 0.0f)
        crossed_face = 1;
    }

    if (crossed_face == -1) {
      o = orient3dfast(p[0], p[1], p[3], p_target);
      if (o > 0.0f)
        crossed_face = 2;
      else if (o == 0.0f && orient3d(p[0], p[1], p[3], p_target) > 0.0f)
        crossed_face = 2;
    }

    if (crossed_face == -1) {
      o = orient3dfast(p[0], p[2], p[1], p_target);
      if (o > 0.0f)
        crossed_face = 3;
      else if (o == 0.0f && orient3d(p[0], p[2], p[1], p_target) > 0.0f)
        crossed_face = 3;
    }

    if (crossed_face == -1)
      return curr_tet_id;

    uint64_t next_tet_id = curr->neighbors[crossed_face];
    if (next_tet_id == UINT64_MAX)
      return curr_tet_id;

    curr_tet_id = next_tet_id;
  }
  return curr_tet_id;
}

HOT_LOOP static void __dig_cavity(sif_delaunay_tessellation_t* tess,
  const sif_field_t* field, uint64_t p_idx, uint64_t gen_marker,
  uint64_t start_tet, cavity_face_t* out_faces, uint64_t* out_n_faces,
  uint64_t* dead_tets, uint64_t* out_n_dead, uint64_t* visited_array) {

  uint64_t n_faces = 0, n_dead = 0;
  uint64_t stack[__MAX_CAVITY_SIZE];
  uint64_t stack_size = 0;

  stack[stack_size++] = start_tet;
  visited_array[start_tet] = gen_marker;

  real_t p_target[3];
  __get_vertex_coords(field, p_idx, p_target);

  while (stack_size > 0) {
    uint64_t curr_tet_id = stack[--stack_size];
    sif_tetrahedron_t* curr = &tess->tetrahedrons[curr_tet_id];
    dead_tets[n_dead++] = curr_tet_id;

    if (n_dead >= __MAX_CAVITY_SIZE - 1 ||
        stack_size >= __MAX_CAVITY_SIZE - 4) {
      SIF_LOG_ERROR("delaunay", "Cavity bounds exceeded! Geometry degenerate.");
      break;
    }

    for (uint8_t i = 0; i < 4; i++) {
      uint64_t neighbor_id = curr->neighbors[i];
      if (neighbor_id == UINT64_MAX)
        continue;

      if (visited_array[neighbor_id] != gen_marker) {
        visited_array[neighbor_id] = gen_marker;

        sif_tetrahedron_t* neighbor = &tess->tetrahedrons[neighbor_id];
        real_t pA[3], pB[3], pC[3], pD[3];
        __get_vertex_coords(field, neighbor->vertices[0], pA);
        __get_vertex_coords(field, neighbor->vertices[1], pB);
        __get_vertex_coords(field, neighbor->vertices[2], pC);
        __get_vertex_coords(field, neighbor->vertices[3], pD);

        if (insphere(pA, pB, pC, pD, p_target) > 0.0) {
          stack[stack_size++] = neighbor_id;
        }
      }
    }
  }

  uint64_t dead_marker = gen_marker | __DEAD_MARKER_OFFSET;
  for (uint64_t d = 0; d < n_dead; d++)
    visited_array[dead_tets[d]] = dead_marker;

  for (uint64_t d = 0; d < n_dead; d++) {
    uint64_t dead_id = dead_tets[d];
    sif_tetrahedron_t* dead_tet = &tess->tetrahedrons[dead_id];

    for (uint8_t i = 0; i < 4; i++) {
      uint64_t neighbor_id = dead_tet->neighbors[i];

      if (neighbor_id != UINT64_MAX &&
          visited_array[neighbor_id] == dead_marker)
        continue;

      cavity_face_t* face = &out_faces[n_faces++];

      if (i == 0) {
        face->v[0] = dead_tet->vertices[1];
        face->v[1] = dead_tet->vertices[3];
        face->v[2] = dead_tet->vertices[2];
      } else if (i == 1) {
        face->v[0] = dead_tet->vertices[0];
        face->v[1] = dead_tet->vertices[2];
        face->v[2] = dead_tet->vertices[3];
      } else if (i == 2) {
        face->v[0] = dead_tet->vertices[0];
        face->v[1] = dead_tet->vertices[3];
        face->v[2] = dead_tet->vertices[1];
      } else if (i == 3) {
        face->v[0] = dead_tet->vertices[0];
        face->v[1] = dead_tet->vertices[1];
        face->v[2] = dead_tet->vertices[2];
      }
      face->outside_tet = neighbor_id;
      face->outside_face = 255;

      if (neighbor_id != UINT64_MAX) {
        sif_tetrahedron_t* outside = &tess->tetrahedrons[neighbor_id];
        for (uint8_t j = 0; j < 4; j++) {
          if (outside->neighbors[j] == dead_id) {
            face->outside_face = j;
            break;
          }
        }
      }
    }
  }
  *out_n_faces = n_faces;
  *out_n_dead = n_dead;
}

#define __EDGE_HT_SIZE 262144

typedef struct {
  uint64_t v_start;
  uint64_t v_end;
  uint16_t face_idx;
  uint8_t edge_idx;
  uint64_t marker; /* Replaces occupied for zero-clear hashing */
} _edge_entry_t;

static inline uint32_t __edge_hash(uint64_t v0, uint64_t v1) {
  uint64_t h = v0 * 2654435761ULL ^ v1 * 40503ULL;
  return (uint32_t)(h & (__EDGE_HT_SIZE - 1));
}

HOT_LOOP static void __link_internal_cavity(sif_delaunay_tessellation_t* tess,
  cavity_face_t* faces, uint64_t n_faces, _edge_entry_t* ht,
  uint64_t gen_marker) {

  for (uint64_t i = 0; i < n_faces; i++) {
    for (uint8_t e = 0; e < 3; e++) {
      uint64_t vs = faces[i].v[e];
      uint64_t ve = faces[i].v[(e + 1) % 3];
      uint32_t slot = __edge_hash(vs, ve);

      /* Look for an empty slot (one not matching the current marker) */
      while (ht[slot].marker == gen_marker)
        slot = (slot + 1) & (__EDGE_HT_SIZE - 1);

      ht[slot].v_start = vs;
      ht[slot].v_end = ve;
      ht[slot].face_idx = (uint16_t)i;
      ht[slot].edge_idx = e;
      ht[slot].marker = gen_marker;
    }
  }

  for (uint64_t i = 0; i < n_faces; i++) {
    uint64_t tet_A_id = faces[i].new_tet_id;
    sif_tetrahedron_t* tet_A = &tess->tetrahedrons[tet_A_id];

    for (uint8_t e = 0; e < 3; e++) {
      if (tet_A->neighbors[(e + 2) % 3] != UINT64_MAX)
        continue;

      uint64_t vs = faces[i].v[e];
      uint64_t ve = faces[i].v[(e + 1) % 3];
      uint32_t slot = __edge_hash(ve, vs);

      /* Only evaluate slots inserted during THIS specific cavity (gen_marker)
       */
      while (ht[slot].marker == gen_marker) {
        if (ht[slot].v_start == ve && ht[slot].v_end == vs) {
          uint16_t j = ht[slot].face_idx;
          uint8_t k = ht[slot].edge_idx;
          uint64_t tet_B_id = faces[j].new_tet_id;
          tet_A->neighbors[(e + 2) % 3] = tet_B_id;
          tess->tetrahedrons[tet_B_id].neighbors[(k + 2) % 3] = tet_A_id;
          break;
        }
        slot = (slot + 1) & (__EDGE_HT_SIZE - 1);
      }
    }
  }
}

/* --- 5. Main Algorithm --- */

NODISCARD sif_delaunay_tessellation_t* sif_delaunay_tessellation_3d_build(
  sif_field_t* field, uint32_t options) {

  if (!field || field->n_particles < 4) {
    SIF_LOG_ERROR("delaunay", "not enough particles");
    return NULL;
  }

  if (!(field->state_flags & __FIELD_STATE_MORTON_SORTED)) {
    SIF_LOG_ERROR("delaunay", "field is not morton sorted");
    return NULL;
  }

  if (field->n_particles >= UINT64_MAX - 4) {
    SIF_LOG_ERROR("delaunay", "chunk size exceeds 64-bit limits");
    return NULL;
  }

  sif_delaunay_tessellation_t* tess =
    malloc(sizeof(sif_delaunay_tessellation_t));
  tess->field = field;
  tess->capacity = (uint64_t)field->n_particles * 8;
  tess->n_tetrahedrons = 0;
  tess->free_head = UINT64_MAX;

  tess->tetrahedrons = malloc(tess->capacity * sizeof(sif_tetrahedron_t));
  tess->vertex_to_tet =
    sif_malloc_aligned(field->n_particles * sizeof(uint64_t));
  memset(tess->vertex_to_tet, 0xFF, field->n_particles * sizeof(uint64_t));

  uint64_t* visited_array = calloc(tess->capacity, sizeof(uint64_t));
  cavity_face_t* cavity_faces =
    sif_malloc_aligned(__MAX_CAVITY_SIZE * sizeof(cavity_face_t));
  uint64_t* dead_tets =
    sif_malloc_aligned(__MAX_CAVITY_SIZE * sizeof(uint64_t));

  _edge_entry_t* edge_ht =
    sif_calloc_aligned(__EDGE_HT_SIZE, sizeof(_edge_entry_t));

  __init_super_tetrahedron_bounds(field);

  uint64_t super_tet = __alloc_tetrahedron(tess, &visited_array);
  tess->tetrahedrons[super_tet].vertices[0] = (uint64_t)field->n_particles + 0;
  tess->tetrahedrons[super_tet].vertices[1] = (uint64_t)field->n_particles + 1;
  tess->tetrahedrons[super_tet].vertices[2] = (uint64_t)field->n_particles + 2;
  tess->tetrahedrons[super_tet].vertices[3] = (uint64_t)field->n_particles + 3;

  for (int i = 0; i < 4; i++)
    tess->tetrahedrons[super_tet].neighbors[i] = UINT64_MAX;

  uint64_t recent_tet = super_tet;

  for (uint64_t i = 0; i < field->n_particles; i++) {
    uint64_t base_tet = __find_containing_tet(tess, field, i, recent_tet);

    uint64_t n_faces = 0, n_dead = 0;
    __dig_cavity(tess, field, i, i + 1, base_tet, cavity_faces, &n_faces,
      dead_tets, &n_dead, visited_array);

    for (uint64_t d = 0; d < n_dead; d++)
      __delete_tetrahedron(dead_tets[d], tess);

    for (uint64_t f = 0; f < n_faces; f++) {
      uint64_t new_tet_id = __alloc_tetrahedron(tess, &visited_array);
      sif_tetrahedron_t* new_tet = &tess->tetrahedrons[new_tet_id];

      new_tet->vertices[0] = cavity_faces[f].v[0];
      new_tet->vertices[1] = cavity_faces[f].v[1];
      new_tet->vertices[2] = cavity_faces[f].v[2];
      new_tet->vertices[3] = i;

      new_tet->neighbors[3] = cavity_faces[f].outside_tet;
      if (cavity_faces[f].outside_tet != UINT64_MAX) {
        tess->tetrahedrons[cavity_faces[f].outside_tet]
          .neighbors[cavity_faces[f].outside_face] = new_tet_id;
      }
      new_tet->neighbors[0] = UINT64_MAX;
      new_tet->neighbors[1] = UINT64_MAX;
      new_tet->neighbors[2] = UINT64_MAX;

      cavity_faces[f].new_tet_id = new_tet_id;
      recent_tet = new_tet_id;
    }

    tess->vertex_to_tet[i] = recent_tet;
    __link_internal_cavity(tess, cavity_faces, n_faces, edge_ht, i + 1);
  }

  /* --- Phase 3 Cleanup & Compaction --- */

  uint64_t* old_to_new =
    sif_malloc_aligned(tess->n_tetrahedrons * sizeof(uint64_t));
  memset(old_to_new, 0xFF, tess->n_tetrahedrons * sizeof(uint64_t));

  uint64_t valid_count = 0;
  for (uint64_t i = 0; i < tess->n_tetrahedrons; i++) {
    sif_tetrahedron_t* t = &tess->tetrahedrons[i];
    if (t->vertices[0] == UINT64_MAX)
      continue;

    uint8_t connects_to_super = 0;
    for (int j = 0; j < 4; j++) {
      if (t->vertices[j] >= field->n_particles) {
        connects_to_super = 1;
        break;
      }
    }

    if (!connects_to_super)
      old_to_new[i] = valid_count++;
  }

  sif_tetrahedron_t* compacted =
    malloc(valid_count * sizeof(sif_tetrahedron_t));
  for (uint64_t i = 0; i < tess->n_tetrahedrons; i++) {
    if (old_to_new[i] != UINT64_MAX) {
      uint64_t new_id = old_to_new[i];
      memcpy(
        &compacted[new_id], &tess->tetrahedrons[i], sizeof(sif_tetrahedron_t));

      for (int j = 0; j < 4; j++) {
        uint64_t old_neighbor = compacted[new_id].neighbors[j];
        if (old_neighbor != UINT64_MAX) {
          compacted[new_id].neighbors[j] = old_to_new[old_neighbor];
        }
      }
    }
  }

  free(tess->tetrahedrons);
  tess->tetrahedrons = compacted;
  tess->capacity = valid_count;
  tess->n_tetrahedrons = valid_count;
  tess->free_head = UINT64_MAX;

  sif_free_aligned(edge_ht);
  sif_free_aligned(old_to_new);
  free(visited_array);
  sif_free_aligned(cavity_faces);
  sif_free_aligned(dead_tets);

  return tess;
}

void sif_delaunay_tessellation_free(sif_delaunay_tessellation_t* tess) {
  if (tess) {
    if (tess->vertex_to_tet)
      sif_free_aligned(tess->vertex_to_tet);
    free(tess->tetrahedrons);
    free(tess);
    return;
  }
  SIF_LOG_WARNING("delaunay", "cannot free a NULL tessellation");
}

/* Fast Volume Calculation */
static inline real_t __tetrahedron_volume(real_t x0, real_t y0, real_t z0,
  real_t x1, real_t y1, real_t z1, real_t x2, real_t y2, real_t z2, real_t x3,
  real_t y3, real_t z3) {

  real_t dx1 = x1 - x0, dy1 = y1 - y0, dz1 = z1 - z0;
  real_t dx2 = x2 - x0, dy2 = y2 - y0, dz2 = z2 - z0;
  real_t dx3 = x3 - x0, dy3 = y3 - y0, dz3 = z3 - z0;

  real_t cross_x = dy2 * dz3 - dz2 * dy3;
  real_t cross_y = dz2 * dx3 - dx2 * dz3;
  real_t cross_z = dx2 * dy3 - dy2 * dx3;

  real_t det = dx1 * cross_x + dy1 * cross_y + dz1 * cross_z;
  return REAL_ABS(det) / 6.0f;
}

/* Fast Barycentric Coordinates */
static inline void __barycentric_coords(real_t px, real_t py, real_t pz,
  real_t x0, real_t y0, real_t z0, real_t x1, real_t y1, real_t z1, real_t x2,
  real_t y2, real_t z2, real_t x3, real_t y3, real_t z3, real_t* lambda) {

  real_t dx1 = x1 - x0, dy1 = y1 - y0, dz1 = z1 - z0;
  real_t dx2 = x2 - x0, dy2 = y2 - y0, dz2 = z2 - z0;
  real_t dx3 = x3 - x0, dy3 = y3 - y0, dz3 = z3 - z0;
  real_t px0 = px - x0, py0 = py - y0, pz0 = pz - z0;

  real_t det = dx1 * (dy2 * dz3 - dz2 * dy3) - dx2 * (dy1 * dz3 - dz1 * dy3) +
               dx3 * (dy1 * dz2 - dz1 * dy2);
  real_t inv_det = 1.0f / det;

  lambda[1] = (px0 * (dy2 * dz3 - dz2 * dy3) - py0 * (dx2 * dz3 - dz2 * dx3) +
                pz0 * (dx2 * dy3 - dy2 * dx3)) *
              inv_det;
  lambda[2] = (dx1 * (py0 * dz3 - pz0 * dy3) - dy1 * (px0 * dz3 - pz0 * dx3) +
                dz1 * (px0 * dy3 - py0 * dx3)) *
              inv_det;
  lambda[3] = (dx1 * (dy2 * pz0 - dz2 * py0) - dy1 * (dx2 * pz0 - dz2 * px0) +
                dz1 * (dx2 * py0 - dy2 * px0)) *
              inv_det;
  lambda[0] = 1.0f - lambda[1] - lambda[2] - lambda[3];
}

real_t* sif_tessellation_compute_vertex_densities(
  const sif_delaunay_tessellation_t* tess, const sif_field_t* field) {

  real_t* vol_sum = sif_calloc_aligned(field->n_particles, sizeof(real_t));
  if (!vol_sum)
    return NULL;

  for (uint64_t i = 0; i < tess->n_tetrahedrons; i++) {
    const sif_tetrahedron_t* t = &tess->tetrahedrons[i];

    uint64_t v0 = t->vertices[0], v1 = t->vertices[1];
    uint64_t v2 = t->vertices[2], v3 = t->vertices[3];

    real_t vol = __tetrahedron_volume(field->x[v0], field->y[v0], field->z[v0],
      field->x[v1], field->y[v1], field->z[v1], field->x[v2], field->y[v2],
      field->z[v2], field->x[v3], field->y[v3], field->z[v3]);

    vol_sum[v0] += vol;
    vol_sum[v1] += vol;
    vol_sum[v2] += vol;
    vol_sum[v3] += vol;
  }

  real_t* densities = sif_malloc_aligned(field->n_particles * sizeof(real_t));
  for (uint64_t i = 0; i < field->n_particles; i++) {
    if (vol_sum[i] > 0.0f) {
      real_t mass = (field->masses) ? field->masses[i] : 1.0f;
      densities[i] = (4.0f * mass) / vol_sum[i];
    } else {
      densities[i] = 0.0f;
    }
  }

  sif_free_aligned(vol_sum);
  return densities;
}

uint64_t sif_tessellation_find_tetrahedron(
  const sif_delaunay_tessellation_t* tess, const sif_field_t* field, real_t px,
  real_t py, real_t pz, uint64_t start_tet) {

  uint64_t curr = start_tet;

  if (curr == UINT64_MAX || curr >= tess->n_tetrahedrons)
    curr = 0;

  real_t lambdas[4];
  int max_steps = 1000;

  while (max_steps--) {
    const sif_tetrahedron_t* t = &tess->tetrahedrons[curr];

    uint64_t v0 = t->vertices[0], v1 = t->vertices[1];
    uint64_t v2 = t->vertices[2], v3 = t->vertices[3];

    __barycentric_coords(px, py, pz, field->x[v0], field->y[v0], field->z[v0],
      field->x[v1], field->y[v1], field->z[v1], field->x[v2], field->y[v2],
      field->z[v2], field->x[v3], field->y[v3], field->z[v3], lambdas);

    int best_face = -1;
    real_t min_lambda = -1e-5f;

    for (int i = 0; i < 4; i++) {
      if (lambdas[i] < min_lambda) {
        min_lambda = lambdas[i];
        best_face = i;
      }
    }

    if (best_face == -1)
      return curr;

    uint64_t neighbor = t->neighbors[best_face];
    if (neighbor == UINT64_MAX)
      return UINT64_MAX;

    curr = neighbor;
  }
  return UINT64_MAX;
}

real_t sif_tessellation_interpolate_density(
  const sif_delaunay_tessellation_t* tess, const sif_field_t* field,
  uint64_t tet_id, real_t x, real_t y, real_t z,
  const real_t* vertex_densities) {

  const sif_tetrahedron_t* t = &tess->tetrahedrons[tet_id];
  uint64_t v[4] = {
    t->vertices[0], t->vertices[1], t->vertices[2], t->vertices[3]};

  real_t lambdas[4];
  __barycentric_coords(x, y, z, field->x[v[0]], field->y[v[0]], field->z[v[0]],
    field->x[v[1]], field->y[v[1]], field->z[v[1]], field->x[v[2]],
    field->y[v[2]], field->z[v[2]], field->x[v[3]], field->y[v[3]],
    field->z[v[3]], lambdas);

  real_t density = 0.0f;
  for (int i = 0; i < 4; i++) {
    real_t L = lambdas[i];
    if (L < 0.0f)
      L = 0.0f;
    if (L > 1.0f)
      L = 1.0f;

    if (v[i] < field->n_particles) {
      density += L * vertex_densities[v[i]];
    }
  }
  return density;
}

void sif_tessellation_interpolate_velocity(
  const sif_delaunay_tessellation_t* tess, const sif_field_t* field,
  uint64_t tet_id, real_t x, real_t y, real_t z, real_t* out_vx, real_t* out_vy,
  real_t* out_vz) {

  const sif_tetrahedron_t* t = &tess->tetrahedrons[tet_id];
  uint64_t v[4] = {
    t->vertices[0], t->vertices[1], t->vertices[2], t->vertices[3]};

  real_t lambdas[4];
  __barycentric_coords(x, y, z, field->x[v[0]], field->y[v[0]], field->z[v[0]],
    field->x[v[1]], field->y[v[1]], field->z[v[1]], field->x[v[2]],
    field->y[v[2]], field->z[v[2]], field->x[v[3]], field->y[v[3]],
    field->z[v[3]], lambdas);

  real_t vx = 0.0f, vy = 0.0f, vz = 0.0f;
  for (int i = 0; i < 4; i++) {
    real_t L = lambdas[i];
    if (L < 0.0f)
      L = 0.0f;
    if (L > 1.0f)
      L = 1.0f;

    if (v[i] < field->n_particles) {
      vx += L * field->vx[v[i]];
      vy += L * field->vy[v[i]];
      vz += L * field->vz[v[i]];
    }
  }

  *out_vx = vx;
  *out_vy = vy;
  *out_vz = vz;
}
