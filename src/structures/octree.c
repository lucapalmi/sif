/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/octree.h"

#include "sif/utils/logger.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define OCTREE_LEAF_NODE UINT32_MAX
/* Past SIF_MORTON_BITS levels the quantized coordinates are identical, so
 * there is nothing left to split on. */
#define OCTREE_MAX_DEPTH SIF_MORTON_BITS

/*
 * Nodes to allocate up front, from the leaf count the split threshold implies.
 *
 * A tree of L leaves holds (8L - 1) / 7 nodes when every internal node splits
 * fully, which is the ceiling rather than the typical case: clustered data
 * leaves many of the eight children empty. Guessing high wastes a little
 * memory, guessing low costs a copy of the whole node array per doubling, so
 * this aims at the ceiling and lets the doubling in octree_subdivide() cover
 * the rest.
 */
static uint32_t octree_initial_capacity(
  uint64_t n_particles, uint32_t max_per_leaf) {

  const uint64_t leaves = (n_particles + max_per_leaf - 1) / max_per_leaf;
  uint64_t nodes = (8 * leaves + 6) / 7 + 8;

  if (nodes < 64)
    nodes = 64;
  if (nodes > UINT32_MAX / 2)
    nodes = UINT32_MAX / 2;

  return (uint32_t)nodes;
}

static void octree_subdivide(sif_octree_t* tree, uint32_t node_idx,
  const sif_field_t* field, sif_real cx, sif_real cy, sif_real cz,
  sif_real half_span, uint32_t max_per_leaf, uint32_t depth);

sif_octree_t* sif_octree_alloc(sif_field_t* field, uint32_t max_per_leaf) {
  if (!field || field->n_particles == 0 || max_per_leaf == 0) {
    SIF_LOG_ERROR("octree", "invalid field or max_per_leaf");
    return NULL;
  }

  /* The octree stores one contiguous particle range per node, which is only
   * meaningful because Morton order is depth-first octree order. Make the
   * precondition hold instead of failing on it. */
  if (sif_field_require_morton(field) != SIF_OK) {
    SIF_LOG_ERROR("octree", "failed to morton-sort the field");
    return NULL;
  }

  if (sif_field_require_bounds(field) != SIF_OK) {
    SIF_LOG_ERROR("octree", "failed to compute the field bounds");
    return NULL;
  }

  sif_octree_t* tree = malloc(sizeof(sif_octree_t));
  if (!tree) {
    SIF_LOG_ERROR(
      "octree", "failed to allocate octree (%zu bytes)", sizeof(sif_octree_t));
    return NULL;
  }

  const uint32_t capacity =
    octree_initial_capacity(field->n_particles, max_per_leaf);

  tree->nodes = malloc((size_t)capacity * sizeof(sif_octree_node_t));
  if (!tree->nodes) {
    SIF_LOG_ERROR("octree", "failed to allocate octree nodes (%zu bytes)",
      (size_t)capacity * sizeof(sif_octree_node_t));
    free(tree);
    return NULL;
  }

  tree->capacity = capacity;

  tree->root_center[0] = field->center[0];
  tree->root_center[1] = field->center[1];
  tree->root_center[2] = field->center[2];
  tree->root_half_span = field->half_span;

  tree->count = 1;
  tree->nodes[0].first_child = OCTREE_LEAF_NODE;
  tree->nodes[0].p_start = 0;
  tree->nodes[0].p_counting = field->n_particles;
  tree->nodes[0].padding = 0;

  octree_subdivide(tree, 0, field, tree->root_center[0], tree->root_center[1],
    tree->root_center[2], tree->root_half_span, max_per_leaf, 0);

  return tree;
}

void sif_octree_free(sif_octree_t* tree) {
  if (!tree)
    return;

  free(tree->nodes);
  free(tree);
}

/*
 * Splits one node into eight children.
 *
 * cx/cy/cz/half_span describe the node's cube and are only used to hand the
 * children their geometry; the partition itself is driven by the quantized
 * Morton coordinates so it agrees exactly with the sorted particle order.
 */
static void octree_subdivide(sif_octree_t* tree, uint32_t node_idx,
  const sif_field_t* field, sif_real cx, sif_real cy, sif_real cz,
  sif_real half_span, uint32_t max_per_leaf, uint32_t depth) {

  /* Base Case */
  if (tree->nodes[node_idx].p_counting <= max_per_leaf) {
    return;
  }

  /* Safety net. The partition below relies on the particles being grouped by
   * octant along the array, which sif_field_sort_morton guarantees because it
   * quantizes against this same bounding cube. Two coincident particles can
   * still never be separated, so the depth cap stops the recursion rather than
   * letting it run away. */
  if (depth >= OCTREE_MAX_DEPTH) {
    /* Everything left in this node shares the same quantized cell, i.e. the
     * points are coincident to within side/2^21. Nothing can separate them. */
    SIF_LOG_TRACE("octree",
      "reached the Morton resolution limit with %u coincident particles",
      tree->nodes[node_idx].p_counting);
    return;
  }

  /* Reallocate if we don't have room for 8 more children */
  if (tree->count + 8 > tree->capacity) {
    uint32_t new_capacity = tree->capacity * 2;
    sif_octree_node_t* new_nodes =
      realloc(tree->nodes, (size_t)new_capacity * sizeof(sif_octree_node_t));
    if (!new_nodes) {
      SIF_LOG_ERROR(
        "octree", "failed to grow the node array to %u entries", new_capacity);
      return;
    }
    tree->nodes = new_nodes;
    tree->capacity = new_capacity;
  }

  uint32_t first_child_idx = tree->count;
  tree->nodes[node_idx].first_child = first_child_idx;
  tree->count += 8;

  /* Initialize the 8 children as empty leaves */
  for (int i = 0; i < 8; i++) {
    tree->nodes[first_child_idx + i].first_child = OCTREE_LEAF_NODE;
    tree->nodes[first_child_idx + i].p_start = 0;
    tree->nodes[first_child_idx + i].p_counting = 0;
    tree->nodes[first_child_idx + i].padding = 0;
  }

  uint32_t start_idx = tree->nodes[node_idx].p_start;
  uint32_t end_idx = start_idx + tree->nodes[node_idx].p_counting;

  /* Count first, then lay the children out by prefix sum.
   *
   * The previous version tracked a single "current child" cursor while
   * scanning, which silently produced overlapping or out-of-order ranges the
   * moment two octants interleaved. Counting cannot do that: the eight ranges
   * always tile the parent's range exactly, whatever order the particles
   * arrive in. */
  uint32_t counts[8] = {0};

  /* Classify by the SAME quantization the Morton sort used, not by a float
   * comparison against a recomputed center. The two only agree to within
   * rounding, and a disagreement puts a particle in a sibling's range. */
  const sif_real origin_x = tree->root_center[0] - tree->root_half_span;
  const sif_real origin_y = tree->root_center[1] - tree->root_half_span;
  const sif_real origin_z = tree->root_center[2] - tree->root_half_span;
  const sif_real inv_side = 1.0f / (2.0f * tree->root_half_span);
  const uint32_t bit = SIF_MORTON_BITS - 1u - depth;

  for (uint32_t i = start_idx; i < end_idx; i++) {
    const uint32_t qx = sif_field_quantize(field->x[i], origin_x, inv_side);
    const uint32_t qy = sif_field_quantize(field->y[i], origin_y, inv_side);
    const uint32_t qz = sif_field_quantize(field->z[i], origin_z, inv_side);

    const uint8_t octant =
      (uint8_t)((((qx >> bit) & 1u) << 0) | (((qy >> bit) & 1u) << 1) |
                (((qz >> bit) & 1u) << 2));
    counts[octant]++;
  }

  uint32_t running = start_idx;
  for (int i = 0; i < 8; i++) {
    tree->nodes[first_child_idx + i].p_start = running;
    tree->nodes[first_child_idx + i].p_counting = counts[i];
    running += counts[i];
  }

  SIF_ASSERT(running == end_idx);

  sif_real quarter_span = half_span * 0.5f;
  for (int i = 0; i < 8; i++) {
    if (tree->nodes[first_child_idx + i].p_counting > max_per_leaf) {
      sif_real child_cx = cx + ((i & 1) ? quarter_span : -quarter_span);
      sif_real child_cy = cy + ((i & 2) ? quarter_span : -quarter_span);
      sif_real child_cz = cz + ((i & 4) ? quarter_span : -quarter_span);

      octree_subdivide(tree, first_child_idx + i, field, child_cx, child_cy,
        child_cz, quarter_span, max_per_leaf, depth + 1);
    }
  }
}

typedef struct {
  uint32_t node_idx;
  sif_real cx, cy, cz;
  sif_real half_span;
} octree_stack_t;

/* --- Math Helper: Point-to-Box Distance Squared --- */
static inline sif_real sq_dist_point_aabb(sif_real px, sif_real py, sif_real pz,
  sif_real cx, sif_real cy, sif_real cz, sif_real hs) {
  sif_real min_x = cx - hs, max_x = cx + hs;
  sif_real min_y = cy - hs, max_y = cy + hs;
  sif_real min_z = cz - hs, max_z = cz + hs;

  sif_real sq_dist = 0.0;
  sif_real v;

  v = px;
  if (v < min_x)
    sq_dist += (min_x - v) * (min_x - v);
  else if (v > max_x)
    sq_dist += (v - max_x) * (v - max_x);
  v = py;
  if (v < min_y)
    sq_dist += (min_y - v) * (min_y - v);
  else if (v > max_y)
    sq_dist += (v - max_y) * (v - max_y);
  v = pz;
  if (v < min_z)
    sq_dist += (min_z - v) * (min_z - v);
  else if (v > max_z)
    sq_dist += (v - max_z) * (v - max_z);

  return sq_dist;
}

uint64_t sif_octree_find_nearest(const sif_octree_t* tree,
  const sif_field_t* field, sif_real px, sif_real py, sif_real pz) {

  if (!tree || tree->count == 0 || !field || field->n_particles == 0) {
    SIF_LOG_ERROR("octree", "invalid tree or field");
    return UINT64_MAX;
  }

  /* Branch-and-bound descent.
   *
   * The previous implementation walked down to the single leaf containing the
   * query point and searched only that leaf, which is wrong roughly whenever
   * the true nearest neighbour sits just across a cell boundary. This visits
   * the octant containing the query first (to get a tight bound quickly) and
   * then prunes every subtree whose bounding box is already farther than the
   * best candidate found so far. */

  /* Each pop can push at most 8 children, and the depth is capped at build
   * time, so this bound cannot be exceeded. */
  octree_stack_t stack[8 * OCTREE_MAX_DEPTH];
  int32_t sp = 0;

  stack[sp].node_idx = 0;
  stack[sp].cx = tree->root_center[0];
  stack[sp].cy = tree->root_center[1];
  stack[sp].cz = tree->root_center[2];
  stack[sp].half_span = tree->root_half_span;
  sp++;

  sif_real best_dist2 = SIF_REAL_MAX_VAL;
  uint64_t best_p = UINT64_MAX;

  while (sp > 0) {
    const octree_stack_t cur = stack[--sp];

    /* The bound may have tightened since this entry was pushed. */
    if (sq_dist_point_aabb(px, py, pz, cur.cx, cur.cy, cur.cz, cur.half_span) >=
        best_dist2) {
      continue;
    }

    const sif_octree_node_t* nd = &tree->nodes[cur.node_idx];

    if (nd->first_child == OCTREE_LEAF_NODE) {
      const uint32_t end = nd->p_start + nd->p_counting;
      for (uint32_t i = nd->p_start; i < end; i++) {
        const sif_real dx = field->x[i] - px;
        const sif_real dy = field->y[i] - py;
        const sif_real dz = field->z[i] - pz;
        const sif_real d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best_dist2) {
          best_dist2 = d2;
          best_p = i;
        }
      }
      continue;
    }

    /* Push the octant holding the query point last so it is popped first. */
    uint8_t near_octant = 0;
    if (px >= cur.cx)
      near_octant |= 1;
    if (py >= cur.cy)
      near_octant |= 2;
    if (pz >= cur.cz)
      near_octant |= 4;

    const sif_real q = cur.half_span * 0.5f;

    for (int pass = 0; pass < 2; pass++) {
      for (int i = 0; i < 8; i++) {
        const int is_near = (i == (int)near_octant);
        if ((pass == 0) == is_near)
          continue; /* far children on pass 0, the near child on pass 1 */

        const uint32_t child = nd->first_child + (uint32_t)i;
        if (tree->nodes[child].p_counting == 0)
          continue;

        const sif_real ccx = cur.cx + ((i & 1) ? q : -q);
        const sif_real ccy = cur.cy + ((i & 2) ? q : -q);
        const sif_real ccz = cur.cz + ((i & 4) ? q : -q);

        if (sq_dist_point_aabb(px, py, pz, ccx, ccy, ccz, q) >= best_dist2)
          continue;

        SIF_ASSERT(sp < (int32_t)(sizeof(stack) / sizeof(stack[0])));
        stack[sp].node_idx = child;
        stack[sp].cx = ccx;
        stack[sp].cy = ccy;
        stack[sp].cz = ccz;
        stack[sp].half_span = q;
        sp++;
      }
    }
  }

  return best_p;
}

uint64_t sif_octree_search_radius(const sif_octree_t* tree,
  const sif_field_t* field, sif_real px, sif_real py, sif_real pz,
  sif_real radius, uint64_t* out_indices, uint64_t max_capacity) {
  if (!tree || tree->count == 0 || !field || !field->_position_block ||
      !out_indices || max_capacity == 0) {
    SIF_LOG_ERROR("octree", "invalid tree, field or output array");
    return 0;
  }

  sif_real radius2 = radius * radius;
  uint64_t found_count = 0;

  octree_stack_t stack[OCTREE_MAX_DEPTH];
  int top = 0;

  stack[top++] = (octree_stack_t){0, tree->root_center[0], tree->root_center[1],
    tree->root_center[2], tree->root_half_span};

  while (top > 0) {
    octree_stack_t curr = stack[--top];
    sif_octree_node_t* node = &tree->nodes[curr.node_idx];

    if (node->first_child == UINT32_MAX) {
      uint32_t start = node->p_start;
      uint32_t end = start + node->p_counting;

      for (uint32_t i = start; i < end; i++) {
        sif_real dx = field->x[i] - px;
        sif_real dy = field->y[i] - py;
        sif_real dz = field->z[i] - pz;
        if ((dx * dx + dy * dy + dz * dz) <= radius2) {
          if (found_count < max_capacity) {
            out_indices[found_count] = i;
          }
          found_count++;
        }
      }
    } else {
      sif_real q_span = curr.half_span * 0.5f;
      for (int i = 0; i < 8; i++) {
        sif_octree_node_t* child = &tree->nodes[node->first_child + i];

        if (child->first_child == UINT32_MAX && child->p_counting == 0)
          continue;

        sif_real child_cx = curr.cx + ((i & 1) ? q_span : -q_span);
        sif_real child_cy = curr.cy + ((i & 2) ? q_span : -q_span);
        sif_real child_cz = curr.cz + ((i & 4) ? q_span : -q_span);

        if (sq_dist_point_aabb(
              px, py, pz, child_cx, child_cy, child_cz, q_span) <= radius2) {
          stack[top++] = (octree_stack_t){
            node->first_child + i, child_cx, child_cy, child_cz, q_span};
        }
      }
    }
  }

  return found_count;
}

uint64_t sif_octree_search_box(const sif_octree_t* tree,
  const sif_field_t* field, sif_real min_x, sif_real min_y, sif_real min_z,
  sif_real max_x, sif_real max_y, sif_real max_z, uint64_t* out_indices,
  uint64_t max_capacity) {
  if (!tree || tree->count == 0 || !field || !field->_position_block ||
      !out_indices || max_capacity == 0) {
    SIF_LOG_ERROR("octree", "invalid tree, field or output array");
    return 0;
  }

  uint64_t found_count = 0;
  octree_stack_t stack[OCTREE_MAX_DEPTH];
  int top = 0;

  stack[top++] = (octree_stack_t){0, tree->root_center[0], tree->root_center[1],
    tree->root_center[2], tree->root_half_span};

  while (top > 0) {
    octree_stack_t curr = stack[--top];
    sif_octree_node_t* node = &tree->nodes[curr.node_idx];

    if (node->first_child == UINT32_MAX) {
      uint32_t start = node->p_start;
      uint32_t end = start + node->p_counting;

      for (uint32_t i = start; i < end; i++) {
        sif_real px = field->x[i], py = field->y[i], pz = field->z[i];
        if (px >= min_x && px <= max_x && py >= min_y && py <= max_y &&
            pz >= min_z && pz <= max_z) {
          if (found_count < max_capacity) {
            out_indices[found_count] = i;
          }
          found_count++;
        }
      }
    } else {
      sif_real q_span = curr.half_span * 0.5f;
      for (int i = 0; i < 8; i++) {
        sif_octree_node_t* child = &tree->nodes[node->first_child + i];
        if (child->first_child == UINT32_MAX && child->p_counting == 0)
          continue;

        sif_real child_cx = curr.cx + ((i & 1) ? q_span : -q_span);
        sif_real child_cy = curr.cy + ((i & 2) ? q_span : -q_span);
        sif_real child_cz = curr.cz + ((i & 4) ? q_span : -q_span);

        if (min_x <= child_cx + q_span && max_x >= child_cx - q_span &&
            min_y <= child_cy + q_span && max_y >= child_cy - q_span &&
            min_z <= child_cz + q_span && max_z >= child_cz - q_span) {

          stack[top++] = (octree_stack_t){
            node->first_child + i, child_cx, child_cy, child_cz, q_span};
        }
      }
    }
  }

  return found_count;
}
