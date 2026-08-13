/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Dense feed-forward evaluation for compile-time networks. See nn.h for what
 * this is and is not.
 *
 * The whole of it is one triple loop. The only decisions worth recording are
 * the loop order and the blocking, both of which exist so that the inner loop
 * is unit-stride and the weights are read once per block rather than once per
 * row.
 */

#include "math/nn.h"

#include "sif/utils/logger.h"

#include <math.h>
#include <string.h>

#define TAG "nn"

static int validate(const sif_nn_t* nn, const double* x, double* out) {

  if (!nn || !x || !out) {
    SIF_LOG_ERROR(TAG, "network, input and output are all required");
    return SIF_ERR_INVALID;
  }

  if (nn->n_layers == 0 || nn->n_layers > SIF__NN_MAX_LAYERS) {
    SIF_LOG_ERROR(
      TAG, "%u layers; the maximum is %d", nn->n_layers, SIF__NN_MAX_LAYERS);
    return SIF_ERR_INVALID;
  }

  if (!nn->mu || !nn->sd) {
    SIF_LOG_ERROR(TAG,
      "the input standardization is missing; a network evaluated without the "
      "mean and scale it was fitted with is a different function");
    return SIF_ERR_INVALID;
  }

  uint32_t width = nn->n_in;

  for (uint32_t l = 0; l < nn->n_layers; l++) {
    const sif_nn_layer_t* ly = &nn->layers[l];

    if (!ly->W || !ly->b) {
      SIF_LOG_ERROR(TAG, "layer %u has no weights", l);
      return SIF_ERR_INVALID;
    }
    if (ly->n_in != width) {
      SIF_LOG_ERROR(TAG,
        "layer %u takes %u inputs but the layer before it produces %u", l,
        ly->n_in, width);
      return SIF_ERR_INVALID;
    }
    if (ly->n_out == 0 || ly->n_out > SIF__NN_MAX_WIDTH) {
      SIF_LOG_ERROR(TAG, "layer %u is %u wide; the scratch buffers hold %d", l,
        ly->n_out, SIF__NN_MAX_WIDTH);
      return SIF_ERR_INVALID;
    }
    width = ly->n_out;
  }

  if (nn->n_in == 0 || nn->n_in > SIF__NN_MAX_WIDTH) {
    SIF_LOG_ERROR(TAG, "%u inputs; the scratch buffers hold %d", nn->n_in,
      SIF__NN_MAX_WIDTH);
    return SIF_ERR_INVALID;
  }

  if (width != nn->n_out) {
    SIF_LOG_ERROR(TAG,
      "the last layer produces %u outputs but the network declares %u", width,
      nn->n_out);
    return SIF_ERR_INVALID;
  }

  return SIF_OK;
}

/*
 * One layer over a block of rows: dst = act(src W + b).
 *
 * The accumulation runs (row, input, output) rather than (row, output, input),
 * which turns the innermost loop into a unit-stride multiply-add over the
 * output width instead of a dot product with a reduction. Both do the same
 * arithmetic; only this one vectorizes without the compiler having to prove
 * anything about reassociation, which matters because the release build is
 * built with -ffast-math off for this file's callers and we want the same
 * answer either way.
 */
static void layer(
  const sif_nn_layer_t* ly, const double* src, double* dst, uint32_t n_rows) {

  const uint32_t n_in = ly->n_in;
  const uint32_t n_out = ly->n_out;

  for (uint32_t r = 0; r < n_rows; r++) {
    const double* a = src + (size_t)r * n_in;
    double* d = dst + (size_t)r * n_out;

    for (uint32_t j = 0; j < n_out; j++)
      d[j] = ly->b[j];

    for (uint32_t i = 0; i < n_in; i++) {
      const double v = a[i];
      const double* Wi = ly->W + (size_t)i * n_out;
      for (uint32_t j = 0; j < n_out; j++)
        d[j] += v * Wi[j];
    }

    if (ly->act == SIF__NN_TANH) {
      for (uint32_t j = 0; j < n_out; j++)
        d[j] = tanh(d[j]);
    }
  }
}

int sif__nn_eval(
  const sif_nn_t* nn, const double* x, uint32_t n_rows, double* out) {

  const int bad = validate(nn, x, out);
  if (bad != SIF_OK)
    return bad;

  if (n_rows == 0)
    return SIF_OK;

  /* Two buffers, ping-ponged between layers. Sized by the widest layer the
   * header admits, not by this network, so the frame is the same whatever is
   * evaluated. */
  double buf_a[SIF__NN_BLOCK * SIF__NN_MAX_WIDTH];
  double buf_b[SIF__NN_BLOCK * SIF__NN_MAX_WIDTH];

  const uint32_t n_in = nn->n_in;
  const uint32_t n_out = nn->n_out;

  for (uint32_t base = 0; base < n_rows; base += SIF__NN_BLOCK) {
    const uint32_t rows =
      (n_rows - base < SIF__NN_BLOCK) ? (n_rows - base) : SIF__NN_BLOCK;

    /* Standardize into the first buffer. The fit saw (x - mu) / sd, so this is
     * part of the model rather than a convenience. */
    for (uint32_t r = 0; r < rows; r++) {
      const double* xr = x + (size_t)(base + r) * n_in;
      double* ar = buf_a + (size_t)r * n_in;
      for (uint32_t i = 0; i < n_in; i++)
        ar[i] = (xr[i] - nn->mu[i]) / nn->sd[i];
    }

    double* src = buf_a;
    double* dst = buf_b;

    for (uint32_t l = 0; l < nn->n_layers; l++) {
      layer(&nn->layers[l], src, dst, rows);
      double* tmp = src;
      src = dst;
      dst = tmp;
    }

    /* After the swap on the last layer, src holds the output. */
    memcpy(
      out + (size_t)base * n_out, src, (size_t)rows * n_out * sizeof(double));
  }

  return SIF_OK;
}
