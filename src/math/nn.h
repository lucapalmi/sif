#ifndef __SIF_MATH_NN_H__
#define __SIF_MATH_NN_H__

/*
 * Evaluation of a small dense feed-forward network with weights fixed at
 * compile time.
 *
 * Not a public header, and not a machine-learning framework: there is no
 * training, no autodiff, no graph, and no file format. It exists so that an
 * emulator elsewhere in the library can spend its lines on physics rather than
 * on matrix loops, and so that a second emulator does not have to write them
 * again.
 *
 * The weights are `static const` arrays generated from a trained model (see
 * tools/ep_export_c.py), so they live in .rodata: nothing is allocated,
 * nothing is initialized at startup, and the linker drops a network no
 * translation unit references.
 *
 * Double throughout. The networks this serves have a few hundred parameters
 * and are evaluated a few hundred times per call, so the arithmetic is far
 * below the cost of the physics around it; single precision would buy
 * bandwidth nobody is short of and cost the ability to compare against the
 * Python reference to the last bit.
 */

#include <stddef.h>
#include <stdint.h>

#include "sif/core/macros.h"

/* Largest number of layers a network may declare. Raising it costs one pointer
 * per layer in the descriptor and nothing at runtime. */
#define SIF_NN_MAX_LAYERS 8

/* Rows evaluated per pass. The point of batching is that the weights are read
 * once and stay in L1 while a block of inputs streams past them, and a few
 * hundred parameters are already resident after the first row -- so this is
 * chosen for the stack budget rather than for the cache. Two scratch buffers
 * of SIF_NN_BLOCK * SIF_NN_MAX_WIDTH doubles live in sif_nn_eval's frame,
 * which at 16 x 64 is 16 KiB in total: comfortable even on the small stacks
 * OpenMP hands its workers. */
#define SIF_NN_BLOCK 16

typedef enum {
  SIF_NN_LINEAR = 0, /* no activation; what the output layer uses */
  SIF_NN_TANH = 1
} sif_nn_activation_t;

/*
 * @brief One dense layer: out = act(in * W + b).
 *
 * W is row-major with n_in rows and n_out columns, so W[i * n_out + j] is the
 * weight from input i to output j. That is the layout numpy writes with
 * `X @ W`, which is what the exporter emits, and it makes the inner loop over
 * j unit-stride.
 */
typedef struct {
  uint32_t n_in;
  uint32_t n_out;
  const double* W; /* n_in * n_out */
  const double* b; /* n_out */
  sif_nn_activation_t act;
} sif_nn_layer_t;

/*
 * @brief A network, plus the input standardization it was fitted with.
 *
 * `mu` and `sd` are not decoration: the fit standardized its inputs, so an
 * evaluation that skips this is evaluating a different function. They are
 * carried here rather than folded into the first layer's weights so that the
 * C constants can be compared against the Python ones term by term.
 */
typedef struct {
  uint32_t n_layers;
  uint32_t n_in;  /* == layers[0].n_in */
  uint32_t n_out; /* == layers[n_layers - 1].n_out */
  const double* mu; /* n_in; subtracted before the first layer */
  const double* sd; /* n_in; divides after mu */
  sif_nn_layer_t layers[SIF_NN_MAX_LAYERS];
} sif_nn_t;

/*
 * @brief Evaluates the network over a batch of inputs.
 *
 * @param nn Network with all weights non-NULL and consistent dimensions
 * @param x Inputs, row-major, n_rows by nn->n_in, unstandardized
 * @param n_rows Number of inputs; zero is valid and does nothing
 * @param out Outputs, row-major, n_rows by nn->n_out. May not alias `x`.
 *
 * @return SIF_OK, or SIF_ERR_INVALID on a malformed network or NULL argument.
 *
 * @note No allocation: the intermediate activations live in a fixed-size stack
 * buffer, which is what bounds SIF_NN_MAX_WIDTH below. Reentrant and safe to
 * call from several threads on the same `nn`, which is const throughout.
 */
int sif_nn_eval(const sif_nn_t* nn, const double* x, uint32_t n_rows,
  double* out);

/* Widest layer the stack buffer in sif_nn_eval can hold. Exceeding it is
 * reported rather than overflowing; raise it if a future model needs it. */
#define SIF_NN_MAX_WIDTH 64

#endif /* __SIF_MATH_NN_H__ */
