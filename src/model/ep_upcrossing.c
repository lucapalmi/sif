/*
 * The semi-analytic first crossing: Musso & Sheth's up-crossing rate for a
 * correlated walk against a moving barrier, and the local description of the
 * walk it is built from.
 *
 * Everything here is closed form and costs microseconds, against seconds for
 * the Monte Carlo in excursionset.c.
 *
 * The walk runs in its own time variable S = sigma^2(R), which increases as R
 * falls. Everything below is written in S; the caller's ascending radius order
 * is preserved throughout, so index i is always radii[i] and S DESCENDS with
 * the index.
 */

#include "model/ep_internal.h"

#include "sif/model/deltamoments.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>

#define __TAG "ep"

/*
 * Above this the bracket below is evaluated from its asymptotic series rather
 * than from the closed form. phi(y) - y (1 - Phi(y)) is a difference of two
 * nearly equal numbers for large positive y -- both approach phi(y)/y -- and
 * the closed form loses every significant digit there, while four terms of the
 * series are good to better than 1e-12.
 */
#define __EP_Y_ASYMPTOTIC 5.0

/* 1 - Phi(x), the upper tail of the standard normal. Shared with the emulator,
 * which needs it for the walk's first step. */
double sif_ep_upper_tail(double x) {
  return 0.5 * erfc(x / sqrt(2.0));
}

/*
 * E[(z - y)^+] for a standard normal z, i.e. phi(y) - y (1 - Phi(y)).
 *
 * This is the whole of the barrier-motion dependence: y measures how fast the
 * barrier runs away from the walk in units of the walk's own slope scatter, so
 * a large positive y is a barrier fleeing upward and almost never crossed.
 */
static double __mean_excess(double y) {

  if (y > __EP_Y_ASYMPTOTIC) {
    /* phi(y) (1/y^2 - 3/y^4 + 15/y^6 - 105/y^8), from the tail expansion of
     * 1 - Phi. Positive and monotonically decreasing, as the exact form is. */
    const double phi = exp(-0.5 * y * y) / sqrt(2.0 * M_PI);
    const double y2 = y * y;
    const double y4 = y2 * y2;
    return phi * (1.0 / y2) * (1.0 - 3.0 / y2 + 15.0 / y4 - 105.0 / (y4 * y2));
  }

  const double phi = exp(-0.5 * y * y) / sqrt(2.0 * M_PI);
  return phi - y * sif_ep_upper_tail(y);
}

/* --- The local description --- */

int sif_ep_features_init(sif_ep_features_t* f, uint32_t n) {

  f->n = n;
  f->S = NULL;
  f->nu = NULL;
  f->gamma2 = NULL;
  f->y = NULL;
  f->f_up = NULL;

  double** all[] = {&f->S, &f->nu, &f->gamma2, &f->y, &f->f_up};

  for (size_t a = 0; a < sizeof(all) / sizeof(all[0]); a++) {
    *all[a] = sif_calloc_aligned((size_t)n, sizeof(double));
    if (!*all[a]) {
      SIF_LOG_ERROR(__TAG, "failed to allocate the walk description");
      return SIF_ERR_ALLOC;
    }
  }

  return SIF_OK;
}

void sif_ep_features_free(sif_ep_features_t* f) {
  if (!f)
    return;
  sif_free_aligned(f->S);
  sif_free_aligned(f->nu);
  sif_free_aligned(f->gamma2);
  sif_free_aligned(f->y);
  sif_free_aligned(f->f_up);
  f->S = NULL;
  f->nu = NULL;
  f->gamma2 = NULL;
  f->y = NULL;
  f->f_up = NULL;
  f->n = 0;
}

/*
 * <(d delta / dS)^2> by differencing the covariance, for callers who did not
 * ask sif_delta_covariance_pk for the exact form. Only first order, whatever
 * the spacing -- see the header.
 */
static void __deriv_variance_differenced(
  const double* cov, const double* S, uint32_t n, double* V) {

  for (uint32_t i = 1; i + 1 < n; i++) {
    const double h = S[i + 1] - S[i - 1];
    V[i] =
      (S[i + 1] + S[i - 1] - 2.0 * sif_ep_cov_get(cov, i + 1, i - 1)) / (h * h);
  }

  /* One-sided at the ends, so no radius is left without a value. */
  if (n >= 3) {
    double h = S[2] - S[0];
    V[0] = (S[2] + S[0] - 2.0 * sif_ep_cov_get(cov, 2, 0)) / (h * h);
    h = S[n - 1] - S[n - 3];
    V[n - 1] =
      (S[n - 1] + S[n - 3] - 2.0 * sif_ep_cov_get(cov, n - 1, n - 3)) / (h * h);
  }
}

/*
 * df/dx on a NON-UNIFORM grid, second order.
 *
 * The obvious (f[i+1] - f[i-1]) / (x[i+1] - x[i-1]) is second order only when
 * the two spacings match, and here they emphatically do not: S = sigma^2 over
 * a log-spaced radius grid is spaced wildly unevenly. Dropping the f[i] term
 * costs a factor of ten in accuracy on a realistic grid, which is enough to
 * shift `y` and with it the up-crossing rate.
 *
 *   f' = [h1^2 f[i+1] - h2^2 f[i-1] - (h1^2 - h2^2) f[i]]
 *        / (h1 h2 (h1 + h2)),   h1 = x[i] - x[i-1], h2 = x[i+1] - x[i]
 *
 * Signs are irrelevant: x descends here, and the formula does not care.
 */
static double __deriv_nonuniform(
  const double* fv, const double* x, uint32_t n, uint32_t i) {

  if (n < 3)
    return (fv[n - 1] - fv[0]) / (x[n - 1] - x[0]);

  /* Second-order one-sided at the ends, from the same three-point stencil. */
  if (i == 0) {
    const double h1 = x[1] - x[0];
    const double h2 = x[2] - x[1];
    return -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * fv[0] +
           (h1 + h2) / (h1 * h2) * fv[1] - h1 / (h2 * (h1 + h2)) * fv[2];
  }

  if (i + 1 == n) {
    const double h1 = x[n - 2] - x[n - 3];
    const double h2 = x[n - 1] - x[n - 2];
    return h2 / (h1 * (h1 + h2)) * fv[n - 3] -
           (h1 + h2) / (h1 * h2) * fv[n - 2] +
           (h1 + 2.0 * h2) / (h2 * (h1 + h2)) * fv[n - 1];
  }

  const double h1 = x[i] - x[i - 1];
  const double h2 = x[i + 1] - x[i];

  return (h1 * h1 * fv[i + 1] - h2 * h2 * fv[i - 1] -
           (h1 * h1 - h2 * h2) * fv[i]) /
         (h1 * h2 * (h1 + h2));
}

/*
 * The physics, once: f->S is already filled and V is already chosen, so this
 * is shared verbatim between the covariance entry point and the diagonal one
 * the emulator uses. Splitting it is not tidiness -- the network was fitted
 * against these exact expressions, so a second copy that drifted would be a
 * silently different model.
 */
static int __fill_core(sif_ep_features_t* f, const real_t* radii, uint32_t n,
  const real_t* barrier, const double* V) {

  /* Promoted once: the barrier arrives as real_t but every derivative below
   * is taken in double, and differencing a float array in double precision
   * would carry the float's rounding into the derivative rather than the
   * value. */
  double* Bd = malloc((size_t)n * sizeof(double));
  if (!Bd) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the promoted barrier");
    return SIF_ERR_ALLOC;
  }
  for (uint32_t i = 0; i < n; i++)
    Bd[i] = (double)barrier[i];

  int bad = 0;

  for (uint32_t i = 0; i < n; i++) {
    const double S = f->S[i];

    /* Cauchy-Schwarz on <delta delta'> = 1/2, which holds whatever the
     * spectrum: 4 S V >= 1, so gamma2 lands in (0, 1]. A violation means the
     * matrix is not a covariance, or the difference above has gone wrong. */
    const double four_sv = 4.0 * S * V[i];
    if (!(four_sv >= 1.0)) {
      bad++;
      f->gamma2[i] = 1.0;
      f->y[i] = 0.0;
      f->f_up[i] = 0.0;
      continue;
    }

    f->gamma2[i] = 1.0 / four_sv;

    /* Conditional on delta = B, the walk's slope has mean B / 2S -- from
     * <delta delta'> = 1/2 exactly -- and variance V - 1/(4S). */
    const double mu = Bd[i] / (2.0 * S);
    const double sigma_slope = sqrt(V[i] - 1.0 / (4.0 * S));

    const double dB = __deriv_nonuniform(Bd, f->S, n, i);

    f->y[i] = (dB - mu) / sigma_slope;

    const double nu = f->nu[i];
    f->f_up[i] = exp(-0.5 * nu * nu) / sqrt(2.0 * M_PI * S) * sigma_slope *
                 __mean_excess(f->y[i]);

    if (!(f->f_up[i] >= 0.0))
      f->f_up[i] = 0.0;
  }

  free(Bd);

  if (bad > 0) {
    SIF_LOG_ERROR(__TAG,
      "at %d of %u radii the walk correlates with its own derivative more "
      "strongly than Cauchy-Schwarz allows, which means the matrix is not a "
      "covariance; check how it was built",
      bad, n);
    return SIF_ERR_RANGE;
  }

  return SIF_OK;
}

int sif_ep_features_fill_diag(sif_ep_features_t* f, const real_t* radii,
  uint32_t n, const double* S, const real_t* barrier,
  const double* deriv_variance) {

  if (!deriv_variance) {
    SIF_LOG_ERROR(__TAG,
      "the derivative variance is required when only the diagonal is given; "
      "there are no off-diagonal elements left to difference");
    return SIF_ERR_INVALID;
  }

  for (uint32_t i = 0; i < n; i++) {
    if (!(S[i] > 0.0)) {
      SIF_LOG_ERROR(__TAG,
        "sigma^2 at radius %u (%g) is %g; the walk has no scale there", i,
        (double)radii[i], S[i]);
      return SIF_ERR_RANGE;
    }
    f->S[i] = S[i];
    f->nu[i] = (double)barrier[i] / sqrt(S[i]);
  }

  return __fill_core(f, radii, n, barrier, deriv_variance);
}

int sif_ep_features_fill(sif_ep_features_t* f, const real_t* radii, uint32_t n,
  const double* cov, const real_t* barrier, const double* deriv_variance) {

  for (uint32_t i = 0; i < n; i++) {
    f->S[i] = sif_ep_cov_get(cov, i, i);
    if (!(f->S[i] > 0.0)) {
      SIF_LOG_ERROR(__TAG,
        "the covariance diagonal at radius %u (%g) is %g; the walk has no "
        "scale there",
        i, (double)radii[i], f->S[i]);
      return SIF_ERR_RANGE;
    }
    f->nu[i] = (double)barrier[i] / sqrt(f->S[i]);
  }

  const double* V = deriv_variance;
  double* V_owned = NULL;

  if (!V) {
    V_owned = malloc((size_t)n * sizeof(double));
    if (!V_owned) {
      SIF_LOG_ERROR(__TAG, "failed to allocate the derivative variance");
      return SIF_ERR_ALLOC;
    }
    __deriv_variance_differenced(cov, f->S, n, V_owned);
    V = V_owned;

    SIF_LOG_WARNING(__TAG,
      "no derivative variance supplied, so it was differenced off the "
      "covariance; that estimate converges only at first order and is wrong by "
      "several per cent at a realistic radius count, which propagates into the "
      "result. Take it from sif_delta_covariance_pk instead");
  }

  const int rc = __fill_core(f, radii, n, barrier, V);
  free(V_owned);
  return rc;
}

/* --- The hazard and the survival --- */

void sif_ep_hazard_bins(
  const sif_ep_features_t* f, uint32_t n, double* lam) {

  for (uint32_t i = 0; i + 1 < n; i++) {

    const double dS = f->S[i] - f->S[i + 1]; /* positive: S descends */
    const double f0 = f->f_up[i + 1];        /* entered first */
    const double f1 = f->f_up[i];

    /*
     * Integrating the log-linear interpolant is exact for a pure exponential
     * and costs one logarithm:
     *
     *     int f dS  =  (f1 - f0) dS / ln(f1 / f0)
     *
     * the logarithmic mean of the endpoints. It degenerates when the endpoints
     * are close, where the trapezoid is both accurate and stable, so that is
     * the branch taken there.
     */
    double v;

    if (f0 > 0.0 && f1 > 0.0) {
      const double lr = log(f1 / f0);
      v = (fabs(lr) > 1e-6) ? (f1 - f0) * dS / lr : 0.5 * (f0 + f1) * dS;
    } else {
      v = 0.5 * (f0 + f1) * dS;
    }

    lam[i] = (v > 0.0) ? v : 0.0;
  }
}

void sif_ep_survival(const double* lam, const real_t* radii, uint32_t n_bins,
  double alive0, real_t* out) {

  double alive = alive0;

  /* The walk enters the largest-radius bin first and works down. */
  for (int32_t i = (int32_t)n_bins - 1; i >= 0; i--) {
    const double p = alive * (1.0 - exp(-lam[i]));
    alive -= p;

    const double dr = (double)radii[i + 1] - (double)radii[i];
    out[i] = (real_t)(dr > 0.0 ? p / dr : 0.0);
  }
}

/* --- The multiplicity --- */

real_t* sif_ep_multiplicity_upcrossing(
  const sif_ep_features_t* f, const real_t* radii, uint32_t n) {

  const uint32_t n_bins = n - 1;

  real_t* out = sif_calloc_aligned((size_t)n_bins, sizeof(real_t));
  double* lam = malloc((size_t)n_bins * sizeof(double));

  if (!out || !lam) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the multiplicity array");
    sif_free_aligned(out);
    free(lam);
    return NULL;
  }

  sif_ep_hazard_bins(f, n, lam);

  /* Walks that begin above the barrier cross on the walk's first step, at the
   * largest radius, and never enter a bin. That point mass is exactly the
   * one-point tail there, so it is removed rather than estimated. */
  sif_ep_survival(
    lam, radii, n_bins, 1.0 - sif_ep_upper_tail(f->nu[n - 1]), out);

  free(lam);
  return out;
}
