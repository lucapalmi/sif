# The excursion-set multiplicity emulator

A trained replacement for the Monte Carlo first-crossing calculation
(`sif_ep_multiplicity_function`), accurate to **0.13%** and about **2×10⁶ times
faster**, intended for MCMC use where the Monte Carlo is unaffordable.

This document covers what the emulator emulates and why, the physical meaning
of every quantity it consumes, the training pipeline, and the measured results.

---

## 1. The problem

The excursion-set void multiplicity function is obtained by walking a
correlated Gaussian random field δ(R) from large smoothing radii down to small
ones and recording, for each walk, the largest radius at which it first reaches
a moving barrier B(σ). The Monte Carlo (`src/model/excursion_set.c`) does this
directly: it Cholesky-factorises the covariance S(Rᵢ, Rⱼ), draws correlated
walks, and histograms the first crossings.

That calculation is exact but expensive. At 10⁹ paths over 128 radii it costs
roughly 400 s on a workstation, and its statistical error falls only as
1/√N_paths. An MCMC needing 10⁵–10⁶ likelihood evaluations cannot use it.

**What is emulated is not the multiplicity function itself.** The multiplicity
spans orders of magnitude and carries a factor exp(−ν²/2); a network asked to
reproduce it directly would spend all its capacity on dynamic range that is
already known analytically. Instead we emulate the *ratio* between the Monte
Carlo and a semi-analytic baseline, which is a smooth O(1) quantity.

---

## 2. What is actually emulated

### 2.1 The baseline: Musso–Sheth up-crossing

The baseline is the up-crossing rate of Musso & Sheth — the rate at which the
walk crosses the barrier upward faster than the barrier itself moves. In the
walk's own time variable S = σ²(R) (which *increases* as R falls):

```
f_up(S) = exp(−ν²/2) / √(2π S) · Σ_slope · E[(z − y)⁺]
```

where `E[(z − y)⁺] = φ(y) − y(1 − Φ(y))` is the mean excess of a standard
normal. This is exact in the high-barrier limit, where a first crossing and any
crossing are the same event, and runs a few per cent low for a flat barrier and
up to ~15% for a steep moving one.

### 2.2 The target: a hazard ratio, not a multiplicity ratio

The correction is applied to the **hazard**, not to the binned multiplicity.
Per bin *i* the baseline integrated hazard is

```
Λ_up[i] = ∫ f_up dS   over the bin,   evaluated as the logarithmic mean
                                      of the endpoints (exact for a pure
                                      exponential, which the rate nearly is)
```

and the target the network learns is

```
correction = ln( Λ_MC[i] / Λ_up[i] )
```

with Λ_MC extracted from the raw counts through the survival recursion. This
choice matters for three reasons:

1. **Boundedness is structural.** The multiplicity is rebuilt as
   `f[i] = alive · (1 − exp(−Λ[i])) / Δr`, so however wrong the network is, the
   result is non-negative and integrates to at most one. There is no clamp.
2. **The bins decouple.** A multiplicity ratio would force the network to learn
   the survival bookkeeping the baseline already does exactly.
3. **Errors enter multiplicatively**, i.e. as relative errors — which is what a
   likelihood cares about.

Measured over the whole training set the ratio lies in **[0.91, 1.68]**: a
smooth, O(1), well-conditioned regression target.

---

## 3. The features, and what they mean physically

Eight numbers per radius bin. All are dimensionless or scale-free, which is why
one trained network covers every cosmology and barrier in the domain. None of
them is a cosmological parameter: the network never sees Ω_m, h, n_s, z, or the
barrier parameters (α, β, γ) — only the local and historical description of the
walk they produce.

| # | feature | definition | physical meaning |
|---|---------|-----------|------------------|
| 1 | `nu` | B/σ | Barrier height in units of the field's own scatter. The single most important variable: it sets how rare a crossing is, and the correction falls monotonically towards 1 as it grows. |
| 2 | `gamma2` | 1/(4S⟨(dδ/dS)²⟩) | Squared correlation between the walk and its own derivative. 0 = uncorrelated (Markovian) steps, 1 = a walk whose value determines its slope. Bounded in (0,1] by Cauchy–Schwarz since ⟨δδ′⟩ = 1/2 exactly. **This is how the shape of P(k) enters** — it is essentially an effective spectral index. |
| 3 | `y` | (dB/dS − μ)/Σ_slope | How fast the barrier runs away from the walk, in units of the walk's own slope scatter, conditional on δ = B (where the slope has mean μ = B/2S and variance V − 1/4S). Large positive y is a barrier fleeing upward and almost never crossed. **This is the whole of the barrier-motion dependence.** |
| 4 | `dlnnu_dlnS` | d ln ν / d ln S | Local logarithmic shape of the barrier in the walk's own time. Distinct from `y`, which is normalised by the slope scatter; this is the bare shape. |
| 5 | `nu_back` | ν at S/2 | ν half an e-fold back along the walk, clamped at the walk's origin. A one-point summary of recent history. |
| 6 | `lag` | ln(S/S_lookback) | How far back the lookback actually reached — ln 2 normally, less where clamped at the origin. Tells the network a full-history bin from a truncated one. |
| 7 | `cum_lam` | Σ_{j>i} Λ_up[j] | Baseline hazard accumulated *before* entering this bin, i.e. how much crossing probability has already been spent. **Summarises the entire prior history in one scalar**, and is what the surviving population is conditioned on. |
| 8 | `nu_origin` | ν at the largest radius | Where the walk started. First crossing genuinely depends on the starting scale, so this dependence is made explicit rather than assumed away. |

### Why these, and not others

Features 1–3 are the local description the baseline itself uses. Features 4–8
were each added because a measurement demanded it, not by guesswork:

- **A pilot at 2×10⁷ paths** concluded that the local triple (ν, γ², y)
  saturated and that history features added nothing. **That conclusion was
  wrong**, and wrong for an instructive reason: it was drawn against a Monte
  Carlo noise floor of 8×10⁻³ while the fit sat at 1.1×10⁻², only 1.4× above
  it. The floor was masking the effect.
- **Repeated at 10⁹ paths** (floor 6×10⁻⁴), adding `dlnnu_dlnS` and the
  lookback pair cut the weighted RMS by 40%, from 5.4×10⁻³ to 3.2×10⁻³.
- **A dedicated diagnosis** then showed the residual error was concentrated at
  low ν — the small-radius end, which the walk reaches *last*, where survivors
  are a strongly conditioned population. Adding `cum_lam` and `nu_origin` — the
  natural summaries of that conditioning — cut the low-ν error by a further
  **42%** and flattened the error profile across the whole ν range.
- A second lookback point and a γ² history were tested and rejected: ~2% gain,
  not worth carrying into C.

Notably, the survival probability `exp(−cum_lam)` on its own made things
*worse*; it saturates near 1 and loses resolution. The log-scale cumulative
hazard is the informative parametrisation.

### What was ruled out

| hypothesis | test | result |
|---|---|---|
| model too small | 385 → 2689 parameters | 13% gain, high-ν *worse*. Not capacity. |
| optimisation variance | ensemble of 1, 3, 5 networks | 0.137% → 0.139% → 0.137%. **Literally nothing.** Different seeds converge to the same function. |
| systematic misspecification | mean vs RMS residual per ν band | bias/RMS = 0.01–0.06. Unbiased; pure scatter. |
| training-data noise | model residual vs MC noise per band | ratio 1.00–1.07 at ν>3. Data-limited there; the model error is unmeasurably small. |

---

## 4. Network architecture

```
8 inputs → standardise → 16 (tanh) → 16 (tanh) → 1 linear  = 433 parameters
```

- **Input standardisation**: each feature shifted by its training mean and
  scaled by its training standard deviation. Both vectors ship with the weights.
- **Output**: ln(Λ_MC/Λ_up). The correction is `exp(output)`, positive by
  construction.
- **Loss**: unweighted mean squared error. *Flat weights, not
  inverse-variance* — this is deliberate. The Monte Carlo error varies 25×
  across the ν range, and inverse-variance weighting was measured to underfit
  the high-ν end by 2.5× (0.97% vs 0.39%) while leaving the overall RMS
  unchanged at 4.2×10⁻³. The statistically "optimal" weighting was optimal for
  a question nobody asked; the large-radius end carries the cosmological
  information and gets equal say.
- **Optimiser**: L-BFGS-B with analytic gradients, 1500 iterations, L2 = 10⁻⁶.
- **Training sample**: 60,000 bins drawn from the 190,989 usable ones.

Deliberately small: three of the eight inputs dominate, the target is nearly
one-dimensional in ν, and every parameter becomes a constant in the C source.

---

## 5. The training set

### 5.1 Domain

| axis | range | notes |
|---|---|---|
| Ω_c h² | 0.08 – 0.18 | wider than a realistic prior, so chains sit in the interior |
| Ω_b h² | 0.018 – 0.028 | |
| H₀ | 55 – 80 | |
| n_s | 0.90 – 1.02 | |
| Σm_ν | 0 – 0.5 eV | massive neutrinos included |
| z | 0 – 2 | |
| α (barrier) | 0.05 – 2.50 | B(σ) = α[1 + (β/σ)^γ] |
| β (barrier) | 0.01 – 1.50 | floored strictly positive: β = 0 makes (β/σ)^γ undefined |
| γ (barrier) | 0.50 – 3.50 | |

64 cosmologies × 24 barriers = **1536 curves**, 128 radii each, **10⁹ paths per
curve** (1.54×10¹² paths total, ~20 node-hours).

Barrier parameters are drawn independently for every (cosmology, barrier) pair,
so 1536 distinct points fill the 3-D barrier box rather than 24 of them.

### 5.2 Two design decisions worth understanding

**Radii are chosen per curve, not fixed.** Across the barrier box, ν spans
*twelve orders of magnitude* over the accessible radius range. Any fixed σ or R
window would leave most curves either never crossing or crossing instantly.
Each curve's grid is solved so that ν sweeps [0.25, 4.5], intersected with
σ ∈ [0.15, 4] and R ∈ [0.5, 150] Mpc/h. The last clip matters: without it,
high-redshift cosmologies (where σ never reaches 4) ran to R ~ 10⁻³ Mpc/h,
where P(k) is pure extrapolation and γ² drifts far outside anything a real
analysis queries.

**Barrier draws are rejection-sampled for reachability.** A draw is kept only
if ν descends to ≤ 1.2 at physical radii. Without this, ~30% of curves crossed
under 3% of their paths — and because a non-crossing walk steps through *every*
radius, those are simultaneously the most expensive to run and the least
informative. The rejected corner is one where no voids form at any physical
scale (the likelihood is zero there) and where the emulator's behaviour is the
ν ≫ 1 asymptote anyway. Acceptance runs 27–100% (median 78%), and α reaches
2.4–2.5 at z ≈ 0 but only ~1.3 at z ≈ 2 — exactly the growth-factor scaling the
physics requires.

### 5.3 Barrier perturbations

25% of curves carry a smooth 5% ripple (low-order Chebyshev in ln σ) on top of
the SMT shape. These are *harder to predict* — they dominate the worst-error
decile — but **including them in training makes pure-SMT prediction better**
(0.137% vs 0.160%), acting as shape augmentation. The emulator's documented
contract remains the SMT family.

### 5.4 Exclusions

Curves whose walk origin is low enough that >1% of paths start above the
barrier are dropped (32 of 1536). Those paths never enter any radius bin, and
such curves were measured to be predicted badly — 0.77% typical, 11% at the
tail, against 0.13% in-domain. The same threshold becomes the deployed domain
guard.

---

## 6. Results

### 6.1 Accuracy

Cross-validated by holding out **whole cosmologies**, and scored against the
Monte Carlo's own multiplicity taken straight from the counts
(`counts/(N·Δr)`) — not reconstructed the same way as the prediction, which
would hide errors common to both.

| evaluated on | median | 95th | model error | ν<1 | 1–2 | 2–3 | >3 |
|---|---|---|---|---|---|---|---|
| **pure SMT (the contract)** | **0.128%** | **0.60%** | 2.06e-03 | 0.13% | 0.10% | 0.13% | 0.35% |
| all curves | 0.132% | 0.65% | 2.18e-03 | 0.14% | 0.10% | 0.13% | 0.35% |

"Model error" is the RMS log-hazard residual with the Monte Carlo's own noise
removed in quadrature. The ν>3 column is *not* model error — there the residual
equals the reference's noise (ratio 1.00–1.07), so the emulator is more
accurate than this training set can demonstrate.

### 6.2 Generalisation

Holding out contiguous regions of parameter space rather than scattered
cosmologies — the honest test, since the CDM feature manifold is thin:

| held out | median | ν>3 | 95th |
|---|---|---|---|
| random 4-fold (reference) | 0.14% | 0.36% | 0.80% |
| high Σm_ν (>0.35 eV) | 0.15% | 0.36% | 0.82% |
| high z (>1.5) | 0.18% | 0.36% | 0.84% |
| low Ω_c h² (<0.105) | 0.17% | 0.37% | 0.83% |
| high Ω_c h² (>0.155) | 0.15% | 0.34% | 0.84% |
| low n_s (<0.935) | 0.16% | 0.36% | 0.77% |
| low H₀ (<62) | 0.16% | 0.38% | 0.86% |

Removing whole chunks of the box costs essentially nothing.

### 6.3 Validation against a fresh Monte Carlo

24 curves at new cosmologies and new barriers (unrelated seed), 10⁸ paths:

- **pooled median 0.198%**, 95th 1.41% (0.16% at ν<2, 0.37% at ν>2)
- the Monte Carlo's own error on those bins is **0.20%** — the reference is a
  co-contributor, so the emulator's error is below what this test resolves
- 23 of 24 curves clean; the one outlier (α = 0.080) was **flagged by the
  domain guard** on all 127 of its bins, and still came in at 0.335%

### 6.4 Grid independence

The same physical problem sampled at 64, 128 and 256 radii:

| curve | f(64) | f(128) | f(256) | spread |
|---|---|---|---|---|
| v00_0 | 9.75187e-02 | 9.74485e-02 | 9.74156e-02 | 0.11% |
| v01_1 | 1.27616e-01 | 1.27435e-01 | 1.27348e-01 | 0.21% |
| v01_2 | 6.36121e-02 | 6.35121e-02 | 6.34669e-02 | 0.23% |

Worst spread **0.23%** across a 4× resolution change. The Monte Carlo itself
moves ~1% between 50 and 100 radii. Nothing in the fit enforced this — it
follows from the features being properties of the field rather than of the
grid, and it means a coarse grid costs nothing at inference.

### 6.5 Speed

| | |
|---|---|
| emulator (numpy, 128 radii) | 0.21 ms |
| Monte Carlo, 10⁹ paths | ~417 s |
| speed-up | 2.0×10⁶ |

The C implementation, with no numpy overhead on a 433-parameter network, should
be well under 0.1 ms.

### 6.6 A note on the asymptote

The hazard ratio does **not** converge to exactly 1 at high ν; it plateaus at
about **+0.2%**. Splitting those bins by their own error across a 5× range in
precision shows no trend, so this is real and not an estimator bias. Part of it
is probably the known ~7×10⁻⁴ quadrature floor in `deriv_variance` propagating
into the baseline rate. It does not matter, because the network learns it — as
long as inference computes the baseline the same way, the offset is absorbed
rather than propagated. **This is why no envelope forcing the correction to 1
at high ν was used**, despite that being the obvious architectural choice.

---

## 7. The domain guard

Two distinct failure modes, both reported through the optional out-parameter:

1. **Walk origin too low.** If ν at the largest radius falls below 2.33, a
   sizeable fraction of walks start above the barrier and never enter a bin.
   This is the caller's to fix — extend the radius grid outward. Verified to
   fire correctly: a grid truncated to 80 radii reports ν = 2.08, to 60 radii
   ν = 1.64.
2. **Features outside the trained box.** Ordinary extrapolation; the message
   names the responsible feature. A barrier scaled ×4 flags all bins and names
   `y` as furthest out.

Trained ranges (from `ep_emulator.npz`):

| feature | min | max |
|---|---|---|
| nu | 0.22932 | 4.82415 |
| gamma2 | 0.15555 | 0.25348 |
| y | −11.52258 | −0.07319 |
| dlnnu_dlnS | −2.17689 | −0.21515 |
| nu_back | 0.35578 | 4.86560 |
| lag | 0.00389 | 0.69315 |
| cum_lam | 0.00000 | 0.67068 |
| nu_origin | 2.32690 | 4.86560 |

**A per-axis box is known to be insufficient in general.** Trained on ΛCDM and
evaluated on power-law spectra, curves whose features sat *entirely inside* the
box were still 1.6–4.8% wrong, and bins matched to within 0.15 in standardised
feature distance were 3.5% wrong. Local features are sufficient *within* a
spectral family but not *across* families: the shape of the γ²(S) trajectory
carries information the pointwise features do not. Within the documented
contract — CDM-like spectra, SMT barriers — this does not bite, but the guard
should not be read as a general-purpose safety net.

---

## 8. Running the pipeline

```bash
./run_training_pipeline.sh spectra    # 1. P(k) tables from CAMB (local, ~4 min)
./run_training_pipeline.sh submit     # 2. prints the cluster instructions
./run_training_pipeline.sh qa         # 3. QA on the returned runs/
./run_training_pipeline.sh train      # 4. fit + export weights (~2 min)
./run_training_pipeline.sh validate   # 5. fresh-ensemble validation (~26 min)
./run_training_pipeline.sh studies    # 6. ablation and diagnosis
```

Stages skip themselves if their output exists; `FORCE=1` overrides.

### Files

**Core (the specification):**

| file | role |
|---|---|
| `ep_model.py` | **The canonical definition.** Baseline rate, hazard integration, the eight features, survival recursion, `EPEmulator` (inference + guard + persistence). The C implementation must reproduce this exactly. |
| `ep_emulator.py` | The weighted MLP fitter: forward pass, analytic gradients, L-BFGS. Training only. |

**Pipeline:**

| file | role |
|---|---|
| `ep_make_spectra.py` | CAMB → shared-grid P(k) table. Runs locally so the cluster needs only numpy and pysif, and so the spectra are pinned. |
| `ep_trainset_design.py` | The domain: parameter boxes, adaptive radii, reachability, barrier construction. |
| `ep_trainset_worker.py` | One SLURM array task = one cosmology, all its barriers. |
| `ep_trainset.slurm` | The array script (64 tasks, throttled to 16). |
| `ep_trainset_load.py` | Reads counts back and derives features/targets through `ep_model`. |
| `ep_qa.py` | Integrity, noise floor, coverage, asymptote, smoothness. |
| `ep_train_final.py` | Cross-validation, final fit, weight export. |
| `ep_validate.py` | Fresh Monte Carlo, grid independence, guard, speed. |

**Studies (evidence, not needed to train):**

`ep_ratio_pilot.py` (spectral-family pilot), `ep_cdm_coverage.py`,
`ep_cdm_pilot.py`, `ep_cdm_tests.py` (CDM scoping), `ep_feature_ablation.py`
(pilot ablation), `ep_ablation_full.py` (full-statistics ablation),
`ep_lownu_diagnosis.py` (where the residual error lives).

**Data:** `ep_spectra.npz` (64 P(k) tables, 834 KB), `ep_emulator.npz` (trained
weights + domain box), `ep_validation_mc.npz` (fresh validation ensemble).

### Reproducibility

The design is deterministic in its seed **plus the spectrum table** — barrier
acceptance depends on σ, so keep `ep_spectra.npz` with the outputs. Monte Carlo
seeds are a pure function of (cosmology index, barrier index), so any single
task reruns bit-identically, and the counts do not depend on thread count.

---

## 9. Using it from the library

The emulator is one public function, declared in `include/sif/model/excursion_set.h`:

```c
SIF_NODISCARD sif_real* sif_ep_multiplicity_function_emu(const sif_real* radii,
  uint32_t n_radii, const sif_real* sigma, const sif_real* barrier,
  const double* deriv_variance, sif_emu_domain_t* domain, sif_option opt);
```

It takes **`sigma`, not the packed covariance** that `sif_ep_multiplicity_function`
needs. That is deliberate: the emulator reads only the diagonal, so it wants
`n` numbers where the Monte Carlo wants `n(n+1)/2`, and its cost is linear
rather than quadratic in the radius count. `deriv_variance` is **required**
here, unlike the Monte Carlo path — differencing it off a covariance converges
only at first order, which would reintroduce exactly the grid dependence the
emulator exists to avoid. Both come from `sif_delta_covariance_pk`.

`domain` is optional (pass NULL to skip) and reports where the call sat
relative to the training:

```c
typedef struct {
  int      in_domain;        /* zero if either check below fired */
  uint32_t n_bins_outside;   /* bins with a feature outside the trained box */
  sif_real   nu_origin;        /* B/sigma at the largest radius */
  sif_real   first_step_mass;  /* walks starting above the barrier */
  sif_real   expected_error;   /* 0.0013 in domain, 0.0077 just outside */
} sif_emu_domain_t;
```

Leaving the trained region is **not** an error and does not fail the call: the
answer is returned, the log names the responsible quantity, and `in_domain` is
cleared. A sampler should be able to notice a degraded proposal without parsing
log output or aborting a chain. NULL is reserved for genuinely invalid input
(null pointers, fewer than three radii, non-positive sigma, non-monotonic
radii).

From Python:

```python
import pysif.model as model

cov, sigma, high_k, dvar = model.delta_covariance_pk(k, pk, radii)
barrier = model.barrier_smt(sigma, alpha=0.7, beta=0.3, gamma=0.8)

f = model.multiplicity_function_ep_emu(radii, sigma, barrier, dvar)
f, dom = model.multiplicity_function_ep_emu(radii, sigma, barrier, dvar,
                                            return_domain=True)
# dom -> {'in_domain': True, 'n_bins_outside': 0, 'nu_origin': 4.4988,
#         'first_step_mass': 3e-06, 'expected_error': 0.001282}
```

Measured at **48 microseconds** per call at 128 radii through the Python
binding, against roughly 400 s for the 10^9-path walk.

### Implementation

| file | role |
|---|---|
| `src/math/nn.c`, `nn.h` | Internal dense-network evaluation. Not public: there is no training in C, and the weights only ever come from this library's own models. Batched, no allocation, `const` throughout, so it is safe to call from several threads. |
| `src/model/ep_emu_weights.h` | Generated by `ep_export_c.py`. `static const` tables in `.rodata`; every literal verified to round-trip bit-exactly before the header is written. |
| `src/model/ep_emu.c` | The entry point: features, network, survival recursion, domain report. |
| `src/model/ep_upcrossing.c` | The baseline, shared with the Monte Carlo path rather than duplicated — `sif_ep_features_fill_diag`, `sif_ep_hazard_bins`, `sif_ep_survival`. |
| `tests/test_ep_emu.c` | Pins the C against stored values from the Python reference. |

### Why the tests look the way they do

A trained model has no closed form to check against, so the correctness
argument is that the C reproduces the Python it was fitted with — exactly. The
stored cases in `tests/ep_emu_cases.h` (regenerate with
`ep_export_test_cases.py`) pin the whole chain: baseline rate, hazard
integration, all eight features, the network, and the survival recursion. The
measured agreement is **5×10⁻⁸**, which is float rounding on the `sif_real`
output; anything at 10⁻⁵ or worse means a feature is being computed
differently, and no physical invariant would notice that.

The remaining tests assert what must hold whatever the network predicts: the
result is a non-negative density integrating to at most one, it does not depend
on the radius sampling (0.3% across a 4× refinement), and the domain report
fires on both conditions it exists for.

## 10. Known limitations

1. **Contract is CDM-like spectra and SMT barriers.** Cross-family
   generalisation is measurably worse (§7) and is not claimed.
2. **The radius grid must extend far enough out** that ν at the largest radius
   exceeds ~2.33. The guard reports this; it cannot fix it.
3. **Accuracy degrades at the edges of feature coverage** — the outer 5% of
   each feature's range carries 2–3× the typical error.
4. **The emulator reproduces the continuum limit**, which differs from a Monte
   Carlo run on a coarse grid by ~1% at 50 radii. Comparing the two on a coarse
   grid will show a disagreement larger than either method's error, and the
   emulator is the one to trust.
