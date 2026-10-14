# The excursion-set multiplicity emulator

A trained replacement for the Monte Carlo first-crossing calculation
(`sif_ep_multiplicity_function`), accurate to **0.08%** (median, held-out
cosmologies, radii below 80 Mpc/h) for walks starting anywhere from ~2 to
250 Mpc/h, and about **3×10⁶ times faster**. Intended for MCMC use where the
Monte Carlo is unaffordable.

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
roughly 400 s on a workstation (60 s on a 72-core node), and its statistical
error falls only as 1/√N_paths. An MCMC needing 10⁵–10⁶ likelihood evaluations
cannot use it.

**What is emulated is not the multiplicity function itself.** The multiplicity
spans orders of magnitude and carries a factor exp(−ν²/2); a network asked to
reproduce it directly would spend all its capacity on dynamic range that is
already known analytically. Instead we emulate the *ratio* between the Monte
Carlo and a semi-analytic baseline, which is a smooth O(1) quantity.

---

## 2. What is actually emulated

### 2.1 The baseline: the up-crossing rate of Verza et al. (2024), eq. (3.15)

The baseline is the up-crossing rate — the rate at which the walk crosses the
barrier upward faster than the barrier itself moves. In the walk's own time
variable S = σ²(R) (which *increases* as R falls):

```
f_up(S) = exp(−ν²/2) / √(2π S) · Σ_slope · E[(z − y)⁺]
```

where `E[(z − y)⁺] = φ(y) − y(1 − Φ(y))` is the mean excess of a standard
normal. This is exact in the high-barrier limit, where a first crossing and any
crossing are the same event, and runs a few per cent low for a flat barrier and
up to ~15% for a steep moving one.

**This is not the textbook Musso–Sheth rate, and the difference matters.** It
is the specialisation given by Verza et al. (2024) as their eq. (3.15), written
here in a form that costs one transcendental instead of two. Their bracket

```
√(Γ_δδ / 2πS)·exp[−(S/2Γ_δδ)(B/2S − B′)²]
  + ½(B/2S − B′)·{erf[√(S/2Γ_δδ)(B/2S − B′)] + 1}
```

becomes `Σ_slope·[φ(y) − y(1 − Φ(y))]` under `Σ_slope² = Γ_δδ/S` and
`y = (B′ − μ)/Σ_slope` with `μ = B/2S` — the same expression, term for term.

What separates it from Musso & Sheth is that `Γ_δδ = S·⟨(dδ/dS)²⟩ − 1/4` keeps
its scale dependence instead of being frozen at MS's `Γ = 1/3`. `sif` supplies
that term exactly: `sif_delta_covariance_pk` differentiates the window under
the k-integral (their appendix A), and the emulator entry point *refuses to
run* without it rather than falling back on a finite difference. Against the
Monte Carlo, the paper measures the frozen-Γ form at 5–9% over its mass range
and eq. (3.15) at 1.5–7%. It is also what lets a single network cover every
spectrum: the shape of P(k) enters the baseline through `⟨(dδ/dS)²⟩`, so the
correction left over is a smooth O(1) function of scale-free features.

**The barrier slope is differenced in log space.** B′ = dB/dS is the one
derivative not supplied analytically. It is taken as (B/S)·d ln B/d ln S, which
is near-constant for an SMT barrier and so differences to second order on any
grid; a direct dB/dS is curved in S and left a grid-dependent error in f_up.
A barrier that is not positive everywhere falls back to the direct difference.

### 2.2 Why not the authors' better formula, eq. (3.14)

Verza et al. also give eq. (3.14) — the integral over a second scale s < S,
which tracks the Monte Carlo about 50% closer than eq. (3.15). It is not used
as the baseline here, for three reasons:

1. **It would undo the emulator's interface.** `P(s)` needs `C(S, s)` and
   `C′ = ∂C/∂S` for every pair of scales, plus an `∫ds` per crossing scale
   with a non-elementary inner integral `F(x, α, β)`. That reintroduces the
   O(n²) covariance and the nested quadrature the emulator exists to avoid;
   the current path needs only the diagonal and is linear in the radius count.
2. **The result would not get more accurate.** The network learns
   `ln(Λ_MC/Λ_base)`, so the product targets the Monte Carlo whichever
   baseline sits underneath. A better pedestal shrinks the correction range
   from [0.89, 1.70] towards 1; it does not move the accuracy, which §6 shows
   is set by the Monte Carlo noise and the fit, not by the baseline.
3. **The coherence argument runs the other way.** Ground truth here is the
   Monte Carlo — eqs. (3.4)–(3.5), the exact numerical solution of the same
   framework, implemented in `src/model/excursion_set.c`. Eq. (3.14) is itself
   an approximation to that. Putting it under the network would mean
   scaffolding an emulator of the exact answer on an approximation whose own
   error the network then has to absorb.

Eq. (3.14) is still worth having as an **independent analytic cross-check** in
validation — a third curve beside the Monte Carlo and the emulator at moderate
S, where the paper quotes ~1% — since it does not depend on the Monte Carlo
being right. That is an offline diagnostic, not a runtime path.

### 2.3 The target: a hazard ratio, not a multiplicity ratio

The correction is applied to the **hazard**, not to the binned multiplicity.
Per bin *i* the baseline integrated hazard is

```
Λ_up[i] = ∫ f_up dS   over the bin,   with ln f_up replaced by the cubic
                                      through the four radii nearest the bin,
                                      integrated at four Gauss-Legendre nodes
```

and the target the network learns is

```
correction = ln( Λ_MC[i] / Λ_up[i] )
```

with Λ_MC extracted from the raw counts through the survival recursion. This
choice matters for three reasons:

1. **Boundedness is structural.** The survival recursion turns any set of
   non-negative hazards into a non-negative density that integrates to at most
   one, however wrong the network is. There is no clamp.
2. **The bins decouple.** A multiplicity ratio would force the network to learn
   the survival bookkeeping the baseline already does exactly.
3. **Errors enter multiplicatively**, i.e. as relative errors — which is what a
   likelihood cares about.

**The quadrature has to be converged, or the target is not a property of the
walk.** The first version integrated the log-linear interpolant (the
logarithmic mean of the endpoints), exact only for a pure exponential. Its
error depended on the bin width — 0.1–0.5% at ν > 3 on a 129-radius grid, 2%
on 65 — and the target absorbed it, so the network learned an average over the
training grids that no input could tell it about. The cubic-in-ln f rule leaves
0.01–0.05% at 129 radii, converging as the square of the step. Fewer than four
radii fall back to the logarithmic mean (only the semi-analytic up-crossing
path accepts that few).

Measured over the whole training set the ratio lies in **[0.889, 1.703]**: a
smooth, O(1), well-conditioned regression target.

### 2.4 The output: the density at the radii

The network works per bin, but the caller asked for f at radii. The default
output is the density **at the n radii given**:

```
f(Rᵢ) = f_up(Sᵢ) · exp(correction at Sᵢ) · alive(Sᵢ) · |dS/dR|ᵢ
```

with the correction interpolated linearly in ln S between bin centres (held
flat beyond the outermost ones) and the survival `alive` interpolated from the
bin edges the recursion produces. `dS/dR` is again taken in log space,
(S/R)·d ln S/d ln R, which is smooth where dS/dR itself is not.

The old form — n − 1 bin averages, `alive · (1 − exp(−Λ)) / ΔR` on the bin
centres, exactly what a histogram of the Monte Carlo measures — stays available
as `SIF_EP_EMU_BIN_CENTRES` (`binned=True` in Python). The two agree: summed
over a grid, the trapezoid of the point values and the binned total differ by
~10⁻⁴, and the point output depends no more on the sampling than the binned
one (§6.4). All accuracies in §6 are quoted on the binned form, since that is
what the Monte Carlo counts measure directly.

---

## 3. The features, and what they mean physically

Nine numbers per radius bin. All are dimensionless or scale-free, which is why
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
| 7 | `cum_lam` | Σ_{j>i} Λ_up[j] + ½Λ_up[i] | Baseline hazard accumulated up to the bin's *centre*, i.e. how much crossing probability has already been spent. **Summarises the entire prior history in one scalar**, and is what the surviving population is conditioned on. |
| 8 | `nu_origin` | ν at the largest radius | Where the walk started. First crossing genuinely depends on the starting scale for walks starting low, so this dependence is made explicit rather than assumed away. |
| 9 | `gamma2_back` | γ² at S/2 | γ² at the same lookback as `nu_back`. γ²(R) peaks near R ~ 10 Mpc/h, so a local value is reached from either side; this says which side the walk came from. |

### Why these, and not others

Features 1–3 are the local description the baseline itself uses. Features 4–9
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
- **`gamma2_back` was rejected on campaign 1 and adopted on campaign 2.** On
  campaign 1 a γ² history bought ~2%: every walk started below 51 Mpc/h, so
  γ² < 0.2 was only ever met on the small-R side of its peak and the local
  value was unambiguous. Walks starting at hundreds of Mpc/h meet the same
  values from the other side, descending through the peak. A pilot on the
  first large-R curves measured it: with only more data, a void barrier at
  δ_NL = −0.15/−0.20 kept a residual of 3–5× the Monte Carlo noise at ν < 1;
  with `gamma2_back` it fell to the noise. The slope d ln γ²/d ln S, tried in
  the same pilot, helped less.

Notably, the survival probability `exp(−cum_lam)` on its own made things
*worse*; it saturates near 1 and loses resolution. The log-scale cumulative
hazard is the informative parametrisation.

### Every feature must be a property of the walk, not of the grid

The emulator is called on whatever radii the caller chose, so a feature that
moves with the sampling is a different input on every grid. Four early
definitions did, and the large-radius campaigns exposed them by giving the
walk's history a reason to matter:

- **`cum_lam` stopped at the bin's upper edge**, leaving out half the bin's own
  hazard — eight times more of it on a 65-radius grid than on 513. The network
  is rightly sensitive to depletion (∂correction/∂cum_lam ≈ −1), so the answer
  moved by 0.5% between those grids while the Monte Carlo moved by 0.1%. It is
  now taken at the bin centre, where every other feature is.
- **`nu_origin` was ν at the outermost bin centre**, half a step inside the
  origin: 4.31 on a 65-radius grid against 4.48 on 513 for the same walk. It is
  now ν at the largest radius.
- **The hazard quadrature** (§2.3) and **the barrier slope** (§2.1) carried
  first-order grid errors into f_up and so into every feature built on it; and
  the point output's dS/dR (§2.4) did the same to the result. All three are now
  second order, the derivatives by working in log space.

With these, the same problem on 65, 129 and 257 radii agrees to ≤ 0.02% (§6.4),
so there is no longer a spacing the caller has to respect.

### What was ruled out

On campaign 1 (433-parameter network):

| hypothesis | test | result |
|---|---|---|
| model too small | 385 → 2689 parameters | 13% gain, high-ν *worse*. Not capacity. |
| optimisation variance | ensemble of 1, 3, 5 networks | 0.137% → 0.139% → 0.137%. Different seeds converge to the same function. |
| systematic misspecification | mean vs RMS residual per ν band | bias/RMS = 0.01–0.06. Unbiased; pure scatter. |
| training-data noise | model residual vs MC noise per band | ratio 1.00–1.07 at ν>3. Data-limited there; the model error is unmeasurably small. |

**Those comparisons were run with an optimiser that stopped early** (§4): every
network ended near iteration 660 whatever `maxiter` said. Rerun to completion
on the final training set the conclusions still hold, for a sharper reason.
Every architecture from 32×32 to 64×64 lands within the validation noise of the
others (§4), and **averaging the three gains nothing**: the residual left at
ν < 1.5 is a per-curve offset of ~2×10⁻³ in log-hazard, coherent along each
curve and shared by networks of every size. It is not fit noise; most likely it is
what the nine features cannot tell apart — walks they describe alike whose
Monte Carlo differs by that much. Only a new feature, not capacity or
averaging, moves it.

---

## 4. Network architecture

```
9 inputs → standardise → 48 (tanh) → 48 (tanh) → 1 linear  = 2881 parameters
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
- **Optimiser**: L-BFGS-B with analytic gradients, **6000 iterations with
  `ftol = gtol = 0`**, L2 = 10⁻⁶, over all 517,984 usable bins.
- **Parallelism**: the loss is evaluated in blocks of 8192 rows on a thread
  pool, BLAS single-threaded underneath (a threaded BLAS under the pool
  oversubscribes and was measured to stall). 13× faster than one big array.

**The stopping rule is the largest single improvement in this version.**
SciPy's default relative tolerance (~2×10⁻⁹) ends the fit after a few hundred
iterations here, because the loss is ~10⁻⁵ and each step's gain looks
negligible relative to it. Every network stopped near iteration 660 at twice
the training loss a full run reaches.

Widths, scored on data no fit saw (median relative error, bins below
80 Mpc/h). On campaigns 1–2, against the validation sets of §6.2 and the 16
unseen pilot cosmologies:

| network | parameters | generic (V) | thesis barrier (T) | pilot, origin < 100 / 100–300 | sensitivity to grid start |
|---|---|---|---|---|---|
| 32×32, stopped at ~660 it. | 1409 | 0.120% | 0.144% | 0.154% / 0.236% | 0.30% |
| 16×16 | 449 | 0.078% | 0.095% | 0.085% / 0.164% | 0.13% |
| 32×32 | 1409 | 0.077% | 0.092% | 0.083% / 0.157% | 0.11% |
| 48×48 | 2881 | 0.074% | 0.090% | 0.073% / 0.155% | 0.09% |
| 64×64 | 4865 | 0.074% | 0.094% | 0.072% / 0.150% | 0.09% |
| 48×48×48 | 5233 | 0.079% | 0.087% | 0.076% / 0.165% | 0.07% |
| 64×64×64 | 9025 | 0.077% | 0.102% | 0.083% / 0.166% | 0.11% |

Retrained on the final set (campaigns 1–4), against V, T and the far-origin
set W:

| network | parameters | V | T | W |
|---|---|---|---|---|
| 32×32 | 1409 | 0.076% | 0.121% | 0.108% |
| **48×48** | **2881** | **0.080%** | **0.096%** | **0.112%** |
| 64×64 | 4865 | 0.080% | 0.103% | 0.106% |

All fully trained networks sit within the validation noise of each other (the
generic set's Monte Carlo noise alone is 0.057%). 48×48 is kept: the best on
the thesis barrier, never the worst, and free in C; three layers buy nothing.

---

## 5. The training set

Four campaigns over the same 64 cosmologies (same seed), so cross-validation
can hold a cosmology out of all of them at once.

| | 1 (`runs/`) | 2 (`runs_v2/`) | 3 (`runs_v3/`) | 4 (`runs_v4/`) |
|---|---|---|---|---|
| walks placed by | a ν window | their origin | their origin | their origin |
| walk origin (Mpc/h) | 1.5 – 51 (median 8) | 2 – 700, log-uniform | 120 – 250, log-uniform | 30 – 120, log-uniform |
| kept only if | ν ≤ 1.2 reached | ν ≤ 1.2 reached | … and R < 60 reached | ν ≤ 1.2 reached |
| ν at the origin | 4.5 (fixed) | 3.5 – 5.5 | 3.5 – 5.5 | 3.5 – 5.5 |
| β | 0.01 – 1.5, uniform | 0.01 – 1.5, uniform | 10⁻³ – 1.5, log-uniform | 10⁻³ – 1.5, log-uniform |
| barriers per cosmology | 24 | 24 | 12 | 8 |
| curves used | 1504 | 1306 (of 1521) | 768 | 512 |
| k grid | N_K = 2400 | N_K = 10000 | N_K = 10000 | N_K = 10000 |

**4090 curves**, 128 radii and **10⁹ paths each**: 4.09×10¹² paths, 517,984
usable bins. Campaign 2 alone took ~26 node-hours on 72-core nodes.

**Training stops at walks starting at 300 Mpc/h** (`MAX_ORIGIN` in
`ep_train_final.py`); campaign 2's 215 curves beyond it are kept on disk but
not fitted. The contract is walks starting at 2–250 Mpc/h, with margin on both
sides of the trained range. Its lower end is physical rather than chosen: a walk
starting at 1 Mpc/h reaches R = 0.5 before ν can fall to crossing values, i.e.
it would describe voids smaller than 0.5 Mpc/h.

### 5.1 Domain

| axis | range | notes |
|---|---|---|
| Ω_c h² | 0.08 – 0.18 | wider than a realistic prior, so chains sit in the interior |
| Ω_b h² | 0.018 – 0.028 | |
| H₀ | 55 – 80 | |
| n_s | 0.90 – 1.02 | |
| Σm_ν | 0 – 0.5 eV | massive neutrinos included |
| z | 0 – 2 | |
| α (barrier) | campaign 1: 0.05 – 2.50; campaigns 2–4: solved | B(σ) = α[1 + (β/σ)^γ] |
| β (barrier) | 10⁻³ – 1.50 | floored strictly positive: β = 0 makes (β/σ)^γ undefined. At 10⁻³ the barrier is near-constant |
| γ (barrier) | 0.50 – 3.50 | |

Barrier parameters are drawn independently for every (cosmology, barrier)
pair, so the curves fill the barrier box rather than repeating a short list.

### 5.2 Campaign 1: walks placed by a ν window

**Radii are chosen per curve, not fixed.** Across the barrier box, ν spans
*twelve orders of magnitude* over the accessible radius range. Any fixed σ or R
window would leave most curves either never crossing or crossing instantly.
Each curve's grid is solved so that ν sweeps [0.25, 4.5], intersected with
σ ∈ [0.15, 4] and R ∈ [0.5, 150] Mpc/h.

**Barrier draws are rejection-sampled for reachability.** A draw is kept only
if ν descends to ≤ 1.2 at physical radii. Without this, ~30% of curves crossed
under 3% of their paths — and because a non-crossing walk steps through *every*
radius, those are simultaneously the most expensive to run and the least
informative. Acceptance runs 27–100% (median 78%).

**What it did not cover.** Measured afterwards, none of its curves started
beyond R = 51 Mpc/h: the σ ≥ 0.15 clip ends every walk near R ≈ 52 at z = 0,
and reaching larger radii needed a small α *and* a small β at once, a corner
of the box nobody drew. The R ≤ 150 cap never bound. A void barrier at
δ_NL = −0.2 and z = 0 needs walks starting near 150 Mpc/h, and there the
campaign-1 emulator was 0.5–1% wrong at ν < 2 (§6.6). Worse, the guard passed
most of those calls: γ² = 0.18 was in the trained box, met only at R ~ 1 Mpc/h
on the other side of the peak.

### 5.3 Campaigns 2–4: walks placed by where they start

Each curve draws its **origin** — a radius log-uniform over the campaign's range
and a ν there uniform over 3.5–5.5 — then (β, γ) from the box, and solves α so
the smooth barrier sits at that ν at that radius. The emulator never sees α,
only B/σ and its shape, so nothing is lost, and the walk's history is covered
evenly in the quantities the network does see. The inner edge is where ν
reaches 0.25, or R = 0.5, or σ = 4; there is no σ floor and no radius cap other
than the origin. Reachability is campaign 1's rule (ν ≤ 1.2 somewhere the walk
goes), checked over the radii the walk actually visits.

**Campaign 2** spans 2–700 Mpc/h. **Campaign 3** fills the corner campaign 2
left almost empty: a walk starting beyond 150 Mpc/h only comes down below
80 Mpc/h if its barrier is nearly flat in S (ν must fall by ~5× while R falls by
less than one e-fold), so such bins were 0.4% of campaign 2 and
cross-validated at 0.45%. Campaign 3 draws origins at 120–250 Mpc/h and keeps a
barrier only if its walk reaches R < 60. It is what a shallow void threshold at
low redshift produces. **Campaign 4** evens out the training density at
30–120 Mpc/h, which campaigns 1–3 left at a third of the density on either
side.

**β is log-uniform down to 10⁻³ in campaigns 3–4**, rather than uniform from
0.01. That puts more draws on the flat barriers whose walks span these ranges,
and reaches the constant barrier — the Sheth–van de Weygaert δ_v, a standard
choice in its own right — which sat at the edge of the earlier box.

**The origin ν is random because the answer must not depend on it.** Walks
starting above ν ≈ 3.5 have essentially never crossed (first-step mass
< 2.3×10⁻⁴), so the Monte Carlo is origin-independent to that level. With every
campaign-1 curve starting at ν = 4.5, the network could not separate the
lookback clamp at the origin from the physics, and the emulated answer moved
0.2–1.7% between grids started at different ν. It now moves 0.01–0.05% (§6.4).

### 5.4 Barrier perturbations

25% of curves carry a smooth 5% ripple (low-order Chebyshev in ln σ) on top of
the SMT shape. These are *harder to predict* — they dominate the worst-error
decile — but **including them in training makes pure-SMT prediction better**
(0.137% vs 0.160% on campaign 1), acting as shape augmentation. The emulator's
documented contract remains the SMT family. In campaigns 2–4, α is solved on
the smooth shape, so a rippled curve lands within 5% of its drawn origin ν.

### 5.5 Exclusions

Curves whose walk origin is low enough that >1% of paths start above the
barrier are dropped (32 of campaign 1's 1536; none of the later campaigns,
whose origin ν is ≥ 3.5 by construction). Those paths never enter any radius
bin, and such curves are predicted badly — 1.7% median for the final model —
which is the figure the guard quotes when it fires (§7). Campaign 2 drew 1521
of 1536 barriers: 15 origins found no reachable barrier in 200 draws.

### 5.6 The k grid is part of the training data

The baseline needs `⟨(dδ/dS)²⟩`, whose top-hat integrand is `P(k) sin²(kR)`
with no window suppression at all, and it fails in two ways that σ never
notices: a table stopping too low in k, and one too coarse to resolve the
oscillation, which needs `k·Δln k·R < π/4` and therefore tightens with radius.
`sif_delta_covariance_pk` warns about both above a tenth of the integral.

Campaign 1's grid (`1e-4, 500, 2400`, Δln k = 0.0064) was adequate to
4.7×10⁻⁴ at R = 150 Mpc/h. At R = 600 it is 0.43% off — above the emulator's
own error — so the later campaigns use **N_K = 10000** (Δln k = 0.0015),
measured against a Δln k = 0.0005 reference running to k = 2000: ≤ 1.5×10⁻⁴ out
to R = 800, σ unchanged. **Lowering `N_K` or `K_MAX` would corrupt the training
targets silently**, since the Monte Carlo and the baseline both read the same
`deriv_variance` and would move together, leaving the learned ratio looking
perfectly healthy. The same applies to a caller's own table: an analysis
walking from R ~ 250 Mpc/h needs a P(k) sampled about this finely.

---

## 6. Results

Errors are relative errors on the multiplicity against the Monte Carlo's own
counts (`counts/(N·ΔR)`, binned form) — not reconstructed the same way as the
prediction, which would hide errors common to both. "Model error" is the RMS
log-hazard residual with the Monte Carlo's own noise removed in quadrature.

### 6.1 Cross-validation

4-fold, holding out **whole cosmologies across all four campaigns**:

| evaluated on | bins | median | 95th | model error | ν<1 | 1–2 | 2–3 | >3 |
|---|---|---|---|---|---|---|---|---|
| all bins | 517,984 | 0.085% | 0.49% | 1.4×10⁻³ | 0.08% | 0.07% | 0.08% | 0.30% |
| campaign 1 | 190,989 | 0.078% | 0.49% | 9.1×10⁻⁴ | 0.08% | 0.06% | 0.08% | 0.32% |
| campaigns 2–4 | 326,995 | 0.090% | 0.48% | 1.6×10⁻³ | 0.09% | 0.08% | 0.08% | 0.29% |
| **R < 80 Mpc/h** | 445,225 | **0.081%** | **0.44%** | 1.3×10⁻³ | 0.08% | 0.06% | 0.08% | 0.31% |
| R < 80, origin < 30 | 273,066 | 0.079% | 0.48% | 1.0×10⁻³ | 0.08% | 0.06% | 0.08% | 0.32% |
| R < 80, origin 30–120 | 112,404 | 0.074% | 0.30% | 1.1×10⁻³ | 0.07% | 0.07% | 0.07% | 0.27% |
| R < 80, origin 120–300 | 59,755 | 0.110% | 0.49% | 2.5×10⁻³ | 0.11% | 0.10% | 0.06% | 0.07% |
| R ≥ 80 | 72,759 | 0.120% | 0.74% | 1.7×10⁻³ | 0.12% | 0.10% | 0.09% | 0.30% |

The far-origin corner went from 0.45% (1,361 bins, campaigns 1–2) to 0.110%
(59,755 bins) with campaign 3, and nothing else moved by more than the fold
noise. The ν>3 column is *not* model error — there the residual equals the
reference's noise, so the emulator is more accurate than this training set can
demonstrate.

### 6.2 Independent sets

None of these touched training or model selection. All at 10⁹ paths.

**Fresh generic ensemble** (`ep_validate_v2.py`, set V): 8 new cosmologies ×
6 origin-placed barriers, unrelated seeds.

| bins | median | 95th | MC noise | model error |
|---|---|---|---|---|
| all | 0.084% | 0.37% | 0.057% | 2.0×10⁻³ |
| R < 80 | **0.080%** | 0.34% | 0.057% | 1.9×10⁻³ |
| R < 80, origin 2–30 | 0.080% | 0.38% | 0.062% | |
| R < 80, origin 30–150 | 0.081% | 0.30% | 0.049% | |

Per cosmology, the median below 80 Mpc/h runs 0.062–0.102%, with no trend in
Ω_c h², H₀, n_s, Σm_ν or z. No curve is flagged by the guard.

**A calibrated void barrier at analysis thresholds** (set T): the barrier of a
real void-size-function analysis (B = α[1 + (β/σ)^γ], γ = 0.53, fitted α(δ) and
β(δ)), at δ_NL from −0.15 to −0.70 at z = 0, on the grid that analysis uses:
200 radii, ν = 4.8 at the origin. Bins inside the analysis range:

| δ_NL | −0.15 | −0.20 | −0.25 | −0.30 | −0.40 | −0.50 | −0.60 | −0.70 |
|---|---|---|---|---|---|---|---|---|
| walk origin (Mpc/h) | 185 | 146 | 119 | 100 | 71 | 51 | 37 | 25 |
| median error | 0.16% | 0.21% | 0.18% | 0.11% | 0.09% | 0.09% | 0.08% | 0.07% |
| MC noise | 0.08% | 0.08% | 0.09% | 0.09% | 0.09% | 0.09% | 0.09% | 0.09% |

Pooled below 80 Mpc/h: **0.096%** against 0.078% Monte Carlo noise. The
shallow thresholds sit at 0.16–0.21%, about twice their noise: these curves
share one barrier family and one cosmology, so the per-curve offset of §3 lands
on all of them with the same sign rather than averaging away. Every width
tried, and their average, gives the same values there.

**The far-origin corner** (set W): 8 new cosmologies × 4 barriers with walks
starting at 150–200 Mpc/h — three SMT draws and one near-constant (β = 10⁻³).
Below 80 Mpc/h: **0.112%** (SMT 0.126%, near-constant 0.082%) against 0.043%
noise; no curve flagged.

### 6.3 The Monte Carlo is converged

The network learns the ~128-radius Monte Carlo, so that has to be converged to
better than the accuracy asked of the emulator. Four problems at 65, 129, 257
and 513 radii over the same span, on nested grids so the finer runs' counts sum
exactly into the coarse bins:

| radii | ν < 1.5 | 1.5 – 3 |
|---|---|---|
| 65 | −0.05% to −0.11% | −0.02% to −0.16% |
| 129 | −0.01% to −0.04% | 0.00% to −0.05% |
| 257 | 0.00% to −0.02% | ±0.03% |

relative to 513, ±0.02% (ν < 1.5) and ±0.04% (1.5–3) statistical. Training
grids use d ln S ≈ 0.018 per step (median), between the 129- and 257-radius
rows. Coarse grids under-count crossings that happen between radii, but at
these spacings by a few hundredths of a per cent.

### 6.4 Grid and origin independence

Two properties the physics fixes and the fit does not enforce, over ν 1–3.5:

- **grid**: the density at the radii on nested 65-, 129- and 257-radius grids
  (ν 0.5–4.5), compared where the radii coincide — no interpolation;
- **origin**: the same spacing, the grid started at ν = 4.0, 4.8 and 5.2
  against 4.5.

| case | origin 4.0 | 4.8 | 5.2 | grid |
|---|---|---|---|---|
| void barrier δ_NL −0.15 | 0.03% | 0.01% | 0.02% | 0.01% |
| void barrier δ_NL −0.20 | 0.03% | 0.01% | 0.02% | 0.02% |
| void barrier δ_NL −0.45 | 0.04% | 0.01% | 0.02% | 0.01% |
| void barrier δ_NL −0.70 | 0.05% | 0.02% | 0.03% | 0.02% |
| SMT, origin 30 Mpc/h | 0.03% | 0.01% | 0.02% | 0.02% |
| SMT, origin 150 | 0.03% | 0.02% | 0.04% | 0.02% |
| SMT, origin 400 | 0.03% | 0.02% | 0.04% | 0.02% |

A 65-radius grid over this span is d ln S ≈ 0.05–0.06 per step, three times
coarser than training, and still within 0.02% of 257. The previous emulator
moved by 0.2–1.7% with the grid start and by up to 0.23% across a 4×
refinement.

### 6.5 Speed

| | |
|---|---|
| C emulator through pysif, 128 radii | 122 µs |
| C emulator through pysif, 200 radii | 189 µs |
| Monte Carlo, 10⁹ paths, 128 radii | ~400 s (workstation), ~60 s (72 cores) |
| speed-up | ~3×10⁶ |

2.5× the previous version's 48 µs, from the larger network and the higher-order
quadrature. An MCMC of 10⁶ evaluations spends two minutes in the emulator.

### 6.6 Against the previous emulator

The emulator this replaces (8 features, 16×16, campaign 1 only, the old feature
definitions), scored on the same fresh Monte Carlo as §6.2, bins below
80 Mpc/h:

| set | previous median | this median | previous 95th | this 95th | MC noise |
|---|---|---|---|---|---|
| V generic, all | 0.24% | **0.080%** | 1.59% | 0.34% | 0.057% |
| V origin 2–30 | 0.21% | 0.080% | 0.89% | 0.38% | 0.062% |
| V origin 30–120 | 0.32% | 0.078% | 2.44% | 0.29% | 0.050% |
| V origin 120–250 | 0.62% | 0.14% | 2.60% | 0.31% | 0.046% |
| V, per cosmology | 0.20 – 0.35% | 0.06 – 0.10% | | | |
| T thesis barrier, pooled | 0.20% | **0.096%** | 1.13% | 0.85% | 0.078% |
| T δ_NL −0.20 | 0.55% | 0.23% | 0.66% | 0.35% | 0.061% |
| T δ_NL −0.30 | 0.16% | 0.08% | 0.38% | 0.22% | 0.073% |
| W far origin, all | 0.66% | **0.112%** | 1.72% | 0.45% | 0.043% |
| pilot, 16 cosmologies, origin 30–120 | 0.35% | 0.14% | 1.81% | 0.52% | 0.155% |
| pilot, origin 120–250 | 0.73% | 0.19% | 1.48% | 0.51% | 0.152% |

Three to six times better wherever the old one was in its domain, and the
95th percentile, which the old features' grid dependence dominated, by 3–8×.
On its own campaign-1 domain the old emulator cross-validated at 0.128% (model
error 2.1×10⁻³); the same bins are now at 0.078% (9.1×10⁻⁴).

### 6.7 A note on the asymptote

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

The guard should fire on calls that are actually wrong and on nothing else.
Two checks, both reported through the optional out-parameter:

1. **Walk origin too low.** If ν at the largest radius falls below 2.35, a
   sizeable fraction of walks start above the barrier and never enter a bin.
   This is the caller's to fix — extend the radius grid outward. The error
   quoted when it fires is the measured one on such curves, 1.7%. Measured
   against the Monte Carlo, origins at ν = 2.5 and 3.0 are still sound
   (0.20%, 0.14%), so the threshold is where it should be.
2. **Features outside the trained box.** Extrapolation; the warning gives the
   radii affected and the worst value against its trained range, e.g.
   `36 of 127 bins outside the trained range, R = 0.1 to 0.487: cum_lam =
   0.8492 (trained 0 to 0.7792)`.

**What the box does not police, and why.** Several of its edges are where the
training walks happened to start, not limits of what the model knows:

- **Bins above the trained ν (5.68) are skipped.** Nothing crosses there
  (< 10⁻⁸ of the walks), and the network was measured to stay within a few
  per cent of one out to ν = 50, so they cannot reach the answer.
- **`nu_origin` is held at its trained maximum (5.77) before the network**,
  and not checked: above ν ≈ 3.5 nothing has crossed, so a walk started
  further out is the same walk. Left to extrapolate, the network moved the
  answer by up to 0.4% at ν_origin = 8 and by 40% at ν_origin = 48 (a flat
  barrier started at 600 Mpc/h); held, by < 0.1% and not at all. Its low end
  is check 1.
- **`nu_back`'s upper edge is not checked**: it is passed only in bins at
  ν > 4, where under 3×10⁻⁵ of the walks cross.
- **`lag` and `cum_lam` are floored at zero**: at a walk's first bin both are
  set by half a grid step, so any grid finer than training's fell "below" the
  trained minimum.
- The box spans **every bin the training walks visited**, not only the bins
  noisy enough to be fitted.

One edge is drawn *inside* the training range instead: **`dlnnu_dlnS` stops
at −0.40**, not at the trained −0.215. No SMT barrier exceeds −0.5 (a constant
one sits on it); only the 5% ripples reach above, so past −0.40 the training
holds 0.14% of its bins and its maximum is set by 14 of them. Barriers rising
with σ were measured sound at −0.44 (0.16%) and wrong at −0.375 (0.35%) and
−0.25 (0.83%), all three passed by the trained edge.

**Measured, on calls with a Monte Carlo answer** (10⁸ paths, 0.15–0.18% noise):

| call | error | old guard | new guard |
|---|---|---|---|
| SMT, grid started at ν = 4.5 | 0.16% | in | in |
| SMT, grid started at ν = 8 | 0.12% | **out** | in |
| flat SMT, walk from 600 Mpc/h (ν_origin = 48) | 40% before the clamp, 0.22% after | out | in |
| SMT / constant / void barrier, grid down to ν = 0.1 | 0.12–0.18%; the bins below ν = 0.21 ±0.17% | out | out (those bins) |
| origin at ν = 2.5 / 3.0 | 0.20% / 0.14% | in | in |
| WDM cutoff α = 0.2 Mpc/h | 0.62% | out | out |
| power law n = −1 | 22% (and P(k) table undersampled for it) | out | out |
| barrier with a bump | 0.57% | out | out |
| B ∝ σ^0.5 / σ^0.25 (rising) | 0.83% / 0.35% | **in** | out |
| B ∝ σ^0.1; SMT × σ^0.2, σ^0.5; SMT with a ±20% step | 0.15–0.18% | in | in |
| power law n = −2; WDM cutoff α = 0.05 | 0.25%; 0.30% | in | **in** |

and without one: **0 of 600** random sound calls flagged (9 cosmologies, SMT
barriers with β down to 10⁻³, origins to 250 Mpc/h, origin ν 3.5–7, 32–400
radii, inner ν 0.25–0.7) where the old guard flagged 216; 0 of 84 validation
curves. The C and the Python agree on every one.

The grids ending below ν = 0.21 are a real, mild extrapolation (±0.17% in
those bins against +0.01–0.07% for the in-domain bins of the same curves), and
are reported. Extending the training's inner edge from ν = 0.25 to 0.1 would
remove it.

**Pairwise checks were tried and rejected.** Occupancy maps of every feature
pair (32 quantile cells per axis) raised no false alarms (1 of 4090 held-out
training curves, 0 of 84 validation) and caught the B ∝ σ^0.5 barrier above,
but not the same shape normalised differently nor B ∝ σ^0.25: they catch what
falls in their gaps, which is not a guarantee. The `dlnnu_dlnS` edge does that
job on physical grounds.

Trained ranges (from `ep_emulator.npz`; the `dlnnu_dlnS` maximum is the guard's
edge, not the data's):

| feature | min | max |
|---|---|---|
| nu | 0.20794 | 5.67833 (bins above not checked) |
| gamma2 | 0.04567 | 0.25350 |
| y | −16.75425 | −0.07319 |
| dlnnu_dlnS | −2.68014 | −0.40 |
| nu_back | 0.33441 | (not checked) |
| lag | 0 | 0.69315 |
| cum_lam | 0 | 0.77915 |
| nu_origin | (check 1) | 5.77245 (held there) |
| gamma2_back | 0.04567 | 0.25350 |

**The guard limits γ², not the radius.** γ² falls steadily past its peak, so
its trained floor of 0.0457 is its value at the origin of the farthest
training walks (~300 Mpc/h). Once the walk starts beyond the trained ν, only
the bins where it can cross are checked, so a flat barrier from 600 Mpc/h whose
crossings all happen inside 250 is in domain, and correctly so (0.22%); a call
is flagged when its *crossings* reach γ² below the floor.

**A per-axis box cannot see everything.** Outside the contract it misses what
differs only in the shape of the whole trajectory: the n = −2 power law and the
weak WDM cutoff above pass at 0.25–0.30%. Campaign 1 showed the same inside the
CDM family: its box contained γ² = 0.18 from small radii and so passed walks at
100 Mpc/h it had never seen; `gamma2_back` now makes that case visible to the
network. Inside the contract — CDM-like spectra, SMT barriers — nothing tested
was missed.

---

## 8. Running the pipeline

```bash
./run_training_pipeline.sh spectra    # 1. P(k) tables from CAMB (local, ~10 min)
./run_training_pipeline.sh submit     # 2. prints the cluster instructions
./run_training_pipeline.sh qa         # 3. QA on the returned runs
./run_training_pipeline.sh train      # 4. CV + final fit
./run_training_pipeline.sh validate   # 5. fresh-ensemble validation + invariance
./run_training_pipeline.sh export     # 6. C weights and stored test cases
./run_training_pipeline.sh studies    # 7. ablation and diagnosis
```

Stages skip themselves if their output exists; `FORCE=1` overrides. Campaigns
2–4 are the same worker with `--campaign 2/3/4` (`CAMPAIGN=` for the SLURM
array). Training the final 48×48 network takes ~20 min on a 72-core node; its
CV is fastest as four fold jobs merged afterwards:

```bash
R="runs runs_v2 runs_v3 runs_v4"
python tools/ep_train_final.py $R --fold 0 --fold-out f0.npz   # ... 3
python tools/ep_train_final.py $R --no-cv -o tools/ep_emulator.npz
python tools/ep_train_final.py --merge f0.npz f1.npz f2.npz f3.npz -o tools/ep_emulator.npz
```

On a cluster that caps running jobs per user (raven: 16, whatever their size),
`ep_packed.slurm` runs several single-node tasks inside one multi-node job.

### Files

**Core (the specification):**

| file | role |
|---|---|
| `ep_model.py` | **The canonical definition.** Baseline rate, hazard integration, the nine features, survival recursion, the density at the radii, `EPEmulator` (inference + guard + persistence). The C implementation must reproduce this exactly. |
| `ep_emulator.py` | The MLP fitter: forward pass, analytic gradients, block-parallel loss, L-BFGS. Training only. |

**Pipeline:**

| file | role |
|---|---|
| `ep_make_spectra.py` | CAMB → shared-grid P(k) table. Runs locally so the cluster needs only numpy and pysif, and so the spectra are pinned. |
| `ep_trainset_design.py` | The domain: parameter boxes, the four campaigns' radius rules, reachability, barrier construction. |
| `ep_trainset_worker.py` | One SLURM array task = one cosmology, all its barriers; `--campaign 1/2/3/4`. |
| `ep_trainset.slurm` | The array script (64 tasks); `CAMPAIGN=` selects the campaign. |
| `ep_packed.slurm` | Several single-node tasks per multi-node job, for per-user job caps. |
| `ep_trainset_load.py` | Reads counts back and derives features/targets through `ep_model`. |
| `ep_qa.py` | Integrity, noise floor, coverage, asymptote, smoothness. |
| `ep_train_final.py` | Cross-validation (whole or per fold), final fit, accuracy record. |
| `ep_validate_v2.py`, `ep_validate_v2.slurm` | Fresh Monte Carlo (sets V, T, C, W), its analysis, grid and origin invariance. |
| `ep_validate.py` | Campaign 1's validation (fresh ensemble at 10⁸ paths, guard, speed). |
| `ep_export_c.py`, `ep_export_test_cases.py` | The C header and the stored test cases (both output forms). |

**Studies (evidence, not needed to train):**

`ep_ratio_pilot.py` (spectral-family pilot), `ep_cdm_coverage.py`,
`ep_cdm_pilot.py`, `ep_cdm_tests.py` (CDM scoping), `ep_feature_ablation.py`
(pilot ablation), `ep_ablation_full.py` (full-statistics ablation),
`ep_lownu_diagnosis.py` (where the residual error lives).

**Data:** `ep_spectra.npz` (campaign 1, N_K = 2400), `ep_spectra_v2.npz`
(campaigns 2–4, N_K = 10000), `ep_validation_spectra.npz`, `ep_emulator.npz`
(trained weights + domain box + accuracy), `ep_validation_mc.npz` (campaign 1's
validation ensemble, also the source of the C test cases). The Monte Carlo
outputs live in `runs/`, `runs_v2/`, `runs_v3/`, `runs_v4/` and `val_v2/` at
the repository root.

### Reproducibility

The design is deterministic in its seed **plus the spectrum table** — barrier
acceptance depends on σ, so keep each campaign's spectra with its outputs.
Monte Carlo seeds are a pure function of (campaign, cosmology index, barrier
index), so any single task reruns bit-identically, and the counts do not depend
on thread count. The fit sums its row blocks in a fixed order, so it does not
depend on thread count either.

---

## 9. Using it from the library

The emulator is one public function, declared in `include/sif/model/excursion_set.h`:

```c
SIF_NODISCARD sif_real* sif_ep_multiplicity_function_emu(const sif_real* radii,
  uint32_t n_radii, const sif_real* sigma, const sif_real* barrier,
  const double* deriv_variance, sif_emu_domain_t* domain, sif_option opt);
```

It returns **n values, the density at the radii**. With
`SIF_EP_EMU_BIN_CENTRES` in `opt` it returns n − 1 bin averages instead, on the
bin centres, as the Monte Carlo does.

It takes **`sigma`, not the packed covariance** that `sif_ep_multiplicity_function`
needs. That is deliberate: the emulator reads only the diagonal, so it wants
`n` numbers where the Monte Carlo wants `n(n+1)/2`, and its cost is linear
rather than quadratic in the radius count. `deriv_variance` is **required**
here, unlike the Monte Carlo path — differencing it off a covariance converges
only at first order, which would reintroduce exactly the grid dependence the
emulator exists to avoid. Both come from `sif_delta_covariance_pk`.

**Place the grid by ν, not by R.** The outermost radius should sit where
ν = B/σ ≳ 3.5 — the emulator then starts the walk where nothing has crossed
yet, and the answer does not depend on exactly where (§6.4); further out costs
nothing (§7). The innermost should keep ν ≳ 0.21, the trained floor. The spacing is free: 65 radii over ν 0.5–4.5 already agree with 257
to 0.02%.

`domain` is optional (pass NULL to skip) and reports where the call sat
relative to the training:

```c
typedef struct {
  int      in_domain;        /* zero if either check below fired */
  uint32_t n_bins_outside;   /* bins with a feature outside the trained box */
  sif_real   nu_origin;        /* B/sigma at the largest radius */
  sif_real   first_step_mass;  /* walks starting above the barrier */
  sif_real   expected_error;   /* 0.00081 in domain, 0.017 outside */
} sif_emu_domain_t;
```

Leaving the trained region is **not** an error and does not fail the call: the
answer is returned, the log names the responsible quantity, and `in_domain` is
cleared. A sampler should be able to notice a degraded proposal without parsing
log output or aborting a chain. NULL is reserved for genuinely invalid input
(null pointers, fewer than four radii, non-positive sigma, non-monotonic
radii).

From Python:

```python
import pysif.model as model

cov, sigma, high_k, dvar = model.delta_covariance_pk(k, pk, radii)
barrier = model.ep_barrier_smt(sigma, alpha=0.7, beta=0.3, gamma=0.8)

f = model.ep_multiplicity_function_emu(radii, sigma, barrier, dvar)   # len(radii)
f_bins = model.ep_multiplicity_function_emu(radii, sigma, barrier, dvar,
                                            binned=True)              # len(radii) - 1
f, dom = model.ep_multiplicity_function_emu(radii, sigma, barrier, dvar,
                                            return_domain=True)
# dom -> {'in_domain': True, 'n_bins_outside': 0, 'nu_origin': 4.7997,
#         'first_step_mass': 7.9e-07, 'expected_error': 0.00081}
```

Code written against the previous version, which read the result on the bin
centres, needs `binned=True` (or to use the radii themselves).

### Implementation

| file | role |
|---|---|
| `src/math/nn.c`, `nn.h` | Internal dense-network evaluation. Not public: there is no training in C, and the weights only ever come from this library's own models. Batched, no allocation, `const` throughout, so it is safe to call from several threads. |
| `src/model/ep_emu_weights.h` | Generated by `ep_export_c.py`. `static const` tables in `.rodata`; every literal verified to round-trip bit-exactly before the header is written. `ep_emu.c` refuses to compile against a header with a different number of inputs. |
| `src/model/ep_emu.c` | The entry point: features, network, survival recursion, the density at the radii, domain report. |
| `src/model/ep_upcrossing.c` | The baseline, shared with the semi-analytic up-crossing path rather than duplicated — `sif__ep_features_fill_diag`, `sif__ep_hazard_bins`, `sif__ep_survival`. |
| `tests/test_ep_emu.c` | Pins the C against stored values from the Python reference. |

### Why the tests look the way they do

A trained model has no closed form to check against, so the correctness
argument is that the C reproduces the Python it was fitted with — exactly. The
stored cases in `tests/ep_emu_cases.h` (regenerate with
`ep_export_test_cases.py`) pin the whole chain, in both output forms: baseline
rate, hazard integration, all nine features, the network, the survival
recursion and the density at the radii. The measured agreement is
**7×10⁻⁸**, which is float rounding on the `sif_real` output; anything at 10⁻⁵
or worse means a feature is being computed differently, and no physical
invariant would notice that.

The remaining tests assert what must hold whatever the network predicts: the
result is a non-negative density integrating to at most one, the two output
forms agree, it does not depend on the radius sampling, and the domain report
fires on both conditions it exists for.

## 10. Known limitations

1. **Contract is CDM-like spectra and SMT barriers.** Cross-family
   generalisation is measurably worse (§7) and is not claimed. Within ΛCDM
   with massive neutrinos it holds across the box (§6.2), with no trend in any
   parameter. That is by design: no cosmological parameter enters, only σ(R)
   and ⟨(dδ/dS)²⟩, i.e. the barrier in units of σ along the walk and the shape
   of P(k). A model beyond ΛCDM whose linear P(k) keeps a CDM shape at the
   redshift asked (wCDM, w₀w_a, scale-independent modified growth) is
   therefore covered in principle; it has not been tested against its own
   Monte Carlo. A spectrum of a different shape (a warm-dark-matter cutoff,
   scale-dependent growth) is the risky case.
2. **Walks crossing at 2–250 Mpc/h.** Where the grid starts is free (§7);
   where the walk can cross is what training covers, to 300 Mpc/h, and the
   guard flags crossings only past the γ² floor (~450–600 Mpc/h).
3. **A per-curve floor of ~2×10⁻³ at ν < 1.5** (§3). It averages away over
   varied curves but not over many thresholds of one barrier: the shallow void
   thresholds of §6.2 sit at 0.16–0.21%, twice their noise. Neither capacity
   nor ensembles move it.
4. **The radius grid must start where ν ≳ 2.5**, better 3.5, and end at
   ν ≳ 0.21; the guard reports both. Starting further out costs nothing.
5. **Accuracy degrades at the edges of feature coverage** — the outer 5% of
   each feature's range carries 2–3× the typical error. γ = 0.5, the edge of the
   barrier box, is where realistic void barriers sit; they were validated
   directly (§6.2).
6. **The emulator reproduces the converged Monte Carlo**, which a coarse
   Monte Carlo grid misses by ~0.1% at 65 radii (§6.3). Comparing the two on a
   coarse grid shows that, and the emulator is the one to trust.
