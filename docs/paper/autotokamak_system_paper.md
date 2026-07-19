# Autotokamak: An Agentic Platform for Learning Fast Surrogates of the Grad–Shafranov Equilibrium

*A system description and mathematical reference.*

---

## Abstract

Autotokamak is a research platform that learns fast machine-learning surrogates for the
Grad–Shafranov (GS) plasma-equilibrium problem and wraps the entire learning process in an
LLM-driven agentic loop that plans, runs, diagnoses, and improves the pipeline autonomously.
The ground-truth solver is TokaMaker (from the OpenFUSION Toolkit), a finite-element solver
for the axisymmetric GS equation. Autotokamak treats that solver as an expensive black-box
forward operator $\mathcal{F}:\mathbf{x}\mapsto\psi$ mapping five scalar shaping/current
parameters to a 2-D poloidal-flux field, and asks: *can we learn a cheap approximation
$\hat{\mathcal{F}}\approx\mathcal{F}$, choosing where to spend expensive solves on purpose, and
can an LLM agent orchestrate that whole workflow?* This document describes the system in
paper form and states the mathematics of every stage: the physics forward problem, the
principal-component + Gaussian-process regression surrogate, the AutoML hyperparameter
optimization, the residual-driven active-learning acquisition, and the outer meta-optimization
loop.

---

## 1. Introduction

### 1.1 Two research threads, one platform

Autotokamak pursues two complementary goals under one codebase:

1. **ML surrogate models** for the GS equation — fast approximations of TokaMaker's FEM
   solve, so that a flux field that costs seconds–minutes of FEM can be predicted in
   milliseconds.
2. **LLM-driven agentic workflows** (built on the URSA plan/execute framework) that plan and
   run equilibrium computations and the surrogate-learning process end-to-end, choosing what
   to compute next based on observed model performance.

### 1.2 Three phases

The learning pipeline is organized into three phases, which the outer agent can revisit in any
order:

| Phase | Name | Produces | Core question |
|---|---|---|---|
| 1 | Dataset generation | `dataset.h5`: pairs $(\mathbf{x}_i,\psi_i)$ | Where in parameter space do we sample? |
| 2 | Surrogate AutoML | `winner.pkl`: a trained $\hat{\mathcal{F}}$ | Which model + hyperparameters best fit the data? |
| 3 | Active meta-loop | improved dataset + surrogate | Given observed weakness, what should we do next? |

Phases 1–2 are the classic "generate data, fit a model" loop. Phase 3 is the contribution
that closes the loop: it measures *where the current surrogate is weak* and directs new
expensive solves at exactly those regions, and it lets an LLM decide the high-level action at
each iteration.

### 1.3 Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{x}=(R_0, a, \kappa, \delta, I_p)\in\mathbb{R}^5$ | tokamak shaping/current parameters (input) |
| $\psi(R,Z)$ | poloidal flux field (output), discretized on a grid to $\Psi\in\mathbb{R}^{n_Z\times n_R}$ |
| $\mathcal{F}$ | the true GS forward operator (TokaMaker) |
| $\hat{\mathcal{F}}$ | the learned surrogate |
| $P=n_Z\,n_R$ | number of output pixels ($96\times 64=6144$ in the shipped grid) |
| $k$ | number of retained PCA components |
| $\mathcal{D}=\{(\mathbf{x}_i,\Psi_i)\}_{i=1}^N$ | dataset of solved equilibria |

---

## 2. The physics forward problem

### 2.1 The Grad–Shafranov equation

TokaMaker solves the axisymmetric ideal-MHD force-balance equation — the **Grad–Shafranov
equation** — for the poloidal flux $\psi(R,Z)$ in the 2-D poloidal $(R,Z)$ plane:

$$
\Delta^\!*\,\psi \;=\; -\,\mu_0\, R^2\, p'(\psi) \;-\; F(\psi)\,F'(\psi).
$$

Here:

- $\psi(R,Z)$ is the poloidal flux function; its level sets $\{\psi=\text{const}\}$ are the
  nested magnetic flux surfaces.
- $\Delta^\!*$ is the **Grad–Shafranov operator**, a non-Laplacian elliptic operator that in
  cylindrical coordinates reads
  $$
  \Delta^\!*\psi \;=\; R\,\frac{\partial}{\partial R}\!\left(\frac{1}{R}\frac{\partial \psi}{\partial R}\right) + \frac{\partial^2 \psi}{\partial Z^2}.
  $$
  The extra $1/R$ factor (relative to the ordinary Laplacian) comes from the toroidal geometry.
- $p(\psi)$ is the plasma pressure profile and $p'(\psi)=\mathrm{d}p/\mathrm{d}\psi$.
- $F(\psi)=R\,B_\phi$ is the poloidal-current function ($B_\phi$ = toroidal field), and
  $F'=\mathrm{d}F/\mathrm{d}\psi$.
- $\mu_0$ is the vacuum permeability.

The right-hand side is nonlinear in $\psi$ (the source terms depend on $\psi$ through the
profiles), so TokaMaker solves it by Picard/Newton iteration on a triangular finite-element
mesh of the plasma cross-section.

### 2.2 Boundary: the Last Closed Flux Surface

In **fixed-boundary** mode (this platform's default), the outermost closed contour — the
**Last Closed Flux Surface (LCFS)** — is prescribed and $\psi$ is solved inside it. The LCFS is
an analytic D-shaped curve parameterized by the shaping parameters. A standard tokamak
boundary parameterization is

$$
\begin{aligned}
R(\theta) &= R_0 + a\cos\!\big(\theta + \delta\sin\theta\big),\\
Z(\theta) &= Z_0 + \kappa\, a\sin\theta, \qquad \theta\in[0,2\pi),
\end{aligned}
$$

with

- $R_0$ — major radius of the plasma center (m),
- $a$ — minor radius / half-width (m),
- $\kappa$ — elongation ($1=$ circular cross-section, $\approx1.6$ ITER-like),
- $\delta$ — triangularity ($0=$ symmetric, $>0=$ D-shaped),
- $Z_0$ — vertical center (fixed to $0$ in the shipped configuration).

An **isoflux** constraint forces $\psi$ to be constant along the sampled LCFS points,
$\psi(R(\theta_j),Z(\theta_j))=\psi_b\ \forall j$. It is more accurate but numerically fragile;
when it fails at construction, the solver drops the constraint and re-solves unconstrained.
Whether isoflux held is recorded per sample — if it fell back, the geometry inputs no longer
describe the saved $\psi$ and the sample is flagged.

### 2.3 The forward operator and its feasible set

For the surrogate problem we abstract the whole solve into a deterministic forward operator

$$
\mathcal{F}:\ \mathbf{x}=(R_0,a,\kappa,\delta,I_p)\ \longmapsto\ \psi(R,Z),
$$

where $I_p$ is the total plasma current (which sets the overall scale of $\psi$, via
current-profile normalization). The output is interpolated from the FEM mesh onto a fixed
rectangular grid $\Psi\in\mathbb{R}^{n_Z\times n_R}$ so that downstream ML can treat it as an
image tensor. Pixels **outside the LCFS** carry no plasma and are set to $\mathrm{NaN}$; all
metrics mask them out.

Not every $\mathbf{x}$ yields a valid solve: extreme shaping (high $\kappa$, high $\delta$,
small $a$) drives the solver into isoflux fallback or outright failure. We therefore treat the
solver as a *constrained* black box with an unknown **feasible set**

$$
\Omega=\{\mathbf{x}: \mathcal{F}(\mathbf{x})\ \text{succeeds}\}\subseteq\mathcal{X},
$$

and we learn a probabilistic model of $\Omega$ (Section 5.3). The shipped seed box is
$R_0\in[0.35,0.55]$ m, $a\in[0.10,0.20]$ m, $\kappa\in[1.0,1.6]$, $\delta\in[0,0.4]$,
$I_p\in[80,200]$ kA.

---

## 3. Phase 1 — Dataset generation

### 3.1 Sampling design

A dataset $\mathcal{D}=\{(\mathbf{x}_i,\Psi_i)\}$ is built by drawing input points from the
seed box and solving each with TokaMaker. Blind sampling uses a space-filling design over the
box $[\mathbf{l},\mathbf{u}]\subset\mathbb{R}^5$ — either **Latin Hypercube Sampling (LHS)** or
a **Sobol** low-discrepancy sequence — chosen because for a fixed budget $N$ these cover a
5-D box far more uniformly than i.i.d. uniform draws (lower star-discrepancy, hence lower
integration/coverage error). Every draw is seeded, so the dataset is reproducible.

### 3.2 The dataset tensor

The HDF5 artifact stores, for $N$ samples:

- `inputs` $\in\mathbb{R}^{N\times 5}$ — columns $(R_0,a,\kappa,\delta,I_p)$;
- `psi` $\in\mathbb{R}^{N\times n_Z\times n_R}$ — the flux fields, NaN outside the LCFS;
- `R`, `Z` — 1-D grid axes;
- `success` $\in\{0,1\}^N$ — whether the (isoflux) solve succeeded;
- provenance (`params_bounds`, `fixed_knobs`, `config_hash`, `oft_version`).

Crucially the stored $\psi$ is **physical poloidal flux in Webers**, not per-sample normalized
$\psi_N\in[0,1]$. Normalizing per sample erases the $I_p$ dependence entirely (the whole scale
information), so the physical field is retained even though it is less low-rank (see §4.2).

---

## 4. Phase 2 — The surrogate model

### 4.1 Problem statement

We seek a cheap map $\hat{\mathcal{F}}_\theta$ minimizing expected field error over the input
distribution $p(\mathbf{x})$:

$$
\theta^\star=\arg\min_\theta\ \mathbb{E}_{\mathbf{x}\sim p}\Big[\ \big\|\,\mathcal{F}(\mathbf{x})-\hat{\mathcal{F}}_\theta(\mathbf{x})\,\big\|^2_{\text{masked}}\ \Big],
$$

where $\|\cdot\|_{\text{masked}}$ is the RMS over in-LCFS pixels. Since we only have the finite
sample $\mathcal{D}$, we minimize the empirical cross-validated version of this loss.
Predicting all $P=6144$ pixels directly is wasteful and noisy, so the surrogate factors into
**(a)** a linear dimensionality reduction and **(b)** a low-dimensional regression.

### 4.2 Output reduction: Principal Component Analysis

Equilibrium $\psi$ fields are highly correlated across samples and lie near a low-dimensional
linear subspace. We exploit this with PCA. Flatten each field to $\mathbf{y}_i\in\mathbb{R}^P$
(NaN pixels imputed by the per-pixel training mean $\bar{\mathbf{y}}$), form the centered data
matrix $\mathbf{Y}_c=[\mathbf{y}_1-\bar{\mathbf{y}},\dots]^\top\in\mathbb{R}^{N\times P}$, and
take its singular value decomposition

$$
\mathbf{Y}_c = \mathbf{U}\,\mathbf{\Sigma}\,\mathbf{V}^\top .
$$

The first $k$ right-singular vectors $\mathbf{V}_k=[\mathbf{v}_1,\dots,\mathbf{v}_k]\in\mathbb{R}^{P\times k}$
form an orthonormal basis ($\mathbf{V}_k^\top\mathbf{V}_k=\mathbf{I}_k$) of the retained flux
subspace. The reduction and reconstruction are

$$
\underbrace{\mathbf{c}_i=\mathbf{V}_k^\top(\mathbf{y}_i-\bar{\mathbf{y}})}_{\text{encode } (P\to k)}
\qquad\qquad
\underbrace{\hat{\mathbf{y}}_i=\bar{\mathbf{y}}+\mathbf{V}_k\,\mathbf{c}_i}_{\text{decode } (k\to P)} .
$$

The $j$-th component captures a fraction of the total field variance

$$
\text{EVR}_j=\frac{\sigma_j^2}{\sum_{l}\sigma_l^2},\qquad
\text{cumulative EV}(k)=\sum_{j=1}^{k}\text{EVR}_j,
$$

and $k$ is chosen (or searched) to reach a target cumulative explained variance. Because the
data is physical $\psi$ (not normalized), it is *less* low-rank than normalized data:
$\approx 8$ components reach $\sim85\%$ EV, versus $\sim99\%$ for normalized data — so $k$ (and
the acquisition PCA size) must be set accordingly.

### 4.3 Latent-space regression: the model zoo

The surrogate learns $k$ scalar regressors, one per PCA coefficient,
$g_j:\mathbb{R}^5\to\mathbb{R}$, so that the full surrogate is

$$
\boxed{\ \hat{\mathcal{F}}(\mathbf{x}) \;=\; \bar{\mathbf{y}} \;+\; \sum_{j=1}^{k} g_j(\mathbf{x})\,\mathbf{v}_j \ }
$$

reshaped back to the $n_Z\times n_R$ grid. Inputs are always standardized first,
$\mathbf{z}=(\mathbf{x}-\boldsymbol{\mu}_x)/\mathbf{s}_x$, because the raw features span wildly
different scales ($I_p\sim10^5$ vs $R_0\sim0.4$); without it every kernel distance is dominated
by $I_p$ and the model collapses to the mean predictor. Four regression families ("the zoo")
compete:

**(a) Gaussian Process (GP) regression** — the headline model, with a squared-exponential
(RBF) kernel plus a white-noise term:

$$
k(\mathbf{z},\mathbf{z}') \;=\; \sigma_f^2\,\exp\!\left(-\frac{\|\mathbf{z}-\mathbf{z}'\|^2}{2\ell^2}\right) \;+\; \sigma_n^2\,\delta_{\mathbf{z}\mathbf{z}'} .
$$

Given training inputs $\mathbf{Z}$ and a coefficient target vector $\mathbf{c}^{(j)}$, GP
regression places a Gaussian prior over functions and conditions on data. The **posterior
predictive** at a new $\mathbf{z}_\star$ is Gaussian with closed-form mean and variance:

$$
\begin{aligned}
g_j(\mathbf{z}_\star) \;=\; \mu_j(\mathbf{z}_\star) &= \mathbf{k}_\star^\top\,(\mathbf{K}+\alpha\mathbf{I})^{-1}\,\mathbf{c}^{(j)},\\[4pt]
\sigma_j^2(\mathbf{z}_\star) &= k(\mathbf{z}_\star,\mathbf{z}_\star) - \mathbf{k}_\star^\top\,(\mathbf{K}+\alpha\mathbf{I})^{-1}\,\mathbf{k}_\star,
\end{aligned}
$$

where $\mathbf{K}_{ab}=k(\mathbf{z}_a,\mathbf{z}_b)$ is the Gram matrix, $\mathbf{k}_\star$ is
the vector of covariances between $\mathbf{z}_\star$ and the training points, and $\alpha$ is a
jitter/regularizer. The kernel hyperparameters $(\sigma_f,\ell,\sigma_n)$ are fit by maximizing
the **log marginal likelihood**

$$
\log p(\mathbf{c}^{(j)}\mid\mathbf{Z}) = -\tfrac{1}{2}\,\mathbf{c}^{(j)\top}\mathbf{K}_y^{-1}\mathbf{c}^{(j)} - \tfrac{1}{2}\log|\mathbf{K}_y| - \tfrac{N}{2}\log 2\pi,
\quad \mathbf{K}_y=\mathbf{K}+\sigma_n^2\mathbf{I}.
$$

The GP is the only family that also returns a *calibrated predictive variance* $\sigma_j^2$,
which the active-learning stage reuses.

**(b) Kernel Ridge Regression (KRR)** — same kernel idea, no uncertainty, cheaper:

$$
g_j(\mathbf{z}_\star)=\mathbf{k}_\star^\top\boldsymbol{\eta},\qquad
\boldsymbol{\eta}=(\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{c}^{(j)},
$$

with kernel $\in\{\text{RBF},\text{Laplacian}\}$, regularizer $\lambda$ (`alpha`), and
bandwidth `gamma`.

**(c) Polynomial ridge** — explicit polynomial feature map $\boldsymbol{\phi}(\mathbf{z})$ of
degree $d\in\{1,2,3\}$ into ridge regression:

$$
\mathbf{w}_j=\arg\min_{\mathbf{w}}\ \|\boldsymbol{\Phi}\mathbf{w}-\mathbf{c}^{(j)}\|^2+\lambda\|\mathbf{w}\|^2
=(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}+\lambda\mathbf{I})^{-1}\boldsymbol{\Phi}^\top\mathbf{c}^{(j)},
\qquad g_j(\mathbf{z}_\star)=\boldsymbol{\phi}(\mathbf{z}_\star)^\top\mathbf{w}_j .
$$

**(d) Small MLP** — a capped multilayer perceptron ($\le 2$ hidden layers, width $\le 256$),
a universal approximator trained by L-BFGS/Adam with L2 penalty $\alpha$ and learning rate.

Deep operator/PINN/FNO families are explicitly out of the proof-of-concept scope.

### 4.4 AutoML: the inner optimization problem

For each candidate model the hyperparameters are tuned to minimize **$k$-fold
cross-validated field RMSE in physical $\psi$ units**. Let $\{(\text{tr}_f,\text{va}_f)\}_{f=1}^{K}$
be the CV folds. The inner objective for a hyperparameter vector $\mathbf{h}$ is

$$
\mathcal{L}(\mathbf{h}) \;=\; \frac{1}{K}\sum_{f=1}^{K}
\operatorname{RMSE}\!\Big(\Psi_{\text{va}_f},\ \hat{\mathcal{F}}_{\mathbf{h}}^{(\text{tr}_f)}(\mathbf{X}_{\text{va}_f})\Big),
\qquad
\operatorname{RMSE}(\Psi,\hat\Psi)=\sqrt{\frac{1}{|\mathcal{M}|}\sum_{(i,p)\in\mathcal{M}}(\Psi_{ip}-\hat\Psi_{ip})^2},
$$

where $\mathcal{M}$ is the set of finite (in-LCFS) cells and the PCA is *refit on each fold's
train set only* to avoid leakage. The search solves

$$
\mathbf{h}^\star=\arg\min_{\mathbf{h}\in\mathcal{H}}\ \mathcal{L}(\mathbf{h})
$$

over the model's search space $\mathcal{H}$ using **Optuna's Tree-structured Parzen Estimator
(TPE)** sampler. TPE is a Bayesian-optimization method: it models $p(\mathbf{h}\mid\mathcal{L}
<\mathcal{L}^*)=\ell(\mathbf{h})$ and $p(\mathbf{h}\mid\mathcal{L}\ge\mathcal{L}^*)=g(\mathbf{h})$
and proposes points maximizing the ratio $\ell(\mathbf{h})/g(\mathbf{h})$, i.e. it expands the
Expected-Improvement acquisition without a GP surrogate over $\mathcal{H}$. Failed trials
(e.g. an MLP exceeding its cap) return a large sentinel objective ($10^{10}$) so the search
continues. After the search, the winning $(\text{model},\mathbf{h}^\star,k)$ is **refit on all
non-test samples** and pickled as `winner.pkl` together with its PCA handle.

### 4.5 Evaluation and scoring

The winner is judged against a **trivial baseline** — the per-pixel training mean predictor
$\hat\Psi^{\text{base}}=\bar\Psi_{\text{train}}$ (broadcast over the eval set). Reported metrics
include field RMSE, MAE, relative $L_2$, $R^2$, correlation, and tolerance-band accuracies. The
two headline quantities are

$$
\text{RMSE ratio}=\frac{\operatorname{RMSE}(\hat{\mathcal{F}})}{\operatorname{RMSE}(\hat\Psi^{\text{base}})},
\qquad
\text{error reduction}=100\big(1-\text{RMSE ratio}\big)\ \%,
$$

so "90% accuracy" means the surrogate cuts the baseline field error by 90%. Any acceptable
winner must have RMSE ratio $<1$ (beat doing nothing).

---

## 5. Phase 3 — Active learning: sampling where the model is weak

Blind LHS spends the same effort everywhere. Active learning instead picks the next batch of
expensive solves on purpose, targeting the geometries where the *current* surrogate is
measurably worst, while respecting feasibility and covering a possibly-wider target envelope.
This is the `enrich_active` action. Its default strategy is **residual-driven Upper Confidence
Bound (residual-UCB)**, with fallbacks to a model-agnostic variance criterion and finally to
space-filling.

### 5.1 The acquisition signal: out-of-fold residuals of the current winner

The "observed model performance" signal is the winner's **out-of-fold (OOF) residual** on the
train pool. For each CV fold, the winner's architecture, hyperparameters, and PCA size are
*refit on the fold's train rows*, and each held-out sample's mean absolute field error is
recorded:

$$
r_i \;=\; \frac{1}{|\mathcal{M}_i|}\sum_{p\in\mathcal{M}_i}\big|\,\Psi_{ip}-\hat{\mathcal{F}}^{(-\text{fold}(i))}(\mathbf{x}_i)_p\,\big| .
$$

OOF (not train-set) residuals are essential: interpolating models like GP and KRR fit their
training points to $\approx0$ error, so train residuals carry no signal about generalization.

### 5.2 The error model and the UCB score

We fit a second GP — the **error model** — on the log residuals over box-standardized inputs,

$$
g:\ \mathbf{x}\ \longmapsto\ \log r,\qquad g\sim\mathcal{GP}\big(\mu_g,\,k_{\text{ARD}}\big),
$$

using an **ARD (Automatic Relevance Determination) RBF kernel** with a separate length scale
per input dimension,

$$
k_{\text{ARD}}(\mathbf{z},\mathbf{z}')=\sigma_f^2\exp\!\left(-\tfrac{1}{2}\sum_{d=1}^{5}\frac{(z_d-z'_d)^2}{\ell_d^2}\right)+\sigma_n^2\delta,
$$

so the error model can learn that failures/weaknesses align with a few input directions
(e.g. elongation) and down-weight irrelevant ones. The log transform is deliberate: residual
error is multiplicatively structured, and logging prevents a few catastrophic samples from
dominating the fit. The error model produces posterior mean $\mu_g(\mathbf{x})$ and standard
deviation $\sigma_g(\mathbf{x})$ (from the GP posterior equations of §4.3).

Candidate points are scored by an **Upper Confidence Bound**, in log space so the feasibility
term composes additively:

$$
\boxed{\ a(\mathbf{x}) \;=\; \underbrace{\mu_g(\mathbf{x})}_{\text{exploit measured weakness}} \;+\; \beta\,\underbrace{\sigma_g(\mathbf{x})}_{\text{explore unseen regions}} \;+\; \underbrace{\log P_{\text{feas}}(\mathbf{x})}_{\text{avoid infeasible solves}}\ }
$$

The exploration weight $\beta\in[0,3]$ is exposed to the meta-agent: $\beta=0$ is pure
exploitation (may cluster on one weak pocket); larger $\beta$ trades toward exploring regions
the error model has never seen — which is precisely what covers a target envelope wider than
the seed box. Candidates are drawn from a large Sobol pool over the envelope bounds.

### 5.3 Feasibility weighting: modeling the feasible set $\Omega$

Failed solves cost the same wall-clock as successes, so acquisition is discounted by the
**probability of feasibility**. A GP classifier (GPC) with an ARD-RBF kernel is fit on **all
attempted inputs**, successes and failures, giving $P(\text{success}\mid\mathbf{x})$. The
feasibility weight is the *squared* probability, floored:

$$
P_{\text{feas}}(\mathbf{x}) \;=\; \max\!\Big(\,P(\text{success}\mid\mathbf{x})^2,\ \varepsilon\,\Big),\qquad \varepsilon=0.02 .
$$

Squaring ensures a known-bad region cannot out-bid a merely well-covered feasible one on prior
variance alone; the (harsh) floor $\varepsilon$ ensures an unlucky region is down-weighted but
never permanently written off. If the GPC fails to fit, a distance-weighted $k$-NN estimate of
$P(\text{success})$ is used instead. ARD matters because solve failures typically align with a
few shaping directions; an isotropic $k$-NN lets irrelevant dimensions dilute exactly that
signal.

### 5.4 Batch selection: exact greedy "kriging believer"

We need a *diverse batch* of $n$ new points, not $n$ copies of the single argmax. The key fact:
**a GP's posterior variance depends only on input locations, never on the observed $y$-values.**
Under the "kriging believer" heuristic, a hallucinated observation placed at the GP's own
posterior mean leaves $\mu_g$ unchanged but shrinks $\sigma_g$ near the picked point. So we
select greedily:

1. Pick $\mathbf{x}^{(1)}=\arg\max_{\mathbf{x}} a(\mathbf{x})$.
2. Condition the GP variance on $\mathbf{x}^{(1)}$ (exact rank-1 Cholesky border update),
   which lowers $\sigma_g$ — and hence $a$ — in its neighborhood.
3. Re-score the remaining pool and repeat until $n$ points are chosen.

Because only $\sigma_g$ changes, the update is exact and cheap ($O(n\cdot m)$ per pick for $m$
candidates), and the batch spreads out instead of piling onto one variance peak.

### 5.5 Fallbacks: model-agnostic variance and space-filling

When no trustworthy winner exists yet, acquisition degrades gracefully down a ladder recorded
in the result's provenance:

$$
\texttt{residual\_ucb}\ \longrightarrow\ \texttt{gp\_variance}\ \longrightarrow\ \texttt{maximin\_fallback}.
$$

**gp\_variance (ALM criterion).** Build an independent PCA-GP *emulator*: fit one GP per PCA
component and score a candidate by the total predictive variance of the *reconstructed* field.
Because the PCA basis is orthonormal, the pixel-summed field variance collapses **exactly** to
a weighted sum of per-component GP variances:

$$
\sum_{\text{pixels}}\operatorname{Var}\big[\hat\psi(\mathbf{x})\big] \;=\; \sum_{j=1}^{k} s_j^2\,\sigma_j^2(\mathbf{x}),
$$

where $s_j$ is the train-set standard deviation of coefficient $j$ (the GPs are fit on
standardized coefficients) and $\sigma_j^2(\mathbf{x})$ is component $j$'s posterior variance —
no sampling, no approximation. This is the classical Active-Learning-MacKay (ALM) criterion,
also selected greedily with the same exact variance conditioning and feasibility weighting.

**maximin\_fallback.** With too few successful solves to trust any GP ($<8$), fall back to
feasibility-weighted greedy **maximin space-filling** in standardized input space — each new
point maximizes its (feasibility-weighted) distance to all previously attempted inputs:

$$
\mathbf{x}^{(t+1)} = \arg\max_{\mathbf{x}\in\text{pool}}\ P_{\text{feas}}(\mathbf{x})\cdot \min_{\mathbf{x}'\in\text{chosen}\,\cup\,\text{attempted}} \|\mathbf{x}-\mathbf{x}'\| .
$$

### 5.6 Honest evaluation: the frozen envelope and per-cell errors

Once active learning samples geometries *outside* the seed box, a test shard carved from the
seed dataset measures accuracy on the wrong distribution. The fix is a single **frozen,
full-envelope evaluation set**: $n_{\text{eval}}$ points (default 256) are drawn once by LHS
over the *target envelope* (the seed box widened, clipped to physically sane values), solved,
and never grown — every winner across every meta-iteration is scored on the same samples. The
acquisition draws over the same envelope, so the eval and acquisition distributions agree.

Aggregate RMSE cannot distinguish "uniformly mediocre" from "great in the seed box, broken at
high $\kappa$", which is exactly the distinction active sampling must close. So scoring is
**stratified per geometry cell**: each eval sample is binned into a joint cell ($n_b$ bins per
dimension $\Rightarrow n_b^5$ cells) plus per-parameter tercile marginals, and RMSE is computed
per occupied cell. The headline numbers are

$$
\text{worst\_cell\_rmse}=\max_{c\in\text{occupied}}\operatorname{RMSE}_c,\qquad
\text{mean\_cell\_rmse}=\frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\operatorname{RMSE}_c,
$$

the latter weighting every geometry region equally regardless of how many samples landed in it.
An accuracy-style early-stopping bar can be phrased per cell: stop only once *every* occupied
cell beats its own local baseline by the target margin,
$\min_c 100\,(1-\operatorname{RMSE}_c/\operatorname{RMSE}_c^{\text{base}})\ge\tau$.

### 5.7 Control arm

To prove active sampling is worth it, the blind-LHS action `regen_dataset` is retained
unchanged as an **A/B control**. The headline experiment fixes the seed dataset and budgets and
compares `enrich_active` vs `regen_dataset` on the same frozen envelope eval set, judged on
**worst-cell RMSE reduction per OFT solve spent**.

---

## 6. The meta-loop — outer optimization by an LLM agent

### 6.1 The decision problem

Phase 3's outer loop is a sequential decision process. At iteration $t$ the agent observes a
deterministic **diagnosis** of the current state $s_t$ (dataset statistics, winner metrics,
per-cell errors, feasibility rates) and must choose one **typed action** $u_t$ from a small
vocabulary:

$$
u_t\in\mathcal{A}=\{\ \texttt{regen\_dataset},\ \texttt{enrich\_active},\ \texttt{extend\_search},\ \texttt{terminate}\ \},
$$

- `regen_dataset(overrides)` — reshape/resample the dataset (blind LHS; the control arm);
- `enrich_active(n_new, strategy, β, …)` — the active-learning acquisition of §5;
- `extend_search(focus)` — run a fresh Phase-2 AutoML round with directives (which models to
  emphasize, which hyperparameters to widen, trial budget);
- `terminate(reason)` — stop.

The action is emitted as validated structured JSON (`ActionDecision`), so the action space is
small, typed, and checked before dispatch. The loop halts when the agent terminates, when an
early-stopping quality bar is met (absolute RMSE, RMSE ratio, aggregate accuracy, or the
strict per-cell worst-cell accuracy), or when a max-iteration safety cap is hit.

### 6.2 Division of labor: what is learned vs. what is deterministic

A deliberate split keeps the science reproducible while giving the LLM genuine open-ended room:

| Job | Owner |
|---|---|
| Which meta-action to take, and its knobs ($n_{\text{new}}$, $\beta$) | LLM (DSPy module, GEPA-optimizable) |
| Which nested Phase-2 round to run | LLM (DSPy module) |
| Acquisition core (residual GP, UCB, batch, feasibility) | Deterministic library |
| Diagnosis of current state | Deterministic library |
| Open-ended representation / data-engineering search | URSA code-generation agent |

Everything numerically load-bearing (the GP kernels, the UCB math, the CV folds) is
deterministic library code; only the *high-level choices* are the LLM surface.

### 6.3 The meta-objective and its optimization (GEPA)

The whole meta-run is scored by a composite objective combining **hard gates** (which must all
pass for any nonzero score) and weighted **quality terms**:

$$
S \;=\;
\Big(\textstyle\prod_{g\in\text{gates}}\mathbb{1}[g\ \text{passes}]\Big)
\cdot \sum_{q}\ w_q\,Q_q,
\qquad \sum_q w_q = 1 .
$$

The hard gates check that the deliverables exist, the report parses, an iteration log is
present, and the winner actually loads and predicts. The quality terms (with weights) reward:

| Term | $w_q$ | What it measures |
|---|---|---|
| `final_rmse_vs_baseline` | 0.35 | $1-\text{RMSE}_{\text{final}}/\text{RMSE}_{\text{base}}$, clipped to $[0,1]$ |
| `improvement_over_iterations` | 0.20 | $(\text{RMSE}_{\text{first}}-\text{RMSE}_{\text{last}})/\text{RMSE}_{\text{first}}$ |
| `budget_efficiency` | 0.10 | fraction of total improvement achieved by the halfway iteration |
| `no_waste` | 0.15 | fraction of post-first-winner iterations that improved best RMSE by $\ge1\%$ |
| `terminated_by_agent` | 0.15 | decisive stop (agent-chosen or target-reached) vs. cap-riding |
| `runner_cleanliness` | 0.05 | only valid action types were used |

This objective is what makes a decisive short run score higher than a budget-burning one that
reaches the same accuracy — an explicit pressure toward efficiency. The LLM's decision module is
authored in **DSPy** and its prompt/instructions are optimized by **GEPA** (a reflective
prompt-evolution optimizer) to maximize the expected score $\mathbb{E}[S]$ across recorded
meta-traces. DSPy is retained *only* for this GEPA-optimizability (and as an elegant way to
write typed LLM programs); the numerical pipeline does not depend on it.

---

## 7. The agent layer (URSA) and system architecture

The platform is a **two-layer** codebase:

- **Layer 1 — `core`**: the stable physics/IO substrate — schema, LCFS + mesh construction,
  the TokaMaker solve with isoflux-fallback retry, diagnostic scalar extraction ($R_{\text{axis}}$,
  $q_0$, $q_{95}$, …), atomic IO, logging. A hard invariant: `OFT_env` is a **process-wide
  singleton** (one per Python kernel), so all solve parallelism is process-level (subprocess /
  `spawn`), never threads.
- **Layer 2 — `agent`**: URSA plan/execute runners, the prompt YAMLs they consume, the meta-loop
  orchestrator, and the DSPy scorers/modules. Two runner styles exist: **structured** mode runs
  the deterministic library with one typed LLM decision per round; **codegen** mode lets a URSA
  agent author the runner script itself. The `PlanningAgent` emits steps; the `ExecutionAgent`
  writes code, runs it, inspects output, and threads a summary into the next step; a feedback
  variant re-invokes the planner with execution history so it can patch failures.

The URSA agent's genuine open-ended role is the **representation / data-engineering search**:
under a fixed evaluation protocol it reads the model diagnostics and authors and evaluates
alternative input featurizations and output transforms — the one part of the pipeline
deliberately left to code-generating LLM search rather than to a fixed library.

Data flows Phase 1 $\to$ 2 $\to$ 3, and every run also writes an `experiments/<run_id>/trace.json`
plus a self-contained HTML report (physics + dataset provenance, score gates, quality bars, the
winner rationale in the agent's own words, model comparison, per-round search decisions, an
Optuna history plot, and an evaluation gallery).

---

## 8. Summary

Autotokamak treats an expensive FEM equilibrium solver as a black-box forward operator
$\mathcal{F}:\mathbb{R}^5\to\psi(R,Z)$ and learns a fast surrogate $\hat{\mathcal{F}}$ by
(i) reducing the flux field to a handful of PCA coefficients, (ii) regressing each coefficient
from the shaping parameters with a small zoo of models (GP, kernel-ridge, poly-ridge, MLP),
(iii) tuning hyperparameters by TPE Bayesian optimization against cross-validated field RMSE,
and (iv) actively growing the training set with a residual-driven UCB acquisition that fits a
GP error model to the winner's out-of-fold residuals, discounts by a GP-classifier feasibility
model, and selects a diverse batch by exact kriging-believer variance conditioning. An outer
meta-loop, whose high-level decisions are made by a DSPy/GEPA-optimized LLM and whose numerics
are deterministic library code, iterates these phases against a frozen full-envelope,
per-geometry-cell evaluation set — measuring, and closing, exactly where the surrogate is weak.

The recurring mathematical objects — the GS PDE, the orthonormal PCA basis, the GP posterior
mean/variance, the cross-validated RMSE objective, the UCB acquisition, and the gated composite
meta-score — are the load-bearing formulas of the entire system.
```
