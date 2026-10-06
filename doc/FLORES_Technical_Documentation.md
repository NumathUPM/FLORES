# FLORES — Flow Linear Operators: Resolvent and Eigenvalue Stability

**Technical Documentation**

NUMATH Research Group · School of Aeronautical and Space Engineering (ETSIAE) · Universidad Politécnica de Madrid (UPM) · Project TRANSDIFFUSE

> Original code author: **Alejandro Martínez Cava** (Doctoral Thesis, NUMATH/UPM)
> Repository: <https://github.com/NumathUPM/FLORES> · License: MIT

---

### About this document

This document is the Markdown version of the FLORES technical documentation,
**cross-checked against the source code of the `miguel_dev` branch**
(`solver/`, `tools/`, `python_env_installation/`, `test_cases/`).

Wherever the original text did not match the code, the text has been
corrected to describe what the code actually does. Points that need a
decision or a fix in the code are flagged with a box like this:

> ⚠️ **Implementation note** — description of the discrepancy.

All such notes are collected in
[Chapter 11 — Known Discrepancies and Open Issues](#11-known-discrepancies-and-open-issues).

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Software Architecture](#2-software-architecture)
3. [Installation](#3-installation)
4. [Test Cases](#4-test-cases)
5. [Domain Reduction](#5-domain-reduction)
6. [Eigenvalue Solver](#6-eigenvalue-solver)
7. [Resolvent Analysis](#7-resolvent-analysis)
8. [Post-Processing](#8-post-processing)
9. [SLURM Job Submission](#9-slurm-job-submission)
10. [Summary of Design Decisions](#10-summary-of-design-decisions)
11. [Known Discrepancies and Open Issues](#11-known-discrepancies-and-open-issues)
12. [Glossary](#12-glossary)
- [Appendix A — Original vs. Refactored Solver](#appendix-a--original-vs-refactored-solver)

---

## 1. Introduction

### 1.1 Overview

**FLORES** is a Python-based computational toolkit for global hydrodynamic
stability analysis and resolvent-based input/output analysis of compressible
flows. It is developed within the **NUMATH** research group at the *Escuela
Técnica Superior de Ingeniería Aeronáutica y del Espacio* (ETSIAE),
Universidad Politécnica de Madrid (UPM), in the framework of the
**TRANSDIFFUSE** project.

The toolkit performs two complementary analyses:

1. **Global eigenvalue stability analysis.** Computes the eigenspectrum of the
   linearised Navier–Stokes operator and the associated global modes (direct
   modes, adjoint modes and structural sensitivity).
2. **Resolvent analysis.** Computes the optimal forcing and response modes and
   the associated gain curves $`\lambda_1^2(\omega)`$, characterising
   pseudo-resonances and amplification mechanisms.

Both analyses use numerical Jacobian matrices produced by the **DLR TAU
Code**. These are read from `samg.matrix.amg.pval` (SAMG/NetCDF format),
together with the cell volumes (`samg.matrix.vol`) and the DOF coordinates
(`samg.matrix.coo`).

### 1.2 Mathematical Background

#### 1.2.1 Linearised Compressible Navier–Stokes

The starting point is the compressible Reynolds-Averaged Navier–Stokes (RANS)
equations, written in conservative form for the state vector
$`\boldsymbol{q} = (\rho,\ \rho u,\ \rho v,\ \rho w,\ \rho E)^T`$ (plus the
turbulence variables, if any):

```math
\mathbf{M}\,\frac{\partial\boldsymbol{q}}{\partial t} = \mathcal{R}(\boldsymbol{q}),
\tag{1}
```

Here $`\mathcal{R}(\boldsymbol{q})`$ is the discrete spatial residual (inviscid
and viscous fluxes), and $`\mathbf{M}`$ is the diagonal mass matrix whose
entries are the cell volumes of the finite-volume mesh. A steady base flow
$`\bar{\boldsymbol{q}}`$ satisfies $`\mathcal{R}(\bar{\boldsymbol{q}}) = \boldsymbol{0}`$.

Any flow state is decomposed into the base flow plus a small perturbation
$`\varepsilon\tilde{\boldsymbol{q}}`$, with $`\varepsilon \ll 1`$:

```math
\boldsymbol{q}(\boldsymbol{x}, t) = \bar{\boldsymbol{q}}(\boldsymbol{x}) + \varepsilon\,\tilde{\boldsymbol{q}}(\boldsymbol{x}, t).
\tag{2}
```

Substituting (2) into (1), expanding in a Taylor series around
$`\bar{\boldsymbol{q}}`$ and keeping only first-order terms gives the
*linearised Navier–Stokes* (LNS) system:

```math
\mathbf{M}\,\frac{\partial\tilde{\boldsymbol{q}}}{\partial t}
= \underbrace{\left.\frac{\partial\mathcal{R}}{\partial\boldsymbol{q}}\right|_{\bar{\boldsymbol{q}}}}_{\mathbf{J}(\bar{\boldsymbol{q}})}\tilde{\boldsymbol{q}}
+ \mathbf{M}\,\mathbf{P}\,\boldsymbol{f}(t),
\tag{3}
```

Here $`\mathbf{J}(\bar{\boldsymbol{q}})`$ is the *Jacobian matrix* of the
residual evaluated at the base flow. $`\mathbf{P}`$ is a prolongation operator
that maps an external body force $`\boldsymbol{f}(t)`$ into the state space
(see [§1.2.3](#123-resolvent-analysis)). With
$`\mathbf{A} \equiv \mathbf{J}(\bar{\boldsymbol{q}})`$:

```math
\mathbf{M}\,\dot{\tilde{\boldsymbol{q}}} = \mathbf{A}\,\tilde{\boldsymbol{q}} + \mathbf{M}\,\mathbf{P}\,\boldsymbol{f}(t),
\qquad
\mathbf{M}\,\dot{\tilde{\boldsymbol{q}}} = \mathbf{A}\,\tilde{\boldsymbol{q}} \quad (\boldsymbol{f}=\boldsymbol{0}).
\tag{4}
```

**Numerical Jacobian.** FLORES does not assemble $`\mathbf{A}`$ analytically.
It uses the numerical Jacobian that the **DLR TAU Code** computes by
finite-difference perturbation of the discretised residual. The sparse matrix
is stored in SAMG/NetCDF format and read at runtime with
`input_output.openjacobian()`. This function converts the Fortran 1-based
indices to 0-based and returns a SciPy CSR matrix. The entries of
$`\mathbf{A}`$ are then scaled by the non-dimensionalisation factor
$`1/(M_\infty\sqrt{\gamma})`$, with $`\gamma = 1.4`$ hard-coded.

> ⚠️ **Implementation note — ordering of the state vector.** The DOFs are
> interleaved node by node: $`[\rho_0, u_0, w_0, e_0, (t_{1,0}, t_{2,0}),\ \rho_1, \dots]`$.
> For 2-D cases the code assumes the variable order `rho, u, w, e[, turb1, turb2]`,
> with the second spatial coordinate called **z**. `neq` is read from the
> Jacobian file.

**Jacobian volume scaling (TAU option).** The Jacobian exported by TAU may or
may not be divided by the cell volumes. This is controlled by the TAU
parameter file option

```
Jacobian volume scaling (0/1): 1
```

| TAU option | Exported matrix $`\mathbf{A}_\mathrm{TAU}`$ | Physical problem |
|---|---|---|
| `1` | $`\mathbf{M}^{-1}\mathbf{J}`$ (already divided by volume) | $`\mathbf{A}_\mathrm{TAU}\hat{\boldsymbol{q}} = \sigma\hat{\boldsymbol{q}}`$, i.e. the standard EVP (7) with $`\mathbf{L}=\mathbf{A}_\mathrm{TAU}`$ |
| `0` | $`\mathbf{J}`$ (not divided) | $`\mathbf{A}_\mathrm{TAU}\hat{\boldsymbol{q}} = \sigma\mathbf{M}\hat{\boldsymbol{q}}`$, i.e. the generalised EVP (6) |

FLORES does **not** read this option from TAU, and the `.pval` Jacobian file
does not record it. The user must choose the solver settings to match how the
Jacobian was generated:

| TAU `Jacobian volume scaling` | `eig_solver.py` | `resolvent_solver.py` |
|---|---|---|
| `1` | `gen = False` ✅ (default) | ✅ consistent as implemented ($`L = i\omega\mathbf{I} - \mathbf{A}`$, RHS $`\mathbf{P}f`$) |
| `0` | `gen = True` ✅ | ❌ not supported: it would need $`L = i\omega\mathbf{M} - \mathbf{A}`$ and RHS $`\mathbf{M}\mathbf{P}f`$ |

Mixing them, e.g. a Jacobian generated with `0` solved with `gen = False`, or
`1` with `gen = True`, gives eigenvalues and modes that are wrong by a
non-uniform (cell-volume dependent) scaling and **does not raise any error**.
See [§11](#11-known-discrepancies-and-open-issues), item 3.

**Turbulent and mean-flow analyses.** When the base flow comes from a RANS
simulation, the linearisation is performed around the steady turbulent
solution, which satisfies $`\mathcal{R}(\bar{\boldsymbol{q}}) = \boldsymbol{0}`$
including the turbulence model. Alternatively, a *mean-flow analysis* uses the
time-averaged solution of an unsteady RANS calculation as the base state. This
approach has been shown to give accurate frequency predictions for wake flows
near the bifurcation point, where the mean flow is approximately marginally
stable.

#### 1.2.2 Global Eigenvalue Stability Analysis

##### Direct problem

Assume a perturbation of the form

```math
\tilde{\boldsymbol{q}}(\boldsymbol{x}, t) = \hat{\boldsymbol{q}}(\boldsymbol{x})\,e^{\sigma t} + \text{c.c.}
\tag{5}
```

Substituting into (4) gives the *generalised eigenvalue problem* (GEVP):

```math
\mathbf{A}\,\hat{\boldsymbol{q}} = \sigma\,\mathbf{M}\,\hat{\boldsymbol{q}}.
\tag{6}
```

Defining $`\mathbf{L} = \mathbf{M}^{-1}\mathbf{A}`$ turns it into a standard EVP:

```math
\mathbf{L}\,\hat{\boldsymbol{q}} = \sigma\,\hat{\boldsymbol{q}}.
\tag{7}
```

The complex eigenvalue $`\sigma = \sigma_r + i\sigma_i`$ describes the dynamics
of each *global mode*:

- $`\sigma_r`$ is the temporal growth rate. The flow is linearly unstable if
  $`\sigma_r > 0`$ for any mode. The bifurcation point is at $`\sigma_r = 0`$.
- $`\sigma_i`$ is the angular frequency, related to the Strouhal number by
  $`\mathrm{St} = \sigma_i / 2\pi`$.

The right eigenvector $`\hat{\boldsymbol{q}}`$ is the *direct global mode*.

> ⚠️ **Implementation note — which problem is actually solved.**
> `eig_solver.py` has a `gen` flag (default `False`):
>
> - `gen = False` (default) solves the **standard** problem
>   $`\mathbf{A}\hat{\boldsymbol{q}} = \sigma\hat{\boldsymbol{q}}`$ without the
>   mass matrix. This is correct for Jacobians generated with TAU
>   `Jacobian volume scaling = 1` ($`\mathbf{A} = \mathbf{M}^{-1}\mathbf{J}`$, problem (7)).
> - `gen = True` solves the generalised problem (6) with $`\mathbf{B} = \mathbf{M}`$
>   (cell volumes). This is correct for Jacobians generated with
>   `Jacobian volume scaling = 0`.
>
> The choice is left to the user (see the table in [§1.2.1](#121-linearised-compressible-navierstokes)).

##### Adjoint problem

The direct problem alone does not tell *where* the flow is most sensitive to
perturbations, nor how receptive it is to external forcing. That information
comes from the *adjoint problem*, which arises from the Lagrangian formulation
of the eigenvalue sensitivity (Martínez-Cava, 2019, §2.1.1).

The discrete inner product weighted by the mass matrix is

```math
\langle \boldsymbol{c},\,\boldsymbol{d} \rangle = \int_\Omega \boldsymbol{c}^H \boldsymbol{d}\,\mathrm{d}\Omega = \boldsymbol{c}^H\mathbf{M}\boldsymbol{d},
\tag{8}
```

The adjoint operator $`\mathbf{L}^+`$ is defined by
$`\langle \boldsymbol{c}, \mathbf{L}\boldsymbol{d}\rangle = \langle\mathbf{L}^+\boldsymbol{c}, \boldsymbol{d}\rangle`$,
which gives

```math
\mathbf{L}^+ = \mathbf{M}^{-1}\mathbf{A}^H.
\tag{9}
```

(For $`\mathbf{L} = \mathbf{M}^{-1}\mathbf{A}`$:
$`\boldsymbol{c}^H\mathbf{M}\mathbf{M}^{-1}\mathbf{A}\boldsymbol{d} = (\mathbf{M}^{-1}\mathbf{A}^H\boldsymbol{c})^H\mathbf{M}\boldsymbol{d}`$.
For the plain standard operator $`\mathbf{A}`$, the $`\mathbf{M}`$-adjoint is
$`\mathbf{M}^{-1}\mathbf{A}^H\mathbf{M}`$, which is the form given in the
original LaTeX document.)

The *adjoint eigenvalue problem* is

```math
\mathbf{L}^+\,\hat{\boldsymbol{q}}^+ = \sigma^+\,\hat{\boldsymbol{q}}^+, \qquad \sigma^+ = \sigma^*.
\tag{10}
```

Direct and adjoint modes satisfy the *bi-orthogonality condition*

```math
\langle\hat{\boldsymbol{q}}^+_i,\,\hat{\boldsymbol{q}}_j\rangle = (\hat{\boldsymbol{q}}^+_i)^H\mathbf{M}\hat{\boldsymbol{q}}_j = \delta_{ij}.
\tag{11}
```

The adjoint mode identifies the *receptivity* of the flow: the regions and
variables where an external perturbation projects most strongly onto the
instability. In convectively unstable flows the direct mode sits downstream
and the adjoint mode sits upstream, near the source of the instability. This
reflects the non-normality of $`\mathbf{L}`$.

> ⚠️ **Implementation note — what the code computes as "adjoint".**
> `eig_solver.py` does **not** solve (10). It enables SLEPc's two-sided
> Krylov–Schur (`E.setTwoSided(True)`) and stores the **left eigenvectors** of
> the operator passed to SLEPc, i.e. the Euclidean adjoint:
> $`\mathbf{A}^H\boldsymbol{p} = \sigma^*\boldsymbol{p}`$ (for `gen=False`).
> These $`\boldsymbol{p}`$ are written to `eiga_N.pval` **without** the
> transformation $`\hat{\boldsymbol{q}}^+ = \mathbf{M}^{-1}\boldsymbol{p}`$.
> The bi-orthogonality that SLEPc guarantees for these vectors is
> $`\boldsymbol{p}_i^H\hat{\boldsymbol{q}}_j \propto \delta_{ij}`$
> (Euclidean), not (11).

##### Structural sensitivity

Combining the direct and adjoint modes gives the *structural sensitivity*
(Giannetti & Luchini, 2007). It quantifies how far the eigenvalue $`\sigma`$
drifts when a localised feedback is introduced in the flow. The sensitivity
of $`\sigma`$ to a generic perturbation $`\delta\mathbf{A}`$ of the Jacobian is

```math
\delta\sigma = \frac{\langle \hat{\boldsymbol{q}}^+,\,\delta\mathbf{A}\,\hat{\boldsymbol{q}} \rangle}{\langle \hat{\boldsymbol{q}}^+,\,\mathbf{M}\,\hat{\boldsymbol{q}} \rangle}.
\tag{12}
```

For a spatially localised perturbation this leads to the *structural
sensitivity tensor*

```math
\mathbf{S}(\boldsymbol{x}) = \frac{\hat{\boldsymbol{q}}^+(\boldsymbol{x})\otimes\hat{\boldsymbol{q}}(\boldsymbol{x})}{\langle \hat{\boldsymbol{q}}^+,\,\mathbf{M}\,\hat{\boldsymbol{q}} \rangle},
\tag{13}
```

Its Frobenius norm $`\mathcal{S}(\boldsymbol{x}) = \|\mathbf{S}(\boldsymbol{x})\|_F`$
picks out the regions where an internal feedback produces the largest drift
of $`\sigma`$, i.e. the *core of the instability mechanism*. Since
$`\|\boldsymbol{a}\otimes\boldsymbol{b}\|_F = \|\boldsymbol{a}\|\,\|\boldsymbol{b}\|`$,
this is equivalent to
$`\mathcal{S}(\boldsymbol{x}) = \|\hat{\boldsymbol{q}}^+(\boldsymbol{x})\|\,\|\hat{\boldsymbol{q}}(\boldsymbol{x})\| / |\langle \hat{\boldsymbol{q}}^+, \mathbf{M}\hat{\boldsymbol{q}} \rangle|`$,
where the norms are taken over the variables (or only the velocity
components) **of each mesh node**.

> ⚠️ **Implementation note.** The code computes the product **per DOF**
> (`np.abs(adj) * np.abs(dir)`), not per node, so the output contains one
> "sensitivity" field per variable rather than a single scalar field
> $`\mathcal{S}(\boldsymbol{x})`$. See [§6.5](#65-structural-sensitivity).

##### Numerical solution: Krylov–Schur, shift-invert and MUMPS

The full spectrum has size $`N_\mathrm{DOF} = N_v \times N`$, far too large for
direct diagonalisation. Only a few eigenvalues $`N_\mathrm{ev} \ll N_\mathrm{DOF}`$
near a target region of the complex plane are of physical interest. FLORES
combines three strategies to make this tractable.

**Krylov subspace projection.** The Krylov–Schur method (a restarted variant
of Arnoldi) projects the operator onto a subspace of dimension
$`m \ll N_\mathrm{DOF}`$ (`ncv` in the input file). The Rayleigh–Ritz projection
$`\mathbf{H} = \mathbf{V}^H\mathbf{L}\mathbf{V}`$, with
$`\mathbf{V}^H\mathbf{V} = \mathbf{I}`$, gives a small $`m\times m`$ matrix whose
eigenvalues (Ritz values) approximate the wanted eigenvalues. This is
implemented through SLEPc's `EPS` object. Krylov–Schur is the default `EPS`
type and is not set explicitly in `eig_solver.py`.

**Shift-invert spectral transformation.** Plain Arnoldi converges to the
eigenvalues of largest magnitude, which are not the physically relevant ones.
Shift-invert transforms the problem into

```math
(\mathbf{A} - \nu\,\mathbf{B})^{-1}\mathbf{B}\,\hat{\boldsymbol{q}} = \theta\,\hat{\boldsymbol{q}}, \qquad \theta = \frac{1}{\sigma - \nu},
\tag{14}
```

Here $`\nu\in\mathbb{C}`$ is the user-defined shift (`shift_real`,
`shift_imag`), and $`\mathbf{B} = \mathbf{I}`$ when `gen=False` or
$`\mathbf{B} = \mathbf{M}`$ when `gen=True`. The largest $`|\theta|`$ correspond
to the $`\sigma`$ closest to $`\nu`$, and $`\sigma = \nu + 1/\theta`$. In SLEPc this
is the `ST` object of type `sinvert`, combined with
`setWhichEigenpairs(TARGET_MAGNITUDE)` and `setTarget(ν)`.

**LU factorisation with MUMPS.** Each Krylov iteration applies
$`(\mathbf{A} - \nu\mathbf{B})^{-1}`$. FLORES does not use an iterative solver,
which fails on the ill-conditioned compressible operators. Instead it performs
**one** sparse LU factorisation with **MUMPS** (`KSP=preonly`, `PC=lu`). The
factors are reused for every triangular solve during the iteration. MUMPS uses
a fill-reducing ordering (METIS/ParMETIS when available) and runs in parallel
across MPI ranks. In the two-sided solve, the left eigenvectors reuse the same
LU factors through transposed solves, at no extra factorisation cost.

#### 1.2.3 Resolvent Analysis

##### Input–output framework

For *globally stable* or *noise-amplifier* flows, the long-time dynamics are
driven by the forced response rather than by the eigenspectrum. With a
harmonic forcing $`\boldsymbol{f}(t) = \hat{\boldsymbol{f}}\,e^{i\omega t}`$,
the particular solution of (4) is
$`\tilde{\boldsymbol{q}}(t) = \hat{\boldsymbol{q}}\,e^{i\omega t}`$ with

```math
(i\omega\,\mathbf{M} - \mathbf{A})\,\hat{\boldsymbol{q}} = \mathbf{M}\,\mathbf{P}\,\hat{\boldsymbol{f}}.
\tag{15}
```

The *resolvent operator* maps the forcing to the response:

```math
\hat{\boldsymbol{q}} = \mathbf{R}(\omega)\,\hat{\boldsymbol{f}}, \qquad
\mathbf{R}(\omega) = (i\omega\,\mathbf{M} - \mathbf{A})^{-1}\mathbf{M}\,\mathbf{P}.
\tag{16}
```

$`\mathbf{P}\in\mathbb{R}^{N\times N_f}`$ is the **prolongation matrix of the
forcing**. It selects which components of the state vector can be forced.

> ⚠️ **Implementation note — what P does in the code.** In
> `resolvent_solver.py` (`resolvant.pcreate`), $`\mathbf{P}`$ maps the forcing
> to the **two momentum components** of each node: DOFs `i+1` and `i+2`,
> i.e. $`u`$ and $`w`$ in 2-D. It does **not** exclude sponge layers. A spatial
> restriction is achieved only through domain reduction
> ([Chapter 5](#5-domain-reduction)). The forcing space has dimension
> $`N_f = 2N_\mathrm{DOF}/n_{eq}`$.

The response energy is measured with a diagonal weight matrix $`\mathbf{Q}`$.
A common choice for compressible flows is the Chu energy norm:

```math
E = \int_\Omega \left( \frac{\bar{p}}{\bar{\rho}\,\bar{T}}\,|\hat{\rho}|^2 + \bar{\rho}\,|\hat{\boldsymbol{u}}|^2 + \frac{\bar{\rho}\,c_v}{\bar{T}}\,|\hat{T}|^2 \right)\mathrm{d}\Omega.
\tag{17}
```

> ⚠️ **Implementation note — the Chu norm is not implemented.** In the code,
> $`\mathbf{Q} = \mathbf{M}`$ (cell volumes, identical for every variable).
> The gain is therefore the ratio of the *volume-weighted Euclidean norms* of
> response and forcing in conservative variables, not the Chu energy.

##### Optimal gains and modes via SVD

The (weighted) singular value decomposition of the resolvent,
$`\mathbf{R} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^*`$, gives:

- the **optimal response modes** (columns of $`\mathbf{U}`$),
- the **optimal forcing modes** (columns of $`\mathbf{V}`$),
- the **optimal gains** $`\lambda_i^2 = \sigma_i^2`$.

The *gain curve* $`\lambda_1^2(\omega)`$ is the maximum energy amplification as
a function of frequency.

##### Reformulation as an eigenvalue problem

Computing the SVD directly is not affordable for large $`N`$. FLORES solves the
equivalent eigenvalue problem

```math
\mathbf{R}^{\dagger}\mathbf{R}\,\hat{\boldsymbol{f}}_i = \lambda_i^2\,\hat{\boldsymbol{f}}_i,
\tag{18}
```

where $`\mathbf{R}^\dagger`$ is the adjoint with respect to the chosen
forcing/response norms. The operator is never assembled. It is applied
matrix-free as a PETSc *shell matrix*, using **two triangular solves per
application** (one `solve` and one `solveTranspose`) that reuse a single LU
factorisation of $`L(\omega)`$ computed at the start of each frequency.

> ⚠️ **Implementation note — operator actually implemented.** Each Krylov
> iteration of `resolvent_solver.py` applies
>
> ```math
> \mathbf{D} = \mathbf{P}^T\mathbf{M}^{-1}\,\mathbf{L}^{-H}\,\mathbf{Q}\,\mathbf{L}^{-1}\,\mathbf{P}, \qquad \mathbf{L} = i\omega\mathbf{I} - \mathbf{A},\quad \mathbf{Q}=\mathbf{M}.
> ```
>
> Two differences from (15)–(16):
>
> 1. $`L`$ is built with the **identity**, not with $`\mathbf{M}`$
>    (`J.scale(-1); J.shift(iω)`).
> 2. The forcing on the right-hand side is $`\mathbf{P}\hat{\boldsymbol{f}}`$,
>    not $`\mathbf{M}\mathbf{P}\hat{\boldsymbol{f}}`$.
>
> With TAU `Jacobian volume scaling = 1` ($`\mathbf{A} = \mathbf{M}^{-1}\mathbf{J}`$),
> multiplying (15) by $`\mathbf{M}^{-1}`$ gives
> $`(i\omega\mathbf{I} - \mathbf{A})\hat{\boldsymbol{q}} = \mathbf{P}\hat{\boldsymbol{f}}`$,
> which is exactly what the code solves. With `Jacobian volume scaling = 0`
> the resolvent is **wrong**, because the solver has no `gen`-type option. Use
> Jacobians generated with `1`, or scale the rows of $`\mathbf{A}`$ by
> $`\mathbf{M}^{-1}`$ before building $`L`$.
>
> SLEPc treats $`\mathbf{D}`$ as a non-Hermitian problem (`NHEP`). The weight
> $`\mathbf{P}^T\mathbf{M}^{-1}`$ makes $`\mathbf{D}`$ self-adjoint only in the
> $`\mathbf{P}^T\mathbf{M}\mathbf{P}`$ inner product, not in the Euclidean one.
> The gains are taken as $`\mathrm{Re}(\lambda)`$.

##### Adjoint resolvent and receptivity

The adjoint resolvent $`\mathbf{R}^\dagger(\omega)`$ has the same singular
values as $`\mathbf{R}(\omega)`$. Its singular vectors describe the
*receptivity* of the flow to harmonic forcing at frequency $`\omega`$. This
information complements the optimal forcing modes and is physically distinct
from the structural sensitivity (13) of the eigenvalue problem.

---

## 2. Software Architecture

### 2.1 Dependencies

| Package | Purpose | Version (CESVIMA scripts) |
|---|---|---|
| Python | Host language | 3.6+ |
| mpi4py | MPI bindings | built from source (`--no-binary`) |
| PETSc | Parallel sparse linear algebra, **complex scalars** (`--with-scalar-type=complex`) | `release` branch (3.25 at time of writing) |
| SLEPc | Eigenvalue solvers | `release` branch (3.25) |
| petsc4py | Python bindings for PETSc | 3.25.0 (pinned); also tested with 3.26 |
| slepc4py | Python bindings for SLEPc | 3.25.0 (pinned) |
| MUMPS | Sparse direct solver (LU) | 5.4.0 (module) or 5.5.1 (built by `_ompi` script) |
| NumPy | Numerical arrays | any |
| SciPy | Sparse matrices (CSR) on rank 0 | any |
| netCDF4 | Reads TAU Jacobian, writes `.pval` | any |
| matplotlib | Post-processing (`tools/`) | any |
| OpenMPI | MPI implementation (cluster) | 4.1.1 |

> ⚠️ PETSc **must** be built with complex scalars. Both solvers handle
> complex shifts and frequencies and use `PETSc.ScalarType` as complex.

### 2.2 Repository structure (branch `miguel_dev`)

```
FLORES/
├── solver/
│   ├── eig_solver.py          # Global eigenvalue solver (direct, adjoint, sensitivity)
│   ├── resolvent_solver.py    # Resolvent solver (matrix-free shell, direct + adjoint)
│   ├── jac_red.py             # Domain-reduction utilities (class domain_reduction)
│   ├── input_output.py        # Readers: TAU Jacobian (NetCDF), coordinates (.coo)
│   ├── mpi_utils.py           # Row distribution of the Jacobian and PETSc assembly
│   └── save2pval.py           # Writers: modes → TAU .pval (2-D and 3-D expansion)
├── tools/
│   ├── plot_eigenvalues.py        # Spectrum in the complex plane
│   ├── plot_gain.py               # Resolvent gain curve λ²(ω)
│   └── plot_pval_eigfunction.py   # 2-D mode / sensitivity visualisation
├── test_cases/
│   ├── BFS/                       # Backward-facing step: .floresparam + SLURM scripts
│   └── Cylinder_Re45/             # Cylinder wake, Re = 45: eigenvalue .floresparam
├── python_env_installation/
│   ├── Cesvima_UPM_Installation.sh       # CESVIMA, MUMPS from module
│   ├── Cesvima_UPM_Installation_ompi.sh  # CESVIMA, MUMPS 5.5.1 built with OpenMP
│   └── Ubuntu_Installation.sh            # Local workstation, everything downloaded
├── doc/
│   └── FLORES_Technical_Documentation.md
├── README.md
└── LICENSE                                # MIT
```

The original LaTeX document referred to the old scripts `EIGENSOLVER.py`,
`RESOLVANT.py`, `eig_simple.py` and to a `post_processing/` folder. They are
now `solver/eig_solver.py`, `solver/resolvent_solver.py` and `tools/`.

**Input files** (in `input_path`):

| File | Content |
|---|---|
| `samg.matrix.amg.pval` | Jacobian in CSR form (NetCDF: `row_ptr`, `col_ind`, `data`, dims `nnz`, `nvars`, `neq`) |
| `samg.matrix.vol` | Cell volumes, one per line (one per node) |
| `samg.matrix.coo` | Header `ndof ndim`, then one `x z` row per DOF (`neq` identical rows per node) |

**Control files** are INI files with the extension `.floresparam`, passed as
the first command-line argument.

### 2.3 Main scripts

#### 2.3.1 `solver/eig_solver.py` — eigenvalue solver

Supports three modes, controlled by the control file:

- **Direct problem (`adjoint = False`).** Computes the right eigenpairs
  $`(\sigma_k, \hat{\boldsymbol{q}}_k)`$. Eigenvectors are written to
  `eigf_N.pval` and eigenvalues are appended to `eigv_DIR.dat`.
- **Direct + adjoint (`adjoint = True`).** Enables SLEPc's two-sided solver
  (`E.setTwoSided(True)`), which computes right and left eigenvectors in the
  same Krylov–Schur iteration with a single LU factorisation. The left
  eigenvectors satisfy $`\mathbf{A}^H\boldsymbol{p}_k = \sigma_k^*\boldsymbol{p}_k`$
  and are written to `eiga_N.pval`, with their eigenvalues ($`\sigma_k^*`$) in
  `eigv_ADJ.dat`.
- **Structural sensitivity (`sensitivity = True`).** Forces `adjoint = True`
  and calls `compute_structural_sensitivity()`, which writes
  `sensitivity_N.pval` for each pair of new modes.

Shared features:

- configuration through a `.floresparam`/`.ini` file;
- **checkpoint/resume** with duplicate detection
  ([§6.6](#66-checkpoint-and-resume));
- consecutive numbering of eigenvector files across runs;
- the Jacobian is read on rank 0 and each rank receives only its own rows
  (`mpi_utils.assemble_aij`, [§6.2](#62-matrix-assembly));
- per-phase timing with `_t()`;
- optional domain reduction ([Chapter 5](#5-domain-reduction));
- for `beta != 0`, an additional 3-D expanded file (`*3D.pval`) with
  `[physics] output_slices` spanwise slices (default 21).

#### 2.3.2 `solver/resolvent_solver.py` — resolvent solver

Implements the resolvent as a **matrix-free shell operator** in PETSc/SLEPc:

- **Direct resolvent (`adjoint = False`).** For each $`\omega`$: one MUMPS LU
  factorisation of $`L(\omega)`$, then Krylov–Schur on the shell
  $`\mathbf{D}`$. Writes the optimal forcing `eigf_i_<omega>.pval`, the optimal
  response `eigr_i_<omega>.pval` and the eigenvalues `eigv_DIR_<omega>.dat`.
- **Adjoint resolvent (`adjoint = True`).** A second shell
  (`resolvant_adjoint`) **reuses the same KSP/LU** and swaps the order of
  `solve` and `solveTranspose`. Writes `eiga_i_<omega>.pval` and
  `eigv_ADJ_<omega>.dat`.
- **Sensitivity (`compute_sensitivity = True`).** Forces `adjoint = True` and
  calls `compute_sensitivity_field()` for each pair, writing
  `sensitivity_i_<omega>.pval`.

`<omega>` is the Python representation of the complex frequency, e.g.
`eigf_0_1j.pval`, `eigv_DIR_1.5j.dat`.

#### 2.3.3 Auxiliary modules

| Module | Main functions |
|---|---|
| `input_output.py` | `openjacobian()` (TAU Jacobian → CSR + `neq`), `read_coordinates()` (vectorised `.coo` reader; detects `neq` from repeated rows; duplicates rows when `beta != 0`). Also contains readers for TAU modes, residuals, base flows and sensitivities (`openegvec`, `openresidual`, `openbflow`, `opensensitivity`, `opendualgrid`, `openqe`) that the solvers do not use. |
| `mpi_utils.py` | `ownership_ranges()` (PETSc default row split), `scatter_csr_rows()` (rank 0 sends each rank its rows), `assemble_aij()` (distributed AIJ matrix with CSR preallocation), `assemble_diag()` (diagonal matrices with the same layout). |
| `jac_red.py` | Class `domain_reduction`: `create_Pmatrix(coords)`, `reduce_matrix(A)`, `reduce_vector(v)`. |
| `save2pval.py` | `mode2pval()` (complex mode → `.pval` with variables `rho, u, w, e[, turb1, turb2]` and their `_i` imaginary parts), `mode2pval3D()` (spanwise expansion), and functions not used by the solvers, `sol2pval()` and `sens2pval()`. |

> `mode2pval()` writes `global_id = 0 … 2·gridpoints−1`, not the TAU
> `global_id`. The Jacobian file does not contain the TAU ids, so this assumes
> the TAU point ordering is the natural one.

---

## 3. Installation

Automated scripts are provided in `python_env_installation/`. They:

- create a Python virtual environment;
- build PETSc and SLEPc **in place** (no `make install`) from the `release`
  branch on GitLab;
- install `mpi4py`, `petsc4py` and `slepc4py` from source, plus the extra
  dependencies;
- patch the venv's `activate` script with `PETSC_DIR`, `SLEPC_DIR`,
  `PETSC_ARCH` and `LD_LIBRARY_PATH`;
- use checkpoints, so they can be re-run safely after an interruption.

| Script | Platform | MPI / BLAS / MUMPS | venv |
|---|---|---|---|
| `Cesvima_UPM_Installation.sh` | CESVIMA (UPM) | modules `foss/2021a` + `MUMPS/5.4.0-foss-2021a-metis` | `flores_env` |
| `Cesvima_UPM_Installation_ompi.sh` | CESVIMA (UPM) | GCC 10.3 / OpenMPI 4.1.1 / OpenBLAS / ScaLAPACK / METIS / SCOTCH modules; **builds MUMPS 5.5.1 with OpenMP + PT-Scotch** | `myvenv_ompi` |
| `Ubuntu_Installation.sh` | Ubuntu workstation | everything downloaded by PETSc (`--download-mpich`, `--download-f2cblaslapack`, `--download-scalapack`, `--download-mumps`, …) | `myvenv` |

### 3.1 CESVIMA cluster environment

Production runs use the **CESVIMA** HPC cluster at UPM (SLURM, BeeGFS).
Module stack for the standard script:

```bash
module purge
module load foss/2021a
module load MUMPS/5.4.0-foss-2021a-metis
```

This provides GCC 10.3.0, OpenMPI 4.1.1, OpenBLAS 0.3.15, ScaLAPACK 2.1.0 and
MUMPS 5.4.0.

> **Important:** run the installation as a SLURM job, not on the login node.
> Compiling PETSc/SLEPc and `petsc4py`/`slepc4py` uses a lot of memory and the
> login node's OOM policy will kill it.

### 3.2 PETSc/SLEPc build

Key PETSc configuration flags (CESVIMA, standard script):

```bash
./configure \
  --with-debugging=0 \
  --with-shared-libraries=1 \
  --with-mpi=1 \
  --with-scalar-type=complex \
  --with-mpi-dir="$EBROOTOPENMPI" \
  --with-blas-lapack-dir="$EBROOTOPENBLAS" \
  --with-scalapack-dir="$EBROOTSCALAPACK" \
  --with-mumps-dir="$EBROOTMUMPS" \
  --download-cmake \
  --download-metis \
  --download-parmetis
```

**Caveats found during installation:**

- `libgfortran.so.4` lives in `/media/apps/avx512/software/GCCcore/7.2.0/lib64`.
  Append it to `LD_LIBRARY_PATH`; do not prepend it, or newer tools such as
  `cmake` break.
- Build `petsc4py` and `slepc4py` with `CFLAGS="-O0 -g0" MAX_JOBS=1` to avoid
  OOM kills.
- Version pins: `petsc4py==3.25.0` and `slepc4py==3.25.0` must match the
  compiled PETSc/SLEPc versions exactly.
- The checkpoint guards use `ls lib/libpetsc.so.* 1>/dev/null 2>&1` instead of
  `[ -z "$(...)" ]`, which fails under `set -e` with globs.
- If the compute nodes have no internet access, the `--download-*` packages
  must be fetched in advance (the script prints instructions).

### 3.3 Environment activation

After installation:

```bash
source flores_env/bin/activate      # or myvenv / myvenv_ompi depending on the script
```

The SLURM scripts in `test_cases/BFS/` source a site-specific
`load_env_STAB_tool_ompi.sh`, which is not included in the repository.
Example:

```bash
#!/bin/bash
module purge
module load foss/2021a
module load MUMPS/5.4.0-foss-2021a-metis
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/media/apps/avx512/software/GCCcore/7.2.0/lib64
source /path/to/flores_env/bin/activate   # activate already exports PETSC_DIR/SLEPC_DIR/PETSC_ARCH
```

---

## 4. Test Cases

### 4.1 Backward-facing step (`test_cases/BFS`)

Contents of the folder:

| File | Purpose |
|---|---|
| `BFS_eig.floresparam` | Eigenvalue run: M = 0.1, `nev = 1`, `ncv = 20`, shift $`-0.05 + 0.7226i`$, adjoint + sensitivity, domain reduction $`x\in(-10,30)`$, $`z\in(-10,10)`$ |
| `BFS_resolvent.floresparam` | Resolvent run: M = 0.1, $`\omega = 1`$, `nev = 2`, `ncv = 100`, domain reduction $`x\in(-15,45)`$, $`z\in(-15,15)`$ |
| `script_eig.slurm` | `mpirun -n $SLURM_NTASKS python3 $SOLVER_PATH/eig_solver.py BFS_eig.floresparam` |
| `script_resolvent.slurm` | same, with `resolvent_solver.py` |
| `script_plot_eigf.slurm` | Runs `plot_pval_eigfunction.py` |

The scripts use `${FLORES_DIR}` (path to the repository) and source a
site-specific `load_env_*.sh`, which must be provided by the user
([§3.3](#33-environment-activation)).

`test_cases/Cylinder_Re45/Cylinder_eig.floresparam` contains an eigenvalue
set-up for the cylinder wake at Re = 45.

The original documentation describes the following TAU set-ups. The
Jacobians and meshes are **not** in the repository.

#### Laminar validation (`BFS_M0p1_Re500`)

| Parameter | Value |
|---|---|
| Mach number | ≈ 0.1 |
| Reynolds number | 500 (based on step height $`h = 1`$ m) |
| Reference velocity | 106.53 m/s |
| Static temperature | 293.91 K |
| Static pressure | 93 591 Pa |
| Turbulence model | None (fully laminar) |
| MPI domains (TAU) | 40 |
| 2-D extrusion axis | $`Z`$ (offset = 3) |

TAU set-up:

- inlet: reservoir-pressure inflow, $`\rho_0 = 1.1636`$ kg/m³, $`p_0 = 100\,039.56`$ Pa;
- outlet: exit pressure, $`p = 93\,591`$ Pa;
- walls: viscous, laminar, with force monitoring;
- inviscid flux: upwind Roe, 2nd order;
- time integration: backward Euler, implicit, LU-SGS (3 sweeps), CFL = 5;
- convergence: up to 300 000 iterations, residual $`< 10^{-7}`$.

#### Turbulent production case (`M0p31_extIO2_mesh4_TI0p4_SA_central`)

| Parameter | Value |
|---|---|
| Mach number | 0.31 |
| Reference velocity | 106.53 m/s |
| Static temperature | 293.91 K |
| Static pressure | 93 591 Pa |
| Step height | $`h = 0.036093`$ m |
| Turbulence model | Spalart–Allmaras (SA-neg) |
| Inlet TI | 0.4 % |
| Inviscid flux | Central + scalar dissipation |
| Linear solver | LU-SGS, 3 sweeps |
| CFL | 2 |
| MPI domains (TAU) | 12 |
| 2-D extrusion axis | $`Y`$ (offset = 2) |

---

## 5. Domain Reduction

### 5.1 Motivation

The LU factorisation of the Jacobian $`\mathbf{A}`$ is the computational
bottleneck in both memory and CPU time. This becomes prohibitive at high
Reynolds number, in 3-D, or with turbulence models. *Domain reduction* (DR)
restricts the problem to a geometric subdomain that contains the physically
relevant region.

Because the Jacobian is built with a discrete stencil, each node only couples
to itself and to its first and second neighbours. This local connectivity
allows the stability of a single region to be analysed.

### 5.2 Mathematical foundation (Martínez-Cava, 2019, §3.2.1)

Let $`\Omega_m\subset\Omega_n`$ be a subdomain with $`m<n`$ DOFs, with its
unknowns ordered first:

```math
\boldsymbol{q}_n = \{q_{\{1\cdots m\}}\ |\ q_{\{(m+1)\cdots n\}}\}^T.
```

The projection operator $`\mathbf{P}_{DR}\in\mathbb{R}^{n\times m}`$ is

```math
\mathbf{P}_{DR} = \begin{pmatrix}\mathbf{I}_m\\ \mathbf{0}_{(n-m)\times m}\end{pmatrix},
\qquad \mathbf{P}_{DR}^T\mathbf{P}_{DR} = \mathbf{I}_m,
\tag{19}
```

and the reduced matrices are

```math
\mathbf{A}_m = \mathbf{P}_{DR}^T \mathbf{A}_n \mathbf{P}_{DR},\qquad
\mathbf{M}_m = \mathbf{P}_{DR}^T \mathbf{M}_n \mathbf{P}_{DR},\qquad
\mathbf{Q}_m = \mathbf{P}_{DR}^T \mathbf{Q}_n \mathbf{P}_{DR}.
\tag{20}
```

(The subscript $`DR`$ distinguishes this operator from the forcing prolongation
$`\mathbf{P}`$ of the resolvent.)

The reduced problem is $`\mathbf{A}_m\hat{\boldsymbol{q}}_m = \sigma\mathbf{M}_m\hat{\boldsymbol{q}}_m`$,
with $`m \ll n`$. Sanvido et al. showed that DR recovers the disturbances most
relevant to the region of interest and filters out part of the irrelevant
spectrum. The approximation is valid as long as $`\Omega_m`$ contains the
structural sensitivity region of the dominant modes (Giannetti & Luchini,
2007).

Although DR is described as imposing no explicit boundary conditions, it is
**equivalent to a homogeneous Dirichlet condition** on the perturbation
outside $`\Omega_m`$. The product $`\mathbf{P}_{DR}^T\mathbf{A}_n\mathbf{P}_{DR}`$
keeps only the interior block $`\mathbf{A}_{mm}`$ and drops the coupling with
exterior nodes. That is exactly what you get by setting the perturbation to
zero on the exterior nodes. The approximation therefore holds when the modes
of interest are naturally close to zero at the cut boundary.

### 5.3 Implementation in FLORES

DR is enabled with `[domain_reduction] enabled = True`. The box is given by
`xmin`, `xmax`, `zmin`, `zmax`. At runtime:

1. **Rank 0** reads `samg.matrix.coo` (`read_coordinates`, scaled by
   `rlength`) and builds the mask of DOFs strictly inside the box
   (`x > xmin & x < xmax & z > zmin & z < zmax`). Column 0 of the file is
   $`x`$ and column 1 is $`z`$.
2. $`\mathbf{P}_{DR}`$ is **not** built explicitly. `domain_reduction` stores
   the index array `kept_idx`, and `reduce_matrix` extracts the submatrix by
   CSR row slicing followed by CSC column slicing. This is equivalent to (20)
   but much cheaper than two sparse products.
3. Rank 0 reduces $`\mathbf{A}`$ and $`\mathbf{M}`$ (plus $`\mathbf{Q}`$ and
   $`\mathbf{M}^{-1}`$ in the resolvent). It also builds `rgid`, the indices of
   the nodes kept, which are used to scatter the reduced modes back onto the
   full mesh when writing `.pval` files. Everything is broadcast with
   `comm.Bcast`.
4. PETSc assembly, LU factorisation and the eigenvalue/resolvent solves all
   run on the reduced system.

Typical reduction for the BFS cases: about 10×–50× fewer non-zeros, with a
proportional reduction in memory and factorisation time.

```ini
[domain_reduction]
enabled = True
xmin    = -2.0
xmax    =  20.0
zmin    = -5.0
zmax    =  5.0
```

> ⚠️ The selection is made **per DOF** using the coordinates of each row of
> the `.coo` file. Since all DOFs of a node share coordinates, whole nodes
> are kept, and `rgid` is computed as `reduce_vector(localid)[0::neq]`.

---

## 6. Eigenvalue Solver

### 6.1 Control file

Running:

```bash
mpirun -n <N> python3 solver/eig_solver.py case_eig.floresparam
```

Complete example with all keys (defaults in comments):

```ini
[io]
input_path   = JAC/
output_path  = RESULTS_eig/          # created if it does not exist
jac_file     = samg.matrix.amg.pval  # default
vol_file     = samg.matrix.vol       # default
coord_file   = samg.matrix.coo       # default

[physics]
mach    = 0.31
beta    = 0.0        # spanwise wavenumber (0 = 2-D)
rlength = 1.0        # coordinate scale (domain reduction only)
output_slices = 21   # spanwise slices of the *3D.pval output (beta != 0)

[solver]
nev          = 20
ncv          = 63          # 0 or absent -> 3*nev + 1
shift_real   = 0.0         # alternatively: shift = (0+10j)
shift_imag   = 10.0
tol          = 1e-8        # default
max_it       = 15000       # default
adjoint      = True        # default False
gen          = False       # default False: standard EVP A x = σ x  (TAU Jacobian volume scaling = 1)
                           # True: A x = σ M x                      (TAU Jacobian volume scaling = 0)
sensitivity  = True        # default False; forces adjoint = True

[domain_reduction]
enabled = True             # default False
xmin    = -2.0
xmax    =  20.0
zmin    = -5.0
zmax    =  5.0

[checkpoint]
dup_tol_real = 1e-5        # default
dup_tol_imag = 1e-5        # default
```

`mpd` (maximum projected dimension) is set to `ncv - 1`. Any PETSc/SLEPc
option can be passed on the command line (`-eps_*`, `-st_*`,
`-mat_mumps_icntl_*`), because every object calls `setFromOptions()`.

### 6.2 Matrix assembly

#### Jacobian $`\mathbf{A}`$

Only **rank 0** reads the Jacobian, scales it by $`1/(M_\infty\sqrt{1.4})`$
and, if enabled, applies the domain reduction. It then sends each rank only
the block of rows that rank owns (`mpi_utils.assemble_aij`):

```python
# mpi_utils.py (simplified)
ranges = ownership_ranges(n, nproc)          # PETSc default row split
# rank 0: for each rank r, slice rows ranges[r] and comm.Send them
# rank r: comm.Recv its local indptr (0-based), indices, data
A = PETSc.Mat().createAIJ(size=((nloc, n), (nloc, n)),
                          csr=(indptr, indices, data),
                          comm=PETSc.COMM_WORLD)
A.assemble()
```

`createAIJ(csr=...)` preallocates and inserts the local rows in one call
(`MatMPIAIJSetPreallocationCSR`). **Key detail:** `indptr` must be shifted to
local 0-based offsets. Ranks other than 0 never hold more than their own rows,
so per-rank memory decreases with the number of ranks (rank 0 still needs the
full matrix while reading).

#### Mass matrix $`\mathbf{B}`$

$`\mathbf{B}`$ is diagonal. It is built from `samg.matrix.vol`, one volume per
node, repeated `neq` times:

```python
B = assemble_diag(b_data, n)   # same row layout as A
```

$`\mathbf{B}`$ enters the eigenvalue problem only when `gen = True`. It is
always used for the normalisation in the sensitivity computation.

### 6.3 Direct problem

```python
E = SLEPc.EPS().create()
if gen:
    E.setOperators(A, B)                         # A x = σ B x
else:
    E.setOperators(A)                            # A x = σ x
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
if adjoint:
    E.setTwoSided(True)

ST = E.getST(); ST.setType('sinvert'); ST.setShift(shift)
K  = ST.getKSP(); K.setType('preonly')
pc = K.getPC();   pc.setType('lu'); pc.setFactorSolverType('mumps')

E.setTolerances(tol=tol, max_it=max_it)
E.setDimensions(nev, ncv, ncv - 1)
E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
E.setTarget(shift)
E.setFromOptions()
E.solve()
```

The new right eigenvectors are written to `eigf_N.pval` and their eigenvalues
are appended to `eigv_DIR.dat` (format: `index  Re  Im`).

### 6.4 Adjoint problem (two-sided solve)

With `adjoint = True`, two-sided Krylov–Schur computes right and left
eigenvectors in the same iteration, reusing the LU factorisation of
$`(\mathbf{A}-\nu\mathbf{B})`$. The left eigenvectors satisfy

```math
\mathbf{A}^H\boldsymbol{p}_k = \sigma_k^*\,\boldsymbol{p}_k \quad(\texttt{gen=False}),
\qquad
\mathbf{A}^H\boldsymbol{p}_k = \sigma_k^*\,\mathbf{B}^H\boldsymbol{p}_k \quad(\texttt{gen=True}).
```

They are retrieved with `E.getLeftEigenvector(i, yr, yi)` and written to
`eiga_N.pval`. The value written to `eigv_ADJ.dat` is $`\sigma_k^*`$.

Relation to the adjoint in the $`\mathbf{M}`$ inner product (8), for the two
consistent configurations:

- **TAU scaling `0` + `gen = True`** ($`\mathbf{A}=\mathbf{J}`$,
  $`\mathbf{J}\boldsymbol{q} = \sigma\mathbf{M}\boldsymbol{q}`$). The left
  eigenvector *is* the $`\mathbf{M}`$-adjoint mode:
  $`\mathbf{M}^{-1}\mathbf{J}^H\boldsymbol{p} = \sigma^*\boldsymbol{p}`$, so
  $`\hat{\boldsymbol{q}}^+ = \boldsymbol{p}`$. Bi-orthogonality:
  $`\boldsymbol{p}^H\mathbf{M}\hat{\boldsymbol{q}}`$.
- **TAU scaling `1` + `gen = False`** ($`\mathbf{A}=\mathbf{M}^{-1}\mathbf{J}`$,
  $`\mathbf{L}=\mathbf{A}`$). The $`\mathbf{M}`$-adjoint is
  $`\mathbf{L}^+ = \mathbf{M}^{-1}\mathbf{A}^H\mathbf{M}`$, and its eigenvector
  is $`\hat{\boldsymbol{q}}^+ = \mathbf{M}^{-1}\boldsymbol{p}`$. Bi-orthogonality:
  $`\boldsymbol{p}^H\hat{\boldsymbol{q}} = \langle\hat{\boldsymbol{q}}^+,\hat{\boldsymbol{q}}\rangle_\mathbf{M}`$.
  **The code writes $`\boldsymbol{p}`$ to `eiga_N.pval` without applying
  $`\mathbf{M}^{-1}`$.** The spatial distribution therefore differs from the
  $`\mathbf{M}`$-adjoint by a factor $`1/V_\mathrm{cell}`$, which matters on
  stretched meshes.

### 6.5 Structural sensitivity

With `sensitivity = True`, `compute_structural_sensitivity()` runs for each
pair (new direct mode $`i`$, new adjoint mode $`i`$). The vectors are gathered on
rank 0 and the code computes:

```python
Bq_dir      = B * q_dir
inner_prod  = np.dot(np.conj(adj_arr), Bqd_arr)     # <q+, B q>
norm_ip     = abs(inner_prod)                       # 1.0 if < 1e-30
sensitivity = np.abs(adj_arr) * np.abs(dir_arr) / norm_ip
```

which corresponds to, **per DOF** $`j`$:

```math
\mathcal{S}_j = \frac{|\hat q^+_j|\,|\hat q_j|}{|\langle\hat{\boldsymbol{q}}^+,\mathbf{B}\hat{\boldsymbol{q}}\rangle|}.
\tag{21}
```

The result is written to `sensitivity_N.pval` through `mode2pval`, so each
variable `rho`, `u`, `w`, `e` in the file holds the per-DOF sensitivity of
that variable. It can be plotted with `tools/plot_pval_eigfunction.py`.

> ⚠️ **Implementation notes:**
> 1. (21) is not the Frobenius norm of the tensor (13). For that, compute
>    per node $`\|\hat{\boldsymbol{q}}^+(\boldsymbol{x})\|\,\|\hat{\boldsymbol{q}}(\boldsymbol{x})\|`$
>    (usually velocity components only).
> 2. The normalisation always uses $`\boldsymbol{p}^H\mathbf{B}\hat{\boldsymbol{q}}`$
>    (with $`\mathbf{B}=\mathbf{M}`$).
>    - **TAU scaling `0` + `gen = True`:** correct.
>    - **TAU scaling `1` + `gen = False`:** the consistent normalisation is
>      $`\boldsymbol{p}^H\hat{\boldsymbol{q}}`$, and the numerator should use
>      $`\hat{\boldsymbol{q}}^+ = \mathbf{M}^{-1}\boldsymbol{p}`$.
>
>    Because both errors are a constant factor in the denominator, the
>    *shape* of the field is affected only by the $`\mathbf{M}^{-1}`$ in the
>    numerator. The absolute value is affected by both.
> 3. Direct and adjoint modes are paired by their position in the lists of
>    *new* modes. Because duplicates are filtered separately for the direct
>    (`eigv_DIR.dat`) and adjoint (`eigv_ADJ.dat`) lists, the pairs can end up
>    misaligned on a resumed run. It would be safer to pair by index $`i`$ of the
>    same EPS before filtering.

### 6.6 Checkpoint and resume

The checkpoint avoids recomputing across successive SLURM jobs:

1. On start-up, rank 0 reads `eigv_DIR.dat` and `eigv_ADJ.dat` (if present)
   and broadcasts them.
2. Each converged eigenvalue is compared against that set: it is a duplicate
   if $`|\Delta\mathrm{Re}| <`$ `dup_tol_real` **and** $`|\Delta\mathrm{Im}| <`$
   `dup_tol_imag`.
3. Each eigenvalue is labelled `NEW` or `SKIP (duplicate)`.
4. Only `NEW` eigenvalues are appended, and their vectors are saved with
   indices that continue from the highest existing `eigf_N.pval` /
   `eiga_N.pval`.

The output directory is created by rank 0 only, followed by a barrier.

---

## 7. Resolvent Analysis

### 7.1 Control file

```bash
mpirun -n <N> python3 solver/resolvent_solver.py case_resolvent.floresparam
```

```ini
[io]
input_path   = JAC/
output_path  = RESULTS_resolvent/
jac_file     = samg.matrix.amg.pval  # default
vol_file     = samg.matrix.vol       # default
coord_file   = samg.matrix.coo       # default

[physics]
mach    = 0.31
beta    = 0.0
nslices = 7            # slices in the TAU Jacobian, only if beta != 0 (default 7)
slice_spacing = 1.0    # spanwise distance between slices dy, only if beta != 0 (default 1.0)
rlength = 1.0

[frequencies]
omega_start = 10.0   # imaginary part of iω
omega_end   = 150.0
omega_n     = 50     # np.linspace(start, end, n)

[solver]
nev                  = 5
ncv                  = 20       # required (no default)
shift                = 0.0      # real ST shift; 0 = none
tol                  = 1e-6     # default
max_it               = 1000     # default
adjoint              = True
compute_sensitivity  = True     # forces adjoint = True

[domain_reduction]
enabled = True
xmin    = -2.0
xmax    =  20.0
zmin    = -5.0
zmax    =  5.0
```

`ncv` is the dimension of the Krylov subspace and must be larger than `nev`
(SLEPc recommends `ncv` ≥ 2·`nev`).

### 7.2 Direct resolvent: matrix-free shell operator

The class `resolvant` defines the shell. `operator(omega)` transforms the copy
of the Jacobian in place, $`\mathbf{J}\leftarrow -\mathbf{J} + i\omega\mathbf{I}`$,
i.e. $`L = i\omega\mathbf{I} - \mathbf{A}`$, and factorises it once with MUMPS.
Each call to `mult` computes $`\boldsymbol{y} = \mathbf{D}\boldsymbol{f}`$ in
six steps, of which **two** are linear solves:

1. $`\boldsymbol{u}_1 = \mathbf{P}\boldsymbol{f}`$ (prolongation to the $`u,w`$ components)
2. solve $`L\,\boldsymbol{u}_2 = \boldsymbol{u}_1`$ (`ksp.solve`)
3. $`\boldsymbol{u}_3 = \mathbf{Q}\,\boldsymbol{u}_2`$, with $`\mathbf{Q} = \mathbf{M}`$
4. conjugate, solve $`L^T\boldsymbol{u}_4 = \overline{\boldsymbol{u}_3}`$
   (`ksp.solveTranspose`), conjugate again ⇒ $`\boldsymbol{u}_4 = L^{-H}\boldsymbol{u}_3`$
5. $`\boldsymbol{u}_5 = \mathbf{M}^{-1}\boldsymbol{u}_4`$
6. $`\boldsymbol{y} = \mathbf{P}^T\boldsymbol{u}_5`$

SLEPc solves $`\mathbf{D}\hat{\boldsymbol{f}} = \lambda^2\hat{\boldsymbol{f}}`$
with Krylov–Schur (`NHEP`). For each converged pair $`i`$ (at most `nev`):

- gain $`\lambda_i^2 = \mathrm{Re}(\text{eigenvalue}_i)`$, written to
  `eigv_DIR_<omega>.dat` (`index Re Im`);
- **optimal forcing** $`\mathbf{P}\hat{\boldsymbol{f}}_i`$ → `eigf_i_<omega>.pval`;
- **optimal response** $`L^{-1}\mathbf{P}\hat{\boldsymbol{f}}_i`$ → `eigr_i_<omega>.pval`.

The shell matrices have the size of the forcing space,
$`N_f = 2\,n/n_{eq}`$ (the number of columns of $`\mathbf{P}`$), so any `neq`
(laminar 4, RANS 5–6) is supported.

> ⚠️ **Implementation notes:**
> 1. $`\mathbf{P}`$ takes DOFs 1 and 2 of each node. With `beta ≠ 0` (order
>    `rho, u, v, w, e`) these are $`u, v`$, not the in-plane components.
> 2. The `shift` parameter only sets the ST shift (type `shift` by default),
>    not a shift-invert.

### 7.3 Adjoint resolvent

With `adjoint = True`, a second shell is created from `resolvant_adjoint`. It
**reuses the same KSP (same LU factors)** and swaps the order of `solve` and
`solveTranspose`:

1. $`\boldsymbol{u}_1 = \mathbf{P}\boldsymbol{f}`$
2. conjugate, `solveTranspose`, conjugate ⇒ $`\boldsymbol{u}_2 = L^{-H}\boldsymbol{u}_1`$
3. $`\boldsymbol{u}_3 = \mathbf{Q}\boldsymbol{u}_2`$
4. solve $`L\boldsymbol{u}_4 = \boldsymbol{u}_3`$ (`ksp.solve`)
5. $`\boldsymbol{u}_5 = \mathbf{M}^{-1}\boldsymbol{u}_4`$
6. $`\boldsymbol{y} = \mathbf{P}^T\boldsymbol{u}_5`$

The operator is therefore
$`\mathbf{D}^+ = \mathbf{P}^T\mathbf{M}^{-1}L^{-1}\mathbf{Q}L^{-H}\mathbf{P}`$.
The modes $`\mathbf{P}\hat{\boldsymbol{f}}^+_i`$ are written to
`eiga_i_<omega>.pval` and the eigenvalues to `eigv_ADJ_<omega>.dat`.

> ⚠️ **To be verified.** $`\mathbf{D}^+`$ is the resolvent of the adjoint system
> ($`L^H`$) restricted to the forcing space ($`\mathbf{P}^T\cdots\mathbf{P}`$).
> It is *not* the "other" Gram operator $`\mathbf{R}\mathbf{R}^\dagger`$ of the
> SVD, which acts on the full response space. Its eigenvalues therefore need
> not match those of $`\mathbf{D}`$ in general (only when $`\mathbf{P} = \mathbf{I}`$
> and the norms commute). Compare `eigv_DIR_*` with `eigv_ADJ_*` to check.

### 7.4 Sensitivity in the resolvent framework

With `compute_sensitivity = True`, `compute_sensitivity_field()` is called for
each pair $`i`$ after the adjoint solve:

```math
\mathcal{S}_i^{\mathrm{res}} = \frac{|\mathbf{P}\hat{\boldsymbol{f}}^+_i|\ |\mathbf{P}\hat{\boldsymbol{f}}_i|}{|\langle\mathbf{P}\hat{\boldsymbol{f}}^+_i,\ \mathbf{B}\,\mathbf{P}\hat{\boldsymbol{f}}_i\rangle|}
\quad\text{(per DOF)},
```

written to `sensitivity_i_<omega>.pval`. Here $`\hat{\boldsymbol{f}}_i`$ is the
direct optimal forcing and $`\hat{\boldsymbol{f}}^+_i`$ is the adjoint one. This
is a heuristic by analogy with the eigenvalue case, **not** the structural
sensitivity of Giannetti & Luchini.

### 7.5 Frequency sweep and gain curve

`listomegas = linspace(i·omega_start, i·omega_end, omega_n)`. For each
frequency:

1. build $`L(\omega)`$ and factorise it (a fresh copy of $`\mathbf{A}`$ each time);
2. run Krylov–Schur on the direct shell;
3. write `eigv_DIR_<omega>.dat`; $`\lambda_1^2`$ is the real part of the first
   row;
4. optionally run the adjoint shell and the sensitivity;
5. free the KSP, the shell and the matrix copy.

**No** consolidated `gain_curve.dat` is written. `tools/plot_gain.py` reads
all `eigv_DIR_*j.dat` files in a directory and builds the curve.

### 7.6 Three-dimensional case (`beta ≠ 0`, partial)

With `beta ≠ 0`, the resolvent assumes the TAU Jacobian contains `nslices`
spanwise slices separated by `slice_spacing` $`= \Delta y`$. It takes the
row block of the central slice $`c`$ = `nslices // 2` and, assuming
$`\hat{\boldsymbol{q}}_{c\pm1} = \hat{\boldsymbol{q}}_c\,e^{\pm i\beta\Delta y}`$, builds

```math
\mathbf{A}_\beta = \mathbf{J}_{c,c} + \mathbf{J}_{c,c+1}\,e^{i\beta\Delta y} + \mathbf{J}_{c,c-1}\,e^{-i\beta\Delta y}.
```

> ⚠️ Still to be checked for `beta ≠ 0`: the length of `samg.matrix.vol`
> (one volume per node of the whole multi-slice mesh, or of a single slice)
> must match the size of $`\mathbf{A}_\beta`$, and the forcing components
> (note 1 in [§7.2](#72-direct-resolvent-matrix-free-shell-operator)).

---

## 8. Post-Processing

All scripts in `tools/` use the non-interactive `Agg` backend, so they can run
on compute nodes without a display.

### 8.1 `tools/plot_eigenvalues.py`

Plots the spectrum in the complex plane from one or more `.dat` files or
directories (the default file is `eigv_DIR.dat`).

- Horizontal axis: $`\sigma_i = 2\pi\mathrm{St}`$. Vertical axis: $`-\sigma_r`$,
  so stable modes appear at the top.
- Several datasets: different colours and markers, with the legend taken from
  the folder name.
- **Two figures** are always produced, with and without index labels:
  `<name>.png` / `<name>_nolabels.png` for one input, or
  `eigv_comparison.png` / `eigv_comparison_nolabels.png` for several.
- Options: `--figsize W H`, `--dpi`, `--label-fontsize`, `--label-every N`.

```bash
python tools/plot_eigenvalues.py RESULTS_eig/ RESULTS_eig_adj/ --label-every 5
```

### 8.2 `tools/plot_pval_eigfunction.py`

Plots 2-D fields stored in `.pval` files (modes, forcing, response,
sensitivity).

- **Coordinates:** `JAC/samg.matrix.coo` (change with `--jac`; one node every
  `neq` rows, or `--neq`), **or** a TAU mesh with `--mesh file.taumesh`. In
  the second case the TAU triangulation is remapped to the `.coo` order, with
  automatic scaling, and the body contour is drawn.
- **Rendering:** Delaunay triangulation (`matplotlib.tri`) or the TAU
  connectivity; `tripcolor` with `shading='gouraud'`.
- **Colour map:** `coolwarm` with a symmetric `TwoSlopeNorm`, clipped at
  percentile `--clim` (default **90**). Sensitivity and $`|U|`$ use positive
  scales.
- **Main options:**
  - `--modes 0-9`, `--modes 0 2 5`, `--modes 0-4 7` with `--dir`
  - `--vars rho u w e`
  - `--fields eigf eigr eiga sensitivity` (resolvent)
  - `--imag`, `--both`
  - `--xlim`, `--ylim`
  - `--check-mesh`
- **Resolvent mode:** detected automatically when the `--dir` name contains
  `resolvent`. It plots `eigf`, `eigr`, `eiga` and `sensitivity` for each
  index, stacked.
- It can also recompute the sensitivity from `eigf`/`eiga` (function
  `compute_sensitivity`, optionally weighted by the volumes).

```bash
python tools/plot_pval_eigfunction.py --modes 0-5 --dir RESULTS_resolvent/ \
    --mesh MESH/BFS_h4_2D.taumesh --fields eigf eigr --xlim -2 15
```

### 8.3 `tools/plot_gain.py`

Reads every `eigv_DIR_<omega>j.dat` in one or more directories and draws the
gain curve on a semi-log scale.

- One directory: $`\lambda_1^2`$ and $`\lambda_2^2`$, written to `<dir>/gain_curve.png`.
- Several directories (`--dirs A B C --labels ...`): $`\lambda_1^2`$ per case,
  written to `gain_curve_comparison.png`. If `--labels` is omitted, the
  `RESULTS_resolvent_` prefix is stripped from the names.
- `--output` overrides the output file name.
- Ticks: major every 1 and minor every 0.5 in $`\omega`$.

---

## 9. SLURM Job Submission

The scripts in `test_cases/BFS/` use a hybrid MPI + OpenMP configuration.
OpenMP threads are useful with a MUMPS build that has OpenMP enabled
(`_ompi` script).

```bash
#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=eig
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=5        # OpenMP threads per MPI rank
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH -e LOG/err-%j.log
#SBATCH -o LOG/out-%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SOLVER_PATH=/path/to/FLORES/solver
INI_FILE=BFS_eig.floresparam

source load_env_STAB_tool_ompi.sh
mkdir -p LOG

mpirun -n ${SLURM_NTASKS} python3 ${SOLVER_PATH}/eig_solver.py ${INI_FILE}
```

For the resolvent, replace the script name with `resolvent_solver.py` and the
control file with the resolvent one.

### 9.1 Timing instrumentation

```python
def _t(comm, rank, label, t0):
    """Print elapsed time since t0 from rank 0, after a barrier sync."""
    comm.Barrier()
    if rank == 0:
        PETSc.Sys.Print(' [TIMING] {0:<40s} {1:8.2f} s'.format(
            label, time.time() - t0))
```

Phases timed:

- **eigenvalue solver:** Jacobian read, mass matrix build, domain reduction,
  assembly of A, assembly of B, EPS solve, saving direct modes, saving adjoint
  modes, structural sensitivity, TOTAL;
- **resolvent:** the same per frequency (LU, direct EPS, saving, adjoint EPS),
  plus totals.

---

## 10. Summary of Design Decisions

| Decision | Rationale |
|---|---|
| In-place PETSc/SLEPc build | Avoids `make install`; the source trees are used directly as `PETSC_DIR`/`SLEPC_DIR`. |
| Complex PETSc (`--with-scalar-type=complex`) | Complex shifts and frequencies, complex modes. |
| No `setType('seqaij')` | `seqaij` is incompatible with several MPI ranks; `setFromOptions()` selects `mpiaij`. |
| Row distribution from rank 0 + `createAIJ(csr=...)` | Each rank receives and preallocates only its own rows; no replicated copies of the Jacobian. |
| `comm.Bcast` instead of `comm.bcast` | Avoids pickling large sparse matrices. |
| One LU per frequency (resolvent) / per shift (EVP) | Reused for `solve` and `solveTranspose` (adjoint) at no extra cost. |
| Domain reduction by index slicing | Equivalent to $`\mathbf{P}^T\mathbf{A}\mathbf{P}`$ without building $`\mathbf{P}`$ or doing sparse products. |
| Checkpoint with `ls` guard | `[ -z "$(...)" ]` on a glob fails under `set -e`. |
| `libgfortran` appended to `LD_LIBRARY_PATH` | Prepending it breaks newer tools (`cmake`). |
| petsc4py/slepc4py with `MAX_JOBS=1`, `-O0` | Parallel Cython builds run out of memory on CESVIMA. |
| `Agg` backend in post-processing | Compute nodes have no display. |
| `TwoSlopeNorm` with percentile clipping | Prevents saturation from very localised amplitudes. |

---

## 11. Known Discrepancies and Open Issues

Review of branch `miguel_dev`. Ordered by impact.

### 11.1 Open — physics / numerics

| # | Location | Description | Suggestion |
|---|---|---|---|
| 3 | both solvers | Whether the Jacobian is divided by the volume depends on TAU's `Jacobian volume scaling (0/1)`, which FLORES cannot detect. The resolvent only supports `1`; the eigensolver relies on the user setting `gen` to match. Mismatches fail silently. | Add a `[physics] jacobian_volume_scaling = 0/1` key. If `0`, scale the rows of $`\mathbf{A}`$ by $`\mathbf{M}^{-1}`$ on read, so both solvers always work with $`\mathbf{M}^{-1}\mathbf{J}`$ and `gen` is no longer needed. |
| 4 | `eig_solver.py` | Adjoint modes written as $`\boldsymbol{p}`$; with scaling `1` + `gen = False` the $`\mathbf{M}`$-adjoint is $`\mathbf{M}^{-1}\boldsymbol{p}`$. | Apply $`\mathbf{M}^{-1}`$ before writing `eiga_N.pval` (and before the sensitivity). |
| 5 | `eig_solver.py` | Sensitivity per DOF, not per node (Frobenius norm). | Compute per node $`\Vert q^+\Vert \Vert q\Vert `$ (velocity) and write one scalar field. |
| 6 | `eig_solver.py` | Normalisation $`p^H\mathbf{B}q`$ is correct only for scaling `0` + `gen = True`; with scaling `1` + `gen = False` it should be $`p^Hq`$. | Becomes automatic with the fix for #3 + #4 ($`\hat q^+=\mathbf{M}^{-1}p`$, $`\langle\hat q^+,\mathbf{M}\hat q\rangle`$). |
| 7 | `eig_solver.py` | Direct/adjoint pairing after separate duplicate filtering. | Pair by EPS index before filtering. |
| 8 | `resolvent_solver.py` | "Adjoint" operator $`\mathbf{P}^T\mathbf{M}^{-1}L^{-1}\mathbf{Q}L^{-H}\mathbf{P}`$ is not $`\mathbf{R}\mathbf{R}^\dagger`$. **Confirmed on a synthetic case:** at $`\omega = 0.5`$ the direct gains are 1.052, 0.667, 0.638 and the "adjoint" ones 1.651, 0.678, 0.619. | Redefine the adjoint operator (and the resolvent sensitivity that uses it). |
| 9 | `resolvent_solver.py` | Energy norm $`\mathbf{Q}=\mathbf{M}`$ (not Chu). | Implement Chu (requires the base flow) or document it. |
| 10 | `resolvent_solver.py` | $`\mathbf{P}`$ selects DOFs 1–2 (wrong for `beta ≠ 0`). | Choose the components according to `neq`/`beta`. |

### 11.2 Fixed

| # | Location | Problem | Fix |
|---|---|---|---|
| 1 | `resolvent_solver.py` | `beta ≠ 0`: `j0 + j1·e^{iβ} − j1·e^{−iβ}` (`jm1` unused, wrong sign, `Ly = 1`). | `j0 + j1·e^{iβΔy} + jm1·e^{−iβΔy}`, central slice `nslices//2`, `Δy` = `[physics] slice_spacing`. |
| 2 | `resolvent_solver.py` | Shell size `n//2`, only valid for `neq = 4`. | Size of the forcing space, `2*(n//neq)`. |
| 11 | `eig_solver.py` | `os.mkdir(output_path)` on every rank. | Rank 0 + `Barrier`. |
| 12 | both solvers | Whole Jacobian replicated on every rank. | `mpi_utils.py`: rank 0 sends each rank only its rows. |
| 13 | `resolvent_solver.py` | Ignored `jac_file`/`vol_file`; EPS `tol`/`max_it` hard-coded. | Read from `[io]` / `[solver]` (defaults unchanged). |
| 14 | `eig_solver.py` | 3-D expansion with 21 hard-coded slices. | `[physics] output_slices` (default 21). |
| 15 | `input_output.py` | Legacy readers with Python 2 semantics. | Ported to Python 3 (vectorised). |
| 16 | `save2pval.py` | `sens2pval` crashed with `dreduced=True` (variable `rho` created twice); duplicated unpacking in `mode2pval3D`. | Rewritten; `mode2pval3D` output unchanged. `global_id` is still `arange` (TAU ids not available in the Jacobian file). |
| 17 | `resolvent_solver.py` | Incorrect, unused `resolvant.mult_transpose`. | Removed. |
| 18 | repo | `__pycache__/*.pyc` committed. | Removed; `.gitignore` added. |
| 19 | docstrings / README | Old script names. | Updated. |
| 20 | `test_cases/` | Absolute CESVIMA paths; wrong `nev`/`ncv` comment; environment script executed instead of sourced. | `${FLORES_DIR}` / `./JAC/`; comments fixed; `source`. |
| 21 | `miguel_dev` vs `main` | `Cylinder_Re45` and newer README only on `main`. | `main` merged into `miguel_dev`. |
| 22 | both solvers | `Mat.getVecs()` was removed in petsc4py 3.26 (crash; with several ranks the other ranks hung). | `Mat.createVecs()` (available in 3.25 too). |
| 23 | `resolvent_solver.py` | `pcreate` passed int64 row pointers to `setPreallocationCSR`, which fails with 32-bit PETSc indices; full $`\mathbf{P}`$ built on every rank with Python loops. | Each rank builds its own rows of $`\mathbf{P}`$ (vectorised). |

### 11.3 Validation of the fixes

The changes were checked on synthetic TAU-format Jacobians (72 grid points,
`neq = 4` and `neq = 6`, with and without domain reduction), with 1 and 3 MPI
ranks, PETSc/SLEPc 3.26 (complex) and MUMPS:

- eigenvalues (direct and adjoint) agree with a dense SciPy reference to
  < 1e-8, and the resolvent gains (direct and "adjoint" operators) to
  < 1e-5 relative;
- eigenvalues, gains and all `.pval` files (modes up to a complex phase,
  sensitivities exactly) are identical to the previous version of the code
  wherever that version runs;
- the resolvent with `neq = 6` failed in the previous version (PETSc error
  60, nonconforming sizes) and now matches the dense reference;
- the `beta ≠ 0` reduction was checked separately against a 7-slice
  block-tridiagonal operator (error 1e-15; the previous formula gave an error
  of order 10).


---

## 12. Glossary

| Term | Meaning |
|---|---|
| BFS | Backward-Facing Step, the canonical test case. |
| DOF | Degree of freedom. |
| DR | Domain Reduction. |
| EPS | Eigenvalue Problem Solver (SLEPc object that runs Krylov–Schur). |
| FLORES | Flow Linear Operators: Resolvent and Eigenvalue Stability. |
| KSP | Krylov Subspace solver (PETSc linear-solver object; here `preonly`). |
| MUMPS | MUltifrontal Massively Parallel sparse direct Solver. |
| NUMATH | Research group at ETSIAE/UPM. |
| PC | Preconditioner; here the MUMPS LU factorisation. |
| PETSc | Portable, Extensible Toolkit for Scientific computation. |
| RANS | Reynolds-Averaged Navier–Stokes. |
| SLEPc | Scalable Library for Eigenvalue Problem computations. |
| ST | Spectral Transformation; here shift-invert. |
| TRANSDIFFUSE | UPM research project under which FLORES is developed. |

---

## Appendix A — Original vs. Refactored Solver

| Aspect | Original (thesis, Python 2) | `solver/eig_solver.py` / `resolvent_solver.py` |
|---|---|---|
| Parameters | Hard-coded in the script | `.floresparam` (INI) file via CLI |
| Checkpoint | None | Load / skip / append (eigenvalues only) |
| MPI broadcast | `comm.bcast` (pickle) | `comm.Bcast` (buffers) |
| Matrix assembly | `setPreallocationCSR` + `xrange` loops | Rows sent from rank 0 to their owner, `createAIJ(csr=...)` (`mpi_utils.py`) |
| Timing | None | Per-phase `_t()` helper |
| Python | Python 2 (`xrange`, `print`) | Python 3 (legacy functions remain in `input_output.py`) |
| MUMPS options | Basic (`icntl_10`, `icntl_11`) | No `ICNTL` set in code; configurable at runtime with `-mat_mumps_icntl_*` |
| Adjoint | Separate solve with $`\mathbf{A}^H`$ | Two-sided EPS (eig) / shell reusing the LU (resolvent) |
| Multi-slice ($`\beta\neq 0`$) | Supported | Eig: 3-D output only. Resolvent: central-slice Fourier reduction (§7.6), partially validated |
