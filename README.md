# FLORES: Flow Linear Operators: Resolvent and Eigenvalue Stability

MIT License

Copyright (c) 2026 NUMATH https://numath.dmae.upm.es

## Introduction

**FLORES** is a parallel Python toolkit for global stability analysis and resolvent-based input–output analysis of compressible flows. It computes eigenvalues, eigenmodes, and optimal forcing/response modes using PETSc/SLEPc and the MUMPS direct solver, and is designed to run on HPC clusters via MPI.

---

## Table of Contents

- [Background](#background)
- [Repository Structure](#repository-structure)
- [Dependencies & Installation](#dependencies--installation)
- [Usage](#usage)
- [Authors & Acknowledgements](#authors--acknowledgements)
- [License](#license)

---

## Background

FLORES targets two classical problems in hydrodynamic stability theory:

1. **Global eigenvalue analysis** — identifies the natural oscillatory modes of a flow (oscillator behaviour).
2. **Resolvent analysis** — quantifies the linear amplification of external forcing by computing the leading singular values and modes of the resolvent operator (amplifier behaviour).

Both analyses are performed on a linearised Navier–Stokes Jacobian produced by CFD solver: DLR TAU Code.

---


## Repository Structure

```
FLORES/
    ├── solver/
        ├── eig_solver.py            # Global stability solver (direct & adjoint & structural sensitivity)
        ├── resolvent_solver.py      # Resolvent operator solver
        ├── jac_red.py               # Domain-reduction utilities
        ├── save2pval.py             # Output routines (eigenvector → .pval files)
        ├── input_output.py          # Jacobian and coordinate readers
    ├── python_env_installation      #  Scripts to install the python environment
    ├── JAC/                         # Input directory for jacobian matrices
    ├── RESULTS_eig/                 # Output directory for eigenvalue runs
    ├── RESULTS_resolvent/           # Output directory for resolvent runs
    ├── test_cases/           # Test cases
        ├── Cylinder_Re45          # Cylinder Re=45 and M=0.1
    └── README.md
```

---

## Dependencies & Installation


Automated installation scripts are provided in the `python_env_installation/` folder. They handle everything: virtual environment creation, PETSc/SLEPc compilation (inplace, no `make install`), and the installation of `mpi4py`, `petsc4py`, `slepc4py`, and all extra Python dependencies. Both scripts include checkpoint logic, so they can be safely re-run if interrupted.

### `Cesvima_UPM_Installation.sh` — HPC cluster (CESVIMA / UPM)

Designed for the CESVIMA cluster at UPM. It links against the cluster's existing MPI, OpenBLAS, ScaLAPACK, and MUMPS modules (`foss/2021a` toolchain) rather than downloading them, and applies the necessary `libgfortran` path fix for the GCCcore 7.2.0 runtime.

From the root of the repository:

```bash
cd python_env_installation
bash Cesvima_UPM_Installation.sh
```

> **Important:** Submit this script as a SLURM job — do not run it on the login node. PETSc/SLEPc compilation and the `petsc4py`/`slepc4py` builds are memory-intensive and will be killed by the login node's OOM policy.

### `Ubuntu_Installation.sh` — Local Ubuntu workstation

Designed for a standard Ubuntu desktop or laptop. It downloads and compiles all dependencies from scratch (MPICH, BLAS/LAPACK, ScaLAPACK, MUMPS, CMake, METIS, ParMETIS), so no pre-installed MPI or system libraries are required beyond a working C/Fortran compiler.

From the root of the repository:

```bash
cd python_env_installation
bash Ubuntu_Installation.sh
```

### After installation (both platforms)

Both scripts patch the virtual environment's `activate` script with the correct `PETSC_DIR`, `SLEPC_DIR`, `PETSC_ARCH`, and `LD_LIBRARY_PATH` variables. To activate the environment in future sessions:

```bash
source myvenv/bin/activate
```

A sanity check is run automatically at the end of each script, printing the PETSc and SLEPc versions and verifying MPI communication.

---

## Usage

Both solvers are configured through an `.ini` control file passed as a
command-line argument, and can be run serially or in parallel via MPI.

### Global stability analysis (`solver/eig_solver.py`)

Create a control file (e.g. `eigensolver.ini`):

```ini
[io]
input_path   = JAC/
output_path  = RESULTS_eig/
jac_file     = samg.matrix.amg.pval
vol_file     = samg.matrix.vol
coord_file   = samg.matrix.coo

[physics]
mach    = 0.1
beta    = 0.0
rlength = 1.0

[solver]
nev          = 50           # number of eigenvalues requested
ncv          = 0            # Krylov subspace size (0 = auto: nev*3+1)
shift_real   = -0.05        # real part of spectral shift
shift_imag   =  4.0         # imaginary part of spectral shift
tol          = 1e-8         # SLEPc convergence tolerance
max_it       = 15000        # maximum Krylov iterations
adjoint      = False        # set True to also compute adjoint modes
sensitivity  = False        # set True to compute structural sensitivity
                            # (automatically enables adjoint = True)
gen          = False        # generalised EVP (A x = s M x)?

[domain_reduction]
enabled = False             # set True to restrict to a subdomain
xmin    = -2.0
xmax    =  20.0
zmin    = -5.0
zmax    =  5.0

[checkpoint]
dup_tol_real = 1e-5         # duplicate-detection tolerance (real part)
dup_tol_imag = 1e-5         # duplicate-detection tolerance (imag part)
```

Run:

```bash
# Serial
python solver/eig_solver.py eigensolver.ini

# Parallel
mpirun -np 8 python solver/eig_solver.py eigensolver.ini
```

Converged direct eigenvalues are appended to `RESULTS_eig/eigv_DIR.dat`
and eigenvectors written as `RESULTS_eig/eigf_N.pval`. When
`adjoint = True`, adjoint eigenvalues go to `eigv_ADJ.dat` and adjoint
modes to `eiga_N.pval`. When `sensitivity = True`, the structural
sensitivity field for each mode pair is written to
`RESULTS_eig/sensitivity_N.pval`. Duplicate detection across restarts
is built in: previously converged eigenvalues are loaded on startup and
skipped automatically.

---

### Resolvent analysis (`solver/resolvent_solver.py`)

Create a control file (e.g. `resolvent.ini`):

```ini
[io]
input_path   = JAC/
output_path  = RESULTS_resolvent/
coord_file   = samg.matrix.coo

[physics]
mach    = 0.1
beta    = 0.0
rlength = 1.0

[frequencies]
omega_start = 10.0          # start of frequency sweep (imaginary part)
omega_end   = 150.0         # end of frequency sweep
omega_n     = 50            # number of frequencies

[solver]
nev                  = 5    # number of singular values requested
ncv                  = 20   # Krylov subspace size
shift                = 0.0  # spectral shift
adjoint              = False # set True to also run adjoint resolvent
compute_sensitivity  = False # set True to compute resolvent sensitivity
                             # (automatically enables adjoint = True)

[domain_reduction]
enabled = False
xmin    = -2.0
xmax    =  20.0
zmin    = -5.0
zmax    =  5.0
```

Run:

```bash
# Serial
python solver/resolvent_solver.py resolvent.ini

# Parallel
mpirun -np 8 python solver/resolvent_solver.py resolvent.ini
```

For each frequency `omega`, the optimal forcing modes are written to
`RESULTS_resolvent/eigf_i_{omega}.pval`, the optimal response modes to
`eigr_i_{omega}.pval`, and the gain values to `eigv_DIR_{omega}.dat`.
When `adjoint = True`, adjoint modes are written to
`eiga_i_{omega}.pval`. When `compute_sensitivity = True`, the
resolvent structural sensitivity is written to
`sensitivity_i_{omega}.pval`.

---

### Domain reduction

Domain reduction can be enabled in either solver to restrict the
eigenvalue or resolvent problem to a physically relevant subdomain
$\Omega_m \subset \Omega_n$, reducing memory and factorisation cost by
$10\times$–$50\times$. Set `enabled = True` in `[domain_reduction]`
and specify the bounding box. The subdomain should contain the region
of high structural sensitivity of the dominant mode.

---

### SLURM example

```bash
#!/bin/bash
#SBATCH --job-name=flores_eig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source myvenv/bin/activate
mpirun -np 8 python solver/eig_solver.py eigensolver.ini
```



## Authors & Acknowledgements

Copyright (c) 2026 NUMATH https://numath.dmae.upm.es

**Development:** Alejandro Martinez-Cava, Iván Padilla, Miguel Chávez-Modena, 

**Original implementation:** The resolvent and eigenvalue solver architecture is based on the original code developed by **Alejandro Martínez Cava** as part of his doctoral thesis at the Universidad Politécnica de Madrid (UPM). His foundational work on the matrix-free resolvent operator and the PETSc/SLEPc solver infrastructure made this tool possible. Martínez-Cava Aguilar, Alejandro  (2019). Direct and Adjoint Methods for Highly Detached Flows. Tesis (Doctoral), E.T.S. de Ingeniería Aeronáutica y del Espacio (UPM). https://doi.org/10.20868/UPM.thesis.56391. 

This work is part of the **TRANSDIFFUSE** project at UPM.

---

```
MIT License

Copyright (c) 2025 Miguel, Universidad Politécnica de Madrid

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
