#! /usr/bin/env python

from netCDF4 import Dataset
import numpy as np
from scipy.sparse import csr_matrix
import sys,os

## ================================================================= ##

def opendualgrid(duafile):
    """Opens a netCDF TAU jacobian matrix, and stores it into a CSR matrix."""
    dua = Dataset(duafile)
    dua.set_auto_mask(False)
    npoints = dua.dimensions['nallpoints'].size
    
    # Reading local_id
    local_id = np.zeros(npoints)
    local_id[:] = dua.variables['local_id'][:]
    dua.close()
    return local_id

# -------------------------------------------------------------------- #

def openjacobian(jacfile):
    """Opens a netCDF TAU jacobian matrix, and stores it into a CSR matrix."""
    jac = Dataset(jacfile)
    nnz = jac.dimensions['nnz'].size
    n   = jac.dimensions['nvars'].size - 1
    neq = jac.dimensions['neq'].size

    # Reading and index correction (from Fortran to C)
    row_ptr = jac.variables['row_ptr'][:] - 1
    col_ind = jac.variables['col_ind'][:] - 1
    data = jac.variables['data'][:]
    jac.close()
    mjac = csr_matrix((data, col_ind, row_ptr), shape=(n,n))
    mjac.eliminate_zeros()
    return mjac, neq

# -------------------------------------------------------------------- #

def openqe(jacfile):
    """Opens a netCDF TAU jacobian matrix, and stores it into a CSR matrix."""
    jac = Dataset(jacfile)
    nnz = jac.dimensions['nnz'].size
    n   = jac.dimensions['nvars'].size - 1
    neq = jac.dimensions['neq'].size

    # Reading and index correction (from Fortran to C)
    row_ptr = jac.variables['row_ptr'][:]
    col_ind = jac.variables['col_ind'][:]
    data = jac.variables['data'][:]
    jac.close()
    mjac = csr_matrix((data, col_ind, row_ptr), shape=(n,n))
    # mjac.eliminate_zeros()
    return mjac, neq

# -------------------------------------------------------------------- #

def _interleave(fields, neq, dtype):
    """Interleave per-point arrays [rho, u, w, e, (t1, t2)] into a flat
    node-major vector of length gridpoints*neq."""
    gridpoints = len(fields[0])
    q = np.zeros(gridpoints*neq, dtype=dtype)
    for k in range(neq):
        q[k::neq] = fields[k]
    return q

# -------------------------------------------------------------------- #

def openegvec(modefile, neq):
    """Opens a global mode from zTAUev and stores it into a 1D array."""
    mode = Dataset(modefile)
    gridpoints = mode.dimensions['no_of_points'].size // 2
    v = mode.variables
    fields = [v[name][:gridpoints] + 1j*v[name+'_i'][:gridpoints]
              for name in ('rho', 'u', 'w', 'e')]
    if neq > 4:
        fields += [v[name][:gridpoints] + 1j*v[name+'_i'][:gridpoints]
                   for name in ('turb1', 'turb2')]
    gid = v['global_id'][:]
    mode.close()
    return _interleave(fields, neq, 'c16'), gid

# -------------------------------------------------------------------- #

def openresidual(resfile, neq):
    """Opens a residuals file from TAU and stores it into a 1D array."""
    mode = Dataset(resfile)
    gridpoints = mode.dimensions['no_of_points'].size // 2
    v = mode.variables
    names = ['density_residual', 'x-velocity_residual',
             'z-velocity_residual', 'energy_residual']
    if neq > 4:
        names += ['k_residual', 'omega_residual']
    fields = [v[name][:gridpoints] for name in names]
    mode.close()
    return _interleave(fields, neq, 'f8')

# -------------------------------------------------------------------- #

def openbflow(resfile, neq):
    """Opens a base-flow solution file from TAU and stores it into a 1D array."""
    mode = Dataset(resfile)
    gridpoints = mode.dimensions['no_of_points'].size // 2
    v = mode.variables
    names = ['density', 'x_velocity', 'z_velocity', 'pressure']
    if neq > 4:
        names += ['turb_kinetic_energy', 'turb_omega']
    fields = [v[name][:gridpoints] for name in names]
    gid = v['global_id'][:]
    mode.close()
    return _interleave(fields, neq, 'f8'), gid

# -------------------------------------------------------------------- #

def read_coordinates(coordfile, rlength, beta):
    """Read coordinates from TAU coo file.

    Optimised version: uses numpy.loadtxt (C-level parser) instead of
    Python-level readlines/map/list-comprehension, and replaces all
    O(n) Python loops with vectorised numpy operations.

    Speedup vs original: typically 10-50x for large meshes.
    """
    print(' READING COORDINATES FROM COORD FILE: ', coordfile)

    # ── Read header (first line) and data in one C-level call ────────────
    with open(coordfile) as f:
        ndof, ndim = f.readline().split()   # header: total_rows  n_dims

    # numpy.loadtxt is implemented in C and is 10-50x faster than
    # Python-level readlines + map(float, ...) for large files.
    data = np.loadtxt(coordfile, dtype=np.float64, skiprows=1)
    data *= rlength   # scale with reference length

    # ── Detect neq: count leading duplicate coordinate rows ──────────────
    # Vectorised: compare all rows against row 0 simultaneously,
    # find first row that differs — no Python loop needed.
    matches = np.all(data == data[0, :], axis=1)   # bool array
    # argmin returns index of first False (first non-matching row)
    first_diff = int(np.argmin(matches))
    neq = first_diff if first_diff > 0 else 1
    print(' Number of equations in coordinates file = ', neq)

    if beta == 0.0:
        return data

    # ── beta != 0: deduplicate and expand to neq+1 equations ─────────────
    print(' Correcting number of equations in coordinates file...')

    # Extract one row per grid point (every neq-th row starting at 0)
    coord = data[::neq, :]                          # shape (gridpoints, ndim)

    # Expand: each grid point repeated (neq+1) times — pure numpy, no loop
    new_data = np.repeat(coord, neq + 1, axis=0)   # shape (gridpoints*(neq+1), ndim)

    return new_data

# -------------------------------------------------------------------- #

def opensensitivity(sensfile, neq):
    """Opens a sensitivity file and stores it into a 1D array."""
    mode = Dataset(sensfile)
    gridpoints = mode.dimensions['no_of_points'].size // 2
    v = mode.variables
    fields = [v[name][:gridpoints] + 1j*v[name+'_i'][:gridpoints]
              for name in ('rho', 'u', 'w', 'e')]
    if neq > 4:
        fields += [v[name][:gridpoints] + 1j*v[name+'_i'][:gridpoints]
                   for name in ('t1', 't2')]
    mode.close()
    return _interleave(fields, neq, 'c16')

# -------------------------------------------------------------------- #