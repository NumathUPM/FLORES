#! /usr/bin/env python

from netCDF4 import Dataset
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
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

def _import_h5py():
    """h5py is only needed for SOD2D input; import it lazily so that TAU
    users do not need it installed."""
    try:
        import h5py
    except ImportError:
        raise ImportError('Reading SOD2D files requires h5py '
                          '(pip install h5py)')
    return h5py


def _sod2d_neq(n, numpoints):
    """Number of equations = degrees of freedom / number of nodes."""
    if numpoints == 0 or n % numpoints != 0:
        raise ValueError('SOD2D Jacobian size ({0}) is not a multiple of the '
                         'number of nodes ({1})'.format(n, numpoints))
    return n // numpoints


def _natural_key(path):
    """Sort key that orders 'file_2.hdf' before 'file_10.hdf'."""
    import re
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', os.path.basename(path))]


def open_sod2d_jacobian(input_path, jacfile):
    """
    Read a SOD2D Jacobian (WIP HDF5 format) and return it as CSR.

    Datasets: col_ptr, row_ind, values (CSC, 0-based) and node_coords.
    If input_path contains more than one .hdf file, they are read as a
    "split parallel" Jacobian (see open_split_sod2d_jacobian).
    """
    h5py = _import_h5py()

    hdfs_in_path = [x for x in os.listdir(input_path) if x.endswith(".hdf")]
    if len(hdfs_in_path) > 1:
        print(" Detected multiple SOD2D hdfs, reading as split Jacobian...")
        return open_split_sod2d_jacobian(input_path)
    if not os.path.isfile(jacfile) and len(hdfs_in_path) == 1:
        jacfile = os.path.join(input_path, hdfs_in_path[0])

    with h5py.File(jacfile, 'r') as f:
        col_ptr = np.array(f['col_ptr'][:], dtype=np.int32)
        row_ind = np.array(f['row_ind'][:], dtype=np.int32)
        values  = np.array(f['values'][:],  dtype=np.float64)
        numpoints = f['node_coords'].shape[0]

    n   = len(col_ptr) - 1
    neq = _sod2d_neq(n, numpoints)

    mjac = csc_matrix((values, row_ind, col_ptr), shape=(n, n))
    mjac = mjac.tocsr()
    mjac.eliminate_zeros()

    return mjac, neq


def open_split_sod2d_jacobian(input_path):
    """
    Read a SOD2D Jacobian written in the "split parallel" format, where
    each .hdf file in input_path holds a block of consecutive columns
    (one file per SOD2D rank). Files are concatenated in natural order of
    their names (rank number), zeros in each col_ptr are discarded.
    """
    h5py = _import_h5py()

    sorted_files = sorted([os.path.join(input_path, x)
                           for x in os.listdir(input_path)
                           if x.endswith(".hdf")], key=_natural_key)

    nnz = 0
    num_cols = 0
    for jacfile in sorted_files:
        with h5py.File(jacfile, 'r') as f:
            col_ptr_with_zeros = np.array(f['col_ptr'], dtype=np.int32)
            nnz += f['values'].shape[0]
        num_cols += len(filter_col_ptr(col_ptr_with_zeros))

    col_ptr = np.zeros(num_cols + 1, dtype=np.int32)
    row_ind = np.zeros(int(nnz),     dtype=np.int32)
    values  = np.zeros(int(nnz),     dtype=np.float64)

    previous_col = 0
    previous_nnz = 0
    numpoints = None
    for jacfile in sorted_files:
        with h5py.File(jacfile, 'r') as f:
            current_col_ptr_with_zeros = np.array(f['col_ptr'], dtype=np.int32)
            current_filtered_col_ptr = filter_col_ptr(current_col_ptr_with_zeros)

            current_nnz = previous_nnz + current_filtered_col_ptr[-1]
            current_col = previous_col + len(current_filtered_col_ptr)
            col_ptr[previous_col+1:current_col+1] = previous_nnz + current_filtered_col_ptr
            row_ind[previous_nnz:current_nnz] = f['row_ind'][:]
            values[previous_nnz:current_nnz]  = f['values'][:]
            # node coords are currently the same in every file of the split
            if numpoints is None:
                numpoints = f['node_coords'].shape[0]

        previous_col = current_col
        previous_nnz = current_nnz

    print(" Got a nnz total of {0} which should match size of row_ind, "
          "which is {1}".format(col_ptr[-1], len(row_ind)))

    n   = len(col_ptr) - 1
    neq = _sod2d_neq(n, numpoints)

    mjac = csc_matrix((values, row_ind, col_ptr), shape=(n, n))
    mjac = mjac.tocsr()
    mjac.eliminate_zeros()

    return mjac, neq


def filter_col_ptr(initial_col_ptr):
    """Drop the zero entries of a per-file col_ptr (columns not stored
    in that file, and the leading 0)."""
    return initial_col_ptr[initial_col_ptr != 0]

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

def read_sod2d_coordinates(coordfile, rlength, beta, neq=5):
    """
    Read node coordinates from a SOD2D HDF5 file (dataset node_coords)
    and repeat each row neq times, so that there is one row per DOF as in
    the TAU .coo file. Columns: x, y, z.
    """
    h5py = _import_h5py()
    print(' READING COORDINATES FROM SOD2D OUTPUT FILE: ', coordfile)

    with h5py.File(coordfile, 'r') as f:
        coords = np.array(f['node_coords'][:], dtype=np.float64)
    coords *= rlength

    return np.repeat(coords, neq, axis=0)     # shape (gridpoints*neq, ndim)

def dump_sod2d_coordinates(filename: str, coords: np.ndarray, neq: int, kept_idx = None):
    """
    Dump the coords read from SOD2D to an output file matching the
    .coo format from TAU

    This makes visualisation easier later
    """
    print(' DUMPING SOD2D COORDS TO FILE: ', filename)
    if(type(kept_idx) is np.ndarray):
        new_coords = coords[kept_idx,:]
    else:
        new_coords = coords
    with open(filename,"w") as f:
        coords_shape = np.shape(new_coords)
        f.write(f"{coords_shape[0]} {coords_shape[1]} \n")
        for i in range(0, coords_shape[0]):
            f.write(f"{new_coords[i,0]} {new_coords[i,1]} {new_coords[i,2]} \n")


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