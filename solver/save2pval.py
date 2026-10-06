#! /usr/bin/env python

import sys, os
from netCDF4 import Dataset
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _write_var(nc, name, first_half, second_half=None):
    """
    Create a float64 variable on 'no_of_points' and fill both halves.

    Parameters
    ----------
    nc          : open Dataset
    name        : variable name
    first_half  : array of length gridpoints (first half of the TAU layout)
    second_half : array of length gridpoints, or None to mirror first_half
    """
    nc.createVariable(name, 'f8', ('no_of_points',))
    gp = len(first_half)
    nc.variables[name][:gp] = first_half
    nc.variables[name][gp:] = first_half if second_half is None else second_half


def _unpack_sol(sol, neq, is_simulator_sod2d=False):
    """
    Unpack a flat complex solution vector into per-variable arrays via
    strided slicing (no Python loops).

    Returns a dict with keys: rho, u, w, e, [turb1], [turb2]
    """
    if(is_simulator_sod2d):
        d = {
        'rho':  sol[0::neq],
        'u':    sol[1::neq],
        'v':    sol[2::neq],
        'w':    sol[3::neq],
        'e':    sol[4::neq],
        }
        return d   
    d = {
        'rho':  sol[0::neq],
        'u':    sol[1::neq],
        'w':    sol[2::neq],
        'e':    sol[3::neq],
    }
    if neq >= 5:
        d['turb1'] = sol[4::neq]
    if neq >= 6:
        d['turb2'] = sol[5::neq]
    return d


############################################################################
# sens2pval
############################################################################

def sens2pval(filename, gid, sens, npoints, neq, dreduced=False, rgid=None):
    """
    Write a sensitivity field to a TAU-compatible netCDF .pval file.

    Variables written (both halves of 'no_of_points'):
        rho, u, w, e (+ '_i' imaginary parts), v (zero for 2-D),
        sens_R = |Re(u, w)|, sens_Im = |Im(u, w)|, [turb1, turb2 (+ '_i')]

    Parameters
    ----------
    gid      : TAU global_id array (length 2*gridpoints)
    sens     : complex sensitivity vector (length nred, interleaved by neq)
    npoints  : unused, kept for backward compatibility
    dreduced : True if sens lives on a reduced domain
    rgid     : grid-point indices of the reduced domain (dreduced=True)

    Returns
    -------
    sens_R + 1j*sens_Im  (on the reduced/original sens grid)
    """
    gridpoints = len(gid) // 2

    fields = {k: np.array(val) for k, val in _unpack_sol(sens, neq).items()}
    fields['v'] = np.zeros_like(fields['u'])     # spanwise: zero for 2-D

    sens_R  = np.sqrt(fields['u'].real**2 + fields['w'].real**2)
    sens_Im = np.sqrt(fields['u'].imag**2 + fields['w'].imag**2)

    def _full(arr):
        """Scatter a reduced-domain array onto the full grid."""
        if not dreduced:
            return arr
        out = np.zeros(gridpoints, dtype=arr.dtype)
        out[rgid] = arr
        return out

    amg_f = Dataset(filename, 'w')
    amg_f.createDimension('no_of_points', len(gid))
    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = gid

    for vname in ['rho', 'u', 'w', 'e', 'turb1', 'turb2']:
        if vname in fields:
            arr = _full(fields[vname])
            _write_var(amg_f, vname,        arr.real)
            _write_var(amg_f, vname + '_i', arr.imag)
    _write_var(amg_f, 'v',       _full(fields['v']).real)
    _write_var(amg_f, 'sens_R',  _full(sens_R))
    _write_var(amg_f, 'sens_Im', _full(sens_Im))

    amg_f.close()
    return np.array(sens_R + 1j * sens_Im)


############################################################################
# sol2pval
############################################################################

def sol2pval(filename, gid, sol, npoints, neq, dreduced=False, rgid=None):
    """Write a real base-flow solution to a TAU .pval file."""
    gridpoints = npoints // neq

    # Vectorised unpacking
    rho = np.asarray(sol[0::neq], dtype=np.float64)
    u   = np.asarray(sol[1::neq], dtype=np.float64)
    w   = np.asarray(sol[2::neq], dtype=np.float64)
    e   = np.asarray(sol[3::neq], dtype=np.float64)
    v   = np.zeros(gridpoints,    dtype=np.float64)   # spanwise: zero for 2-D

    amg_f = Dataset(filename, 'w', format='NETCDF3_64BIT_OFFSET')
    amg_f.createDimension('no_of_points', gridpoints * 2)

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = gid

    fields = [('density', rho), ('x_velocity', u),
              ('y_velocity', v), ('z_velocity', w), ('pressure', e)]

    if dreduced:
        for vname, arr in fields:
            full = np.zeros(gridpoints)
            full[rgid] = arr
            _write_var(amg_f, vname, full)
    else:
        for vname, arr in fields:
            _write_var(amg_f, vname, arr)

    amg_f.close()


############################################################################
# mode2pval
############################################################################

def mode2pval(filename, sol, npoints, nred, neq, beta=0.0,
              dreduced=False, rgid=None, is_simulator_sod2d=False):
    """
    Write a complex global mode to a TAU-compatible netCDF .pval file.

    Parameters
    ----------
    filename : output path
    sol      : PETSc Vec or numpy array (complex, length nred)
    npoints  : total grid points * neq  (full mesh, for output sizing)
    nred     : number of DOFs in the (possibly reduced) solution vector
    neq      : number of equations per grid point
    beta     : spanwise wavenumber (informational only here)
    dreduced : True if domain reduction was applied
    rgid     : reduced grid point indices (used when dreduced=True)
    """
    # Convert PETSc Vec to numpy if needed
    if hasattr(sol, 'getArray'):
        sol = sol.getArray()
    sol = np.asarray(sol, dtype=np.complex128)

    vars_red = _unpack_sol(sol, neq, is_simulator_sod2d)   # dict: rho, u, w, e, [turb1, turb2]

    gridpoints_out = npoints // neq    # full mesh size for output file

    amg_f = Dataset(filename, 'w', format='NETCDF3_64BIT_OFFSET')
    amg_f.createDimension('no_of_points', gridpoints_out * 2)

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = np.arange(gridpoints_out * 2, dtype='i4')

    
    base_vars = ['rho', 'u', 'w', 'e']
    turb_vars = []
    if neq >= 5:
        turb_vars.append('turb1')
    if neq >= 6:
        turb_vars.append('turb2')
    all_vars = base_vars + turb_vars
    if(is_simulator_sod2d):
        base_vars = ['rho', 'u', 'v', 'w', 'e']
        turb_vars = []
        all_vars = base_vars + turb_vars
    if dreduced:
        # Scatter reduced-domain values into full-size arrays
        for vname in all_vars:
            full = np.zeros(gridpoints_out, dtype=np.complex128)
            full[rgid] = vars_red[vname]
            _write_var(amg_f, vname,       full.real)
            _write_var(amg_f, vname + '_i', full.imag)
    else:
        gp_red = nred // neq
        for vname in all_vars:
            arr = vars_red[vname]
            # Pad with zeros if reduced size < full size (shouldn't happen
            # in the non-dreduced path, but guard against shape mismatch)
            if len(arr) < gridpoints_out:
                full = np.zeros(gridpoints_out, dtype=np.complex128)
                full[:len(arr)] = arr
            else:
                full = arr
            _write_var(amg_f, vname,       full.real)
            _write_var(amg_f, vname + '_i', full.imag)

    amg_f.close()


############################################################################
# mode2pval3D
############################################################################

def mode2pval3D(filename, sol, npoints, nred, neq, beta, nums,
                dreduced=False, rgid=None):
    """
    Write a 3-D reconstructed mode (beta != 0) to a TAU .pval file.

    The spanwise expansion uses numpy broadcasting instead of nested
    Python loops.
    """
    if hasattr(sol, 'getArray'):
        sol = sol.getArray()
    sol = np.asarray(sol, dtype=np.complex128)

    gridpoints_red = nred  // neq
    gridpoints_out = npoints // neq

    # Unpack — vectorised.  Variable layout per grid point:
    #   beta == 0 : rho, u, w, e        (v = 0)
    #   beta != 0 : rho, u, v, w, e
    zeros = np.zeros(gridpoints_red, dtype=np.complex128)
    if beta != 0.0:
        rho = sol[0::neq]
        u   = sol[1::neq]
        v   = sol[2::neq]
        w   = sol[3::neq]
        e   = sol[4::neq] if neq >= 5 else zeros
    else:
        rho = sol[0::neq]
        u   = sol[1::neq]
        w   = sol[2::neq]
        e   = sol[3::neq]
        v   = zeros

    # Scatter to full mesh if domain-reduced
    def _to_full(arr):
        if dreduced:
            full = np.zeros(gridpoints_out, dtype=np.complex128)
            full[rgid] = arr
            return full
        if len(arr) < gridpoints_out:
            full = np.zeros(gridpoints_out, dtype=np.complex128)
            full[:len(arr)] = arr
            return full
        return arr

    rho = _to_full(rho)
    u   = _to_full(u)
    v   = _to_full(v)
    w   = _to_full(w)
    e   = _to_full(e)

    out_file = filename[:-5] + '3D.pval' if filename.endswith('.pval') else filename[:-4] + '3D.pval'
    amg_f = Dataset(out_file, 'w', format='NETCDF3_64BIT_OFFSET')
    amg_f.createDimension('no_of_points', gridpoints_out * nums)

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = np.arange(gridpoints_out * nums, dtype='i4')

    # Spanwise phase expansion — vectorised with numpy broadcasting
    # slice_phases shape: (nums,) ; values: exp(i * beta * y_slice)
    Ly           = 2.0 * np.pi / beta if beta != 0.0 else 1.0
    y_slices     = np.arange(nums) * (Ly / nums)           # (nums,)
    phases       = np.exp(1j * beta * y_slices)            # (nums,)

    for ncname, base_arr in [('rho', rho), ('u', u), ('v', v),
                              ('w', w),    ('e', e)]:
        amg_f.createVariable(ncname, 'f8', ('no_of_points',))
        # base_arr shape: (gridpoints_out,)
        # expanded:       (nums, gridpoints_out) via outer product, then flatten
        expanded = np.real(
            np.outer(phases, base_arr)    # (nums, gridpoints_out)
        ).ravel()                         # (nums * gridpoints_out,)
        amg_f.variables[ncname][:] = expanded

    amg_f.close()