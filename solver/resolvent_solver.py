#! /usr/bin/env python
#
# Usage: python RESOLVANT.py resolvent.ini
#        mpirun -n 4 python RESOLVANT.py resolvent.ini
#
import numpy as np
import sys, os
import time
import gc
import configparser
from scipy.sparse import csr_matrix
from jac_red import domain_reduction
from save2pval import mode2pval
from input_output import openjacobian, read_coordinates
from mpi_utils import assemble_aij, assemble_diag

import petsc4py
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

def _t(comm, rank, label, t0):
    """Print elapsed time since t0 from rank 0, after a barrier sync."""
    comm.Barrier()
    if rank == 0:
        PETSc.Sys.Print(' [TIMING] {0:<45s} {1:8.2f} s'.format(
            label, time.time() - t0))


# ─────────────────────────────────────────────────────────────────────────────
# Control file reader
# ─────────────────────────────────────────────────────────────────────────────

def read_control_file(filepath):
    """
    Read parameters from a .ini control file.

    Expected sections and keys:
        [io]
            input_path          : path to Jacobian and volume files
            output_path         : path for output files
            jac_file            : Jacobian filename    (default: samg.matrix.amg.pval)
            vol_file            : volumes filename     (default: samg.matrix.vol)
            coord_file          : coordinates filename (default: samg.matrix.coo)

        [physics]
            mach                : Mach number
            beta                : spanwise wavenumber (0 for 2D)
            nslices             : number of slices (only used if beta != 0)
            slice_spacing       : spanwise distance between consecutive
                                  slices, dy (only used if beta != 0,
                                  default: 1.0)
            rlength             : reference length (default: 1.0)

        [frequencies]
            omega_start         : start of frequency range (imaginary part)
            omega_end           : end of frequency range (imaginary part)
            omega_n             : number of frequencies

        [solver]
            nev                 : number of eigenvalues requested
            ncv                 : number of column vectors
            shift               : spectral shift (real)
            tol                 : EPS convergence tolerance (default: 1e-6)
            max_it              : EPS maximum iterations    (default: 1000)
            compute_sensitivity : True/False

        [domain_reduction]
            enabled             : apply domain reduction? (default: False)
            xmin                : x lower bound
            xmax                : x upper bound
            zmin                : z lower bound
            zmax                : z upper bound
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError('Control file not found: {0}'.format(filepath))

    cfg = configparser.ConfigParser()
    cfg.read(filepath)

    params = {}

    # [io]
    params['input_path']  = cfg.get('io', 'input_path').strip()
    params['output_path'] = cfg.get('io', 'output_path').strip()
    params['jac_file']    = cfg.get('io', 'jac_file',
                                    fallback='samg.matrix.amg.pval').strip()
    params['vol_file']    = cfg.get('io', 'vol_file',
                                    fallback='samg.matrix.vol').strip()
    params['coord_file']  = cfg.get('io', 'coord_file',
                                    fallback='samg.matrix.coo').strip()

    # [physics]
    params['mach']    = cfg.getfloat('physics', 'mach')
    params['beta']    = cfg.getfloat('physics', 'beta',    fallback=0.0)
    params['nslices'] = cfg.getint  ('physics', 'nslices', fallback=7)
    params['slice_spacing'] = cfg.getfloat('physics', 'slice_spacing', fallback=1.0)
    params['rlength'] = cfg.getfloat('physics', 'rlength', fallback=1.0)

    # [frequencies]
    omega_start = cfg.getfloat('frequencies', 'omega_start')
    omega_end   = cfg.getfloat('frequencies', 'omega_end')
    omega_n     = cfg.getint  ('frequencies', 'omega_n')
    params['listomegas'] = list(np.linspace(omega_start*1j, omega_end*1j, omega_n))

    # [solver]
    params['nev']                 = cfg.getint    ('solver', 'nev')
    params['ncv']                 = cfg.getint    ('solver', 'ncv')
    params['shift']               = cfg.getfloat  ('solver', 'shift')
    params['tol']                 = cfg.getfloat  ('solver', 'tol',    fallback=1e-6)
    params['max_it']              = cfg.getint    ('solver', 'max_it', fallback=1000)
    params['adjoint']             = cfg.getboolean('solver', 'adjoint',             fallback=False)
    params['compute_sensitivity'] = cfg.getboolean('solver', 'compute_sensitivity', fallback=False)

    # sensitivity requires adjoint — enforce it
    if params['compute_sensitivity'] and not params['adjoint']:
        params['adjoint'] = True

    # [domain_reduction]
    params['dreduced'] = cfg.getboolean('domain_reduction', 'enabled', fallback=False)
    params['xmin']     = cfg.getfloat  ('domain_reduction', 'xmin',    fallback=0.0)
    params['xmax']     = cfg.getfloat  ('domain_reduction', 'xmax',    fallback=1.0)
    params['zmin']     = cfg.getfloat  ('domain_reduction', 'zmin',    fallback=-1.0)
    params['zmax']     = cfg.getfloat  ('domain_reduction', 'zmax',    fallback=1.0)

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Resolvent class (direct)
# ─────────────────────────────────────────────────────────────────────────────

class resolvant(object):
    """Direct resolvent operator: D = P^T M^-1 (L*)^-1 Q L^-1 P"""

    def __init__(self, n, Minv, Qe, J, w, neq):
        self.N    = n
        self.Minv = Minv
        self.Q    = Qe
        self.P    = None
        self.PT   = None
        self.ksp  = None
        self.iter = 1
        self.J    = J
        self.pcreate(neq)

    def mult(self, mat, x, y):
        """y <- D * x = P^T M^-1 (L*)^-1 Q L^-1 P x"""
        v1, tmp = self.Q.getVecs()
        v2, tmp = self.Q.getVecs()
        self.P.mult(x, v1)
        self.ksp.solve(v1, v2)
        self.Q.mult(v2, v1)
        v1.conjugate()
        self.ksp.solveTranspose(v1, v2)
        v2.conjugate()
        self.Minv.mult(v2, v1)
        self.P.multTranspose(v1, y)
        self.iter += 1
        v1.destroy(); v2.destroy(); tmp.destroy()

    def operator(self, w):
        """Build L = iw*I - A and perform LU factorization via MUMPS."""
        Print = PETSc.Sys.Print
        Print(' w = {0}'.format(w))
        self.J.scale(-1.0)
        self.J.shift(w)

        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.J)
        self.ksp.setType('preonly')
        pc = self.ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        self.ksp.setFromOptions()

        Print('')
        Print(' ******************************************')
        Print('  Performing LU decomposition...')
        Print('  OMP threads: {0}'.format(os.environ.get('OMP_NUM_THREADS', '1')))
        Print(' ******************************************')
        Print('')
        self.ksp.setUp()

    def cleanup(self):
        if self.ksp is not None:
            self.ksp.destroy()
            self.ksp = None
        if self.P is not None:
            self.P.destroy()
            self.P = None
        self.J = None; self.Q = None; self.Minv = None

    def pcreate(self, neq):
        """Build prolongation matrix P."""
        comm  = MPI.COMM_WORLD
        size  = comm.Get_size()
        Print = PETSc.Sys.Print

        dimrows = self.N
        dimcol  = 2 * dimrows // neq
        irn = np.zeros(dimcol)
        jcn = np.zeros(dimcol)
        val = np.ones(dimcol)
        for i in range(dimcol):
            jcn[i] = i
        j = 0
        for i in range(0, dimrows, neq):
            irn[j]   = i + 1
            irn[j+1] = i + 2
            j += 2
        ppy = csr_matrix((val, (irn, jcn)), shape=(dimrows, dimcol))

        self.P = PETSc.Mat()
        self.P.create(PETSc.COMM_WORLD)
        self.P.setSizes([dimrows, dimcol])
        self.P.setFromOptions()
        self.P.setUp()

        nnz_row   = ppy.getnnz(axis=1)
        nnz_count = np.concatenate([[0], np.cumsum(nnz_row)])
        RowStart, RowEnd = self.P.getOwnershipRange()
        nrows     = RowEnd - RowStart
        nnz_start = nnz_count[RowStart]
        nnz_end   = nnz_count[RowEnd]
        row_proc  = [(ppy.indptr[RowStart+i] - ppy.indptr[RowStart])
                     for i in range(nrows+1)]
        row_proc[0] = 0
        col_proc = ppy.indices[nnz_start:nnz_end]
        val_proc = ppy.data[nnz_start:nnz_end]
        Print(' Assembling P PETSc matrix...')
        self.P.setPreallocationCSR((row_proc, col_proc, val_proc))
        self.P.assemble()


# ─────────────────────────────────────────────────────────────────────────────
# Resolvent class (adjoint)
# ─────────────────────────────────────────────────────────────────────────────

class resolvant_adjoint(object):
    """Adjoint resolvent: D* = P^T M^-1 L^-1 Q (L*)^-1 P
    Reuses the same KSP (LU factorization) as the direct operator.
    """

    def __init__(self, shell_direct):
        self.ksp  = shell_direct.ksp
        self.Q    = shell_direct.Q
        self.Minv = shell_direct.Minv
        self.P    = shell_direct.P
        self.iter = 1

    def mult(self, mat, x, y):
        """y <- D* x = P^T M^-1 L^-1 Q (L*)^-1 P x"""
        v1, tmp = self.Q.getVecs()
        v2, tmp = self.Q.getVecs()
        self.P.mult(x, v1)
        v1.conjugate()
        self.ksp.solveTranspose(v1, v2)
        v2.conjugate()
        self.Q.mult(v2, v1)
        self.ksp.solve(v1, v2)
        self.Minv.mult(v2, v1)
        self.P.multTranspose(v1, y)
        self.iter += 1
        v1.destroy(); v2.destroy(); tmp.destroy()

    def cleanup(self):
        self.ksp = None; self.Q = None; self.Minv = None; self.P = None


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_sensitivity_field(direct_vec, adjoint_vec, B, nvars, n, neq,
                               beta, dreduced, rgid, outfile, rank):
    """
    Compute and save the structural sensitivity field.
    sensitivity_i(x) = |q_adj_i(x)| * |q_dir_i(x)| / |<q_adj_i, B, q_dir_i>|
    """
    Print = PETSc.Sys.Print

    tmp_vec, _ = B.getVecs()
    B.mult(direct_vec, tmp_vec)

    scatter_d, dir_seq = PETSc.Scatter.toZero(direct_vec)
    scatter_a, adj_seq = PETSc.Scatter.toZero(adjoint_vec)
    scatter_t, tmp_seq = PETSc.Scatter.toZero(tmp_vec)
    scatter_d.scatter(direct_vec,  dir_seq,  False, PETSc.Scatter.Mode.FORWARD)
    scatter_a.scatter(adjoint_vec, adj_seq,  False, PETSc.Scatter.Mode.FORWARD)
    scatter_t.scatter(tmp_vec,     tmp_seq,  False, PETSc.Scatter.Mode.FORWARD)

    if rank == 0:
        dir_arr = dir_seq.getArray()
        adj_arr = adj_seq.getArray()
        tmp_arr = tmp_seq.getArray()

        inner = np.dot(np.conj(adj_arr), tmp_arr)
        norm  = abs(inner)
        if norm < 1.0e-30:
            Print(' WARNING: adjoint-direct inner product near zero')
            norm = 1.0

        sensitivity = np.abs(adj_arr) * np.abs(dir_arr) / norm
        wm_vec_np   = sensitivity.astype('c16')
        wm_petsc    = PETSc.Vec().createSeq(len(wm_vec_np))
        wm_petsc.setArray(wm_vec_np)
        mode2pval(outfile, wm_petsc, nvars, n, neq, beta, dreduced, rgid)
        wm_petsc.destroy()

    scatter_d.destroy(); dir_seq.destroy()
    scatter_a.destroy(); adj_seq.destroy()
    scatter_t.destroy(); tmp_seq.destroy()
    tmp_vec.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_slices(params):

    comm  = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank  = comm.Get_rank()
    Print = PETSc.Sys.Print

    t_total = time.time()

    # ── Unpack parameters ────────────────────────────────────────────────────
    input_path          = params['input_path']
    output_path         = params['output_path']
    mach                = params['mach']
    beta                = params['beta']
    nslices             = params['nslices']
    slice_spacing       = params['slice_spacing']
    rlength             = params['rlength']
    listomegas          = params['listomegas']
    nev                 = params['nev']
    ncv                 = params['ncv']
    shift               = params['shift']
    tol                 = params['tol']
    max_it              = params['max_it']
    adjoint             = params['adjoint']
    compute_sensitivity = params['compute_sensitivity']
    dreduced            = params['dreduced']
    xmin                = params['xmin'];  xmax = params['xmax']
    zmin                = params['zmin'];  zmax = params['zmax']

    jacfile = os.path.join(input_path, params['jac_file'])
    volfile = os.path.join(input_path, params['vol_file'])
    coofile = os.path.join(input_path, params['coord_file'])

    if rank == 0:
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
    comm.Barrier()  # all ranks wait until rank 0 has created the output dir

    # ── Print run summary ────────────────────────────────────────────────────
    Print('')
    Print(' ========================================')
    Print('  RESOLVENT ANALYSIS — FLORES')
    Print(' ========================================')
    Print(' Input path  : {0}'.format(input_path))
    Print(' Output path : {0}'.format(output_path))
    Print(' Jacobian    : {0}'.format(jacfile))
    Print(' Mach        : {0}'.format(mach))
    Print(' beta        : {0}'.format(beta))
    Print(' nev         : {0}'.format(nev))
    Print(' ncv         : {0}'.format(ncv))
    Print(' shift       : {0}'.format(shift))
    Print(' tol / max_it: {0} / {1}'.format(tol, max_it))
    Print(' Adjoint     : {0}'.format(adjoint))
    Print(' Sensitivity : {0}'.format(compute_sensitivity))
    Print(' Dom. reduc. : {0}'.format(dreduced))
    if dreduced:
        Print(' XMIN/XMAX   : {0} / {1}'.format(xmin, xmax))
        Print(' ZMIN/ZMAX   : {0} / {1}'.format(zmin, zmax))
    Print(' Frequencies : {0}'.format(listomegas))
    Print(' MPI ranks   : {0}'.format(nproc))
    Print(' OMP threads : {0}'.format(os.environ.get('OMP_NUM_THREADS', '1')))
    Print(' ========================================')
    Print('')

    # ── Read Jacobian ────────────────────────────────────────────────────────
    t0 = time.time()
    Print(' Reading Jacobian')
    if rank == 0:
        mjac, neq = openjacobian(jacfile)
        nvars     = mjac.shape[0]
        Print(' Matrix main dimension = {0}'.format(nvars))
        Print(' Number of equations   = {0}'.format(neq))
        Print('')

        if beta == 0:
            amatrix = mjac
        else:
            nvars //= nslices
            Print(' J0 block main dimension = {0}'.format(nvars))
            # Central slice c couples to its neighbours c-1 and c+1.
            # With q_{c+-1} = q_c exp(+-i beta dy), the row block of slice c
            # collapses to  J_{c,c-1} e^{-i beta dy} + J_{c,c} + J_{c,c+1} e^{+i beta dy}
            dy = slice_spacing
            c  = nslices // 2
            Print(' Extracting and compacting Jacobian (central slice {0},'
                  ' dy = {1})'.format(c, dy))
            midrow  = mjac[nvars*c:nvars*(c+1), :].tocsc()
            jm1     = midrow[:, nvars*(c-1):nvars*c]
            j0      = midrow[:, nvars*c:nvars*(c+1)]
            j1      = midrow[:, nvars*(c+1):nvars*(c+2)]
            amatrix = (j0 + j1*np.exp(1j*beta*dy)
                          + jm1*np.exp(-1j*beta*dy))
            amatrix = amatrix.tocsr()
            del midrow, jm1, j0, j1
        del mjac

        n   = amatrix.shape[0]
        nnz = amatrix.nnz
        meta = np.array([neq, nvars, n, nnz], dtype=np.int64)
    else:
        meta = np.empty(4, dtype=np.int64)

    comm.Bcast(meta, root=0)
    neq, nvars, n, nnz = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])

    gridpoints = n // neq
    _t(comm, rank, 'Jacobian read', t0)

    # ── Mass matrix ──────────────────────────────────────────────────────────
    t0 = time.time()
    Print(' Reading mass matrix')
    if rank == 0:
        with open(volfile, 'r') as f:
            vols_buf = np.array([float(line) for line in f.readlines()],
                                dtype=np.float64)
        ngp = np.array([len(vols_buf)], dtype=np.int64)
    else:
        ngp = np.empty(1, dtype=np.int64)

    comm.Bcast(ngp, root=0)
    if rank != 0:
        vols_buf = np.empty(int(ngp[0]), dtype=np.float64)
    comm.Bcast(vols_buf, root=0)

    b_data = np.repeat(vols_buf, neq).astype(np.complex128)   # diag(M)
    del vols_buf
    _t(comm, rank, 'Mass matrix build', t0)

    # ── Domain reduction — rank 0 only; only vectors are broadcast ───────────
    t0   = time.time()
    rgid = None
    if dreduced:
        Print(' Applying domain reduction')
        Print(' XMIN/XMAX = {0}/{1}'.format(xmin, xmax))
        Print(' ZMIN/ZMAX = {0}/{1}'.format(zmin, zmax))

        if rank == 0:
            coord = read_coordinates(coofile, rlength, beta)
            dr    = domain_reduction(zmin, zmax, xmin, xmax)
            dr.create_Pmatrix(coord)

            nnz_before = amatrix.nnz
            amatrix    = dr.reduce_matrix(amatrix)
            b_data     = dr.reduce_vector(b_data)
            n_red      = amatrix.shape[0]

            localid = np.repeat(np.arange(0, gridpoints, 1, dtype='i4'), neq)
            rgid    = dr.reduce_vector(localid)[0::neq].astype(np.int64)
            del localid

            Print(' Previous NNZ = {0}'.format(nnz_before))
            Print(' New NNZ      = {0}'.format(amatrix.nnz))
            Print(' New leading dimension = {0}'.format(n_red))
            Print('')
            meta_dr = np.array([n_red, len(rgid)], dtype=np.int64)
        else:
            meta_dr = np.empty(2, dtype=np.int64)

        comm.Bcast(meta_dr, root=0)
        n_red, ngrid_red = int(meta_dr[0]), int(meta_dr[1])
        if rank != 0:
            b_data = np.empty(n_red,     dtype=np.complex128)
            rgid   = np.empty(ngrid_red, dtype=np.int64)
        comm.Bcast(b_data, root=0)
        comm.Bcast(rgid,   root=0)

        n    = n_red
        rgid = rgid.astype(int)

    _t(comm, rank, 'Domain reduction', t0)

    # ── Assemble PETSc matrices ───────────────────────────────────────────────
    # Rank 0 sends each rank only the rows it owns (no full-matrix copies).
    t0  = time.time()
    fac = 1. / (mach * np.sqrt(1.4))
    if rank == 0:
        amatrix.data *= fac

    A    = assemble_aij(comm, amatrix if rank == 0 else None, n)
    Print(' Assembled A (Jacobian)')
    B    = assemble_diag(b_data,       n)
    Print(' Assembled B (mass)')
    Binv = assemble_diag(1.0 / b_data, n)
    Print(' Assembled Binv (mass inverse)')
    Q    = assemble_diag(b_data,       n)     # energy weight Q = M
    Print(' Assembled Q (energy weight)')

    if rank == 0:
        del amatrix
    del b_data
    gc.collect()
    _t(comm, rank, 'PETSc matrix assembly', t0)

    # ── Frequency loop ────────────────────────────────────────────────────────
    mpd = ncv - 1

    # Size of the forcing space: two momentum components per grid point
    # (columns of the prolongation matrix P, see resolvant.pcreate)
    nforce = 2 * (n // neq)

    for i_omega, omega in enumerate(listomegas):

        t0_omega = time.time()
        Print('\n ========================================')
        Print('  omega {0}/{1} = {2}'.format(i_omega+1, len(listomegas), omega))
        Print(' ========================================')

        # ── LU factorization ─────────────────────────────────────────────────
        t0      = time.time()
        A_omega = A.copy()
        R = PETSc.Mat().create()
        R.setSizes([nforce, nforce])
        R.setType('python')
        shell = resolvant(n=n, Minv=Binv, Qe=Q, J=A_omega, w=omega, neq=neq)
        R.setPythonContext(shell)
        R.setUp()
        shell.operator(omega)
        _t(comm, rank, '  LU factorization (omega={0})'.format(omega), t0)

        # ── Direct problem ────────────────────────────────────────────────────
        t0 = time.time()
        E = SLEPc.EPS().create()
        E.setOperators(R)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        E.setType('krylovschur')
        E.setTolerances(tol=tol, max_it=max_it)
        E.setDimensions(nev, ncv, mpd)
        if shift != 0.0:
            ST = E.getST()
            ST.setShift(shift)
        E.setFromOptions()

        if rank == 0:
            print(' Target = ', E.getTarget())
            print(' nev    = ', nev, '  ncv = ', ncv)

        Print(' SOLVING DIRECT PROBLEM')
        E.solve()
        _t(comm, rank, '  Direct EPS solve', t0)

        nconv = E.getConverged()
        nsave = min(nconv, nev)
        if rank == 0:
            print(' Converged: {0}  /  Saving: {1}'.format(nconv, nsave))

        xr,      _ = shell.P.getVecs()
        xi,      _ = shell.P.getVecs()
        eigpre,  _ = A_omega.getVecs()
        resppre, _ = A_omega.getVecs()
        eigs         = []
        direct_modes = []

        # ── Save direct modes ─────────────────────────────────────────────────
        t0 = time.time()
        if nsave > 0:
            Print("           k           ||Ax-kx||/||kx|| ")
            Print("--------------------- ------------------")
            for i in range(nsave):
                k     = E.getEigenpair(i, xr, xi)
                error = E.computeError(i)
                eigs.append(k)
                if k.imag != 0.0:
                    Print(" %9f%+9f j    %12g" % (k.real, k.imag, error))
                else:
                    Print(" %12f         %12g" % (k.real, error))

                # Optimal forcing
                eigvecfile = os.path.join(output_path,
                    'eigf_{0}_{1}.pval'.format(i, omega))
                shell.P.mult(xr, eigpre)
                scatter, eigenvec = PETSc.Scatter.toZero(eigpre)
                scatter.scatter(eigpre, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
                if rank == 0:
                    mode2pval(eigvecfile, eigenvec, nvars, n, neq,
                              beta, dreduced, rgid)
                scatter.destroy(); eigenvec.destroy()

                # Optimal response
                shell.ksp.solve(eigpre, resppre)
                respvecfile = os.path.join(output_path,
                    'eigr_{0}_{1}.pval'.format(i, omega))
                scatter_r, respvec = PETSc.Scatter.toZero(resppre)
                scatter_r.scatter(resppre, respvec, False, PETSc.Scatter.Mode.FORWARD)
                if rank == 0:
                    mode2pval(respvecfile, respvec, nvars, n, neq,
                              beta, dreduced, rgid)
                scatter_r.destroy(); respvec.destroy()

                if adjoint and compute_sensitivity:
                    mode_copy, _ = A_omega.getVecs()
                    eigpre.copy(mode_copy)
                    direct_modes.append(mode_copy)

        eigpre.destroy(); resppre.destroy(); xi.destroy(); xr.destroy()
        E.destroy()
        _t(comm, rank, '  Direct mode save', t0)

        if rank == 0:
            eigv_file = os.path.join(output_path,
                'eigv_DIR_{0}.dat'.format(omega))
            with open(eigv_file, 'w') as w:
                for i in range(nsave):
                    w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(
                        i, eigs[i].real, eigs[i].imag))
            print(' Saved {0} eigenvalues -> {1}'.format(nsave, eigv_file))

        # ── Adjoint problem ───────────────────────────────────────────────────
        if adjoint and nsave > 0:

            t0 = time.time()
            Print(' SOLVING ADJOINT PROBLEM (reusing LU)')

            R_adj = PETSc.Mat().create()
            R_adj.setSizes([nforce, nforce])
            R_adj.setType('python')
            shell_adj = resolvant_adjoint(shell)
            R_adj.setPythonContext(shell_adj)
            R_adj.setUp()

            E_adj = SLEPc.EPS().create()
            E_adj.setOperators(R_adj)
            E_adj.setProblemType(SLEPc.EPS.ProblemType.NHEP)
            E_adj.setType('krylovschur')
            E_adj.setTolerances(tol=tol, max_it=max_it)
            E_adj.setDimensions(nev, ncv, mpd)
            E_adj.setFromOptions()
            E_adj.solve()
            _t(comm, rank, '  Adjoint EPS solve', t0)

            nconv_adj = E_adj.getConverged()
            nsave_adj = min(nconv_adj, nev)
            if rank == 0:
                print(' Adjoint converged: {0}  /  Saving: {1}'.format(
                    nconv_adj, nsave_adj))

            xr_adj,     _ = shell.P.getVecs()
            xi_adj,     _ = shell.P.getVecs()
            eigpre_adj, _ = A_omega.getVecs()
            eigs_adj        = []

            t0 = time.time()
            if nsave_adj > 0:
                Print(" ADJOINT MODES:")
                Print("           k*          ||Ax-kx||/||kx|| ")
                Print("--------------------- ------------------")
                for i in range(nsave_adj):
                    k_adj     = E_adj.getEigenpair(i, xr_adj, xi_adj)
                    error_adj = E_adj.computeError(i)
                    eigs_adj.append(k_adj)
                    if k_adj.imag != 0.0:
                        Print(" %9f%+9f j    %12g" % (k_adj.real, k_adj.imag, error_adj))
                    else:
                        Print(" %12f         %12g" % (k_adj.real, error_adj))

                    adjvecfile = os.path.join(output_path,
                        'eiga_{0}_{1}.pval'.format(i, omega))
                    shell.P.mult(xr_adj, eigpre_adj)
                    scatter_a, adjvec = PETSc.Scatter.toZero(eigpre_adj)
                    scatter_a.scatter(eigpre_adj, adjvec, False,
                                      PETSc.Scatter.Mode.FORWARD)
                    if rank == 0:
                        mode2pval(adjvecfile, adjvec, nvars, n, neq,
                                  beta, dreduced, rgid)
                    scatter_a.destroy(); adjvec.destroy()

                    if compute_sensitivity and i < len(direct_modes):
                        wmfile = os.path.join(output_path,
                            'sensitivity_{0}_{1}.pval'.format(i, omega))
                        compute_sensitivity_field(
                            direct_modes[i], eigpre_adj, B,
                            nvars, n, neq, beta, dreduced, rgid,
                            wmfile, rank)
                        if rank == 0:
                            print(' Saved sensitivity {0} -> {1}'.format(i, wmfile))

            if rank == 0:
                eigv_adj_file = os.path.join(output_path,
                    'eigv_ADJ_{0}.dat'.format(omega))
                with open(eigv_adj_file, 'w') as w:
                    for i in range(nsave_adj):
                        w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(
                            i, eigs_adj[i].real, eigs_adj[i].imag))
                print(' Saved {0} adjoint eigenvalues -> {1}'.format(
                    nsave_adj, eigv_adj_file))

            eigpre_adj.destroy(); xr_adj.destroy(); xi_adj.destroy()
            R_adj.destroy(); E_adj.destroy()
            shell_adj.cleanup()
            _t(comm, rank, '  Adjoint mode save', t0)

        # ── Cleanup omega ─────────────────────────────────────────────────────
        for v in direct_modes:
            v.destroy()
        direct_modes.clear()

        shell.cleanup()
        R.destroy()
        A_omega.destroy()
        gc.collect()

        _t(comm, rank, 'TOTAL omega={0}'.format(omega), t0_omega)

    # ── Total ─────────────────────────────────────────────────────────────────
    _t(comm, rank, 'TOTAL (all frequencies)', t_total)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python RESOLVANT.py <control_file.ini>')
        print('       mpirun -n 4 python RESOLVANT.py <control_file.ini>')
        sys.exit(1)

    ctrl_file = sys.argv[1]
    params    = read_control_file(ctrl_file)
    run_slices(params)