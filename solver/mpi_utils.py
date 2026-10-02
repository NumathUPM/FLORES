#! /usr/bin/env python
"""
MPI helpers shared by eig_solver.py and resolvent_solver.py.

The Jacobian is read on rank 0 only. Instead of broadcasting the whole CSR
matrix to every rank, rank 0 sends each rank just the block of rows it owns
in the PETSc layout, so the memory per rank scales with 1/nproc.
"""

import numpy as np
from petsc4py import PETSc


def ownership_ranges(n, nproc):
    """
    Row ranges [(rstart, rend), ...] of each rank for a global size n.

    Same split as PETSc's default (PetscSplitOwnership): the first n % nproc
    ranks get one extra row.
    """
    base, extra = divmod(n, nproc)
    sizes  = [base + (1 if r < extra else 0) for r in range(nproc)]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    return [(int(starts[r]), int(starts[r + 1])) for r in range(nproc)]


def scatter_csr_rows(comm, csr, n):
    """
    Distribute the rows of a CSR matrix held on rank 0.

    Parameters
    ----------
    comm : mpi4py communicator
    csr  : scipy.sparse.csr_matrix of shape (n, n) on rank 0, None elsewhere
    n    : global number of rows

    Returns
    -------
    (rstart, rend), indptr, indices, data
        Local row range and local CSR arrays (indptr starts at 0) in
        PETSc.IntType / PETSc.ScalarType.
    """
    rank  = comm.Get_rank()
    nproc = comm.Get_size()
    ranges = ownership_ranges(n, nproc)

    def _block(rs, re):
        p0, p1  = int(csr.indptr[rs]), int(csr.indptr[re])
        indptr  = (csr.indptr[rs:re + 1] - p0).astype(PETSc.IntType)
        indices = csr.indices[p0:p1].astype(PETSc.IntType)
        data    = csr.data[p0:p1].astype(PETSc.ScalarType)
        return indptr, indices, data

    if rank == 0:
        for r in range(1, nproc):
            indptr, indices, data = _block(*ranges[r])
            comm.Send(np.array([indices.size], dtype=np.int64), dest=r, tag=10)
            comm.Send(indptr,  dest=r, tag=11)
            comm.Send(indices, dest=r, tag=12)
            comm.Send(data,    dest=r, tag=13)
        indptr, indices, data = _block(*ranges[0])
    else:
        rs, re = ranges[rank]
        nnz_local = np.empty(1, dtype=np.int64)
        comm.Recv(nnz_local, source=0, tag=10)
        indptr  = np.empty(re - rs + 1,      dtype=PETSc.IntType)
        indices = np.empty(int(nnz_local[0]), dtype=PETSc.IntType)
        data    = np.empty(int(nnz_local[0]), dtype=PETSc.ScalarType)
        comm.Recv(indptr,  source=0, tag=11)
        comm.Recv(indices, source=0, tag=12)
        comm.Recv(data,    source=0, tag=13)

    return ranges[rank], indptr, indices, data


def assemble_aij(comm, csr, n):
    """
    Build a distributed PETSc AIJ matrix from a CSR matrix held on rank 0.

    The local rows are preallocated and inserted in a single call
    (MatMPIAIJSetPreallocationCSR) from the distributed CSR blocks.
    """
    (rs, re), indptr, indices, data = scatter_csr_rows(comm, csr, n)
    nloc = re - rs
    A = PETSc.Mat().createAIJ(size=((nloc, n), (nloc, n)),
                              csr=(indptr, indices, data),
                              comm=PETSc.COMM_WORLD)
    A.assemble()
    return A


def assemble_diag(diag, n):
    """
    Build a distributed diagonal PETSc matrix with the same row layout as
    assemble_aij. `diag` is the full diagonal (length n) on every rank.
    """
    comm  = PETSc.COMM_WORLD.tompi4py()
    rs, re = ownership_ranges(n, comm.Get_size())[comm.Get_rank()]
    nloc = re - rs
    M = PETSc.Mat().createAIJ(size=((nloc, n), (nloc, n)), nnz=(1, 0),
                              comm=PETSc.COMM_WORLD)
    d = PETSc.Vec().createWithArray(
        np.ascontiguousarray(diag[rs:re], dtype=PETSc.ScalarType),
        size=(nloc, n), comm=PETSc.COMM_WORLD)
    M.setDiagonal(d)
    M.assemble()
    d.destroy()
    return M
