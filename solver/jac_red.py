#! /usr/bin/env python

import sys
import numpy as np
from scipy.sparse import coo_matrix


class domain_reduction(object):
    """
    Restricts a matrix/vector to a spatial subdomain defined by
    [xmin, xmax] x [zmin, zmax].

    Optimised version: all O(n) Python loops replaced with vectorised
    numpy operations; matrix reduction uses direct index slicing instead
    of two full sparse matrix multiplications.
    """

    def __init__(self, zmin, zmax, xmin, xmax):
        super(domain_reduction, self).__init__()
        self.zmin = zmin
        self.zmax = zmax
        self.xmin = xmin
        self.xmax = xmax
        # kept_idx  : indices of DOFs inside the subdomain (sorted)
        # reorder   : full permutation index array (inside first, outside after)
        self.kept_idx = None
        self.reorder  = None
        self.m        = None   # number of DOFs inside subdomain

    def create_Pmatrix(self, coords):
        """
        Build the index arrays that define the subdomain restriction.

        Instead of constructing an explicit permutation matrix and storing
        PO / POT (which cost memory and require two O(nnz) sparse multiplies
        in reduce_matrix), we store only the integer index arrays and use
        direct CSR row/column slicing.

        Parameters
        ----------
        coords : ndarray, shape (nvar, ndim)
            Coordinate array as returned by read_coordinates.
            coords[:, 0] = x,  coords[:, 1] = z
        """
        nvar = coords.shape[0]

        # ── Vectorised mask — replaces the Python for-loop ────────────────
        x = coords[:, 0]
        z = coords[:, 1]
        inside = (x > self.xmin) & (x < self.xmax) & \
                 (z > self.zmin) & (z < self.zmax)

        # Indices of DOFs inside and outside the subdomain
        self.kept_idx = np.where(inside)[0]          # shape (m,)
        outside_idx   = np.where(~inside)[0]         # shape (nvar-m,)

        # Full permutation order: inside first, then outside
        # (kept for reduce_vector compatibility)
        self.reorder = np.concatenate([self.kept_idx, outside_idx])

        self.m = int(self.kept_idx.size)
        print(' New dim = ', self.m, ' Current dim = ', nvar)
        print('')

    def reduce_matrix(self, A):
        """
        Extract the submatrix of A corresponding to the subdomain DOFs.

        Uses direct CSR row slicing followed by CSC column slicing — much
        faster than two full sparse matrix multiplications (PO * A * POT).

        Parameters
        ----------
        A : scipy sparse matrix (any format)

        Returns
        -------
        scipy.sparse.csr_matrix of shape (m, m)
        """
        idx = self.kept_idx
        # Row slicing is O(m + nnz_selected) on CSR
        # Column slicing is O(m + nnz_selected) on CSC
        return A.tocsr()[idx, :].tocsc()[:, idx].tocsr()

    def reduce_vector(self, V):
        """
        Return only the entries of V corresponding to subdomain DOFs.

        Parameters
        ----------
        V : 1-D array of length nvar

        Returns
        -------
        1-D array of length m
        """
        return np.asarray(V)[self.kept_idx]