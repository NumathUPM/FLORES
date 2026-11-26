#!/usr/bin/env python3
"""
plot_pval_eigfunction.py
-----------------
Reads a .pval NetCDF file produced by mode2pval() and plots the
eigenmode components on the 2D computational domain.

Coordinate file (default: JAC/samg.matrix.coo)
  Line 1  :  npoints_total   ndim
  Lines 2+:  x   y   (one row per DOF, neq rows per mesh node -> identical coords)
  -> unique grid nodes = npoints_total // neq

File naming convention
----------------------
  eigf_i_omega.pval      optimal forcing   (direct)
  eigr_i_omega.pval      optimal response
  eiga_i_omega.pval      adjoint mode
  sensitivity_i_omega.pval structural sensitivity (real field)

Directory auto-detection
------------------------
  Directory name contains 'resolvent' -> resolvent mode:
    plots eigf + eigr + eiga + sensitivity for each index

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COORDINATES (one of these is required)
  Default .coo file  : JAC/samg.matrix.coo  (use --jac to change directory)
  TAU mesh NetCDF    : --mesh MESH/BFS_h4_2D.taumesh
  SOD2D mesh (serial): --mesh MESH/back_facing.h5  [--bound-ids 3 4]
  SOD2D mesh (partit): --mesh MESH/back_facing_buffer-112.hdf

INPUT (one of these is required)
  Single .pval file  : python plot_pval_eigfunction.py path/to/eigf_0_1.2j.pval
  Mode sweep         : --modes 0-9          (range)
                       --modes 3            (single)
                       --modes 0 2 5        (list)
                       --modes 0-4 7        (mixed)
                     + --dir RESULTS_eig/   (directory containing the files)

WHAT TO PLOT
  --vars u w         (default: rho u w e)
  --fields eigf eigr (resolvent only; default: all four)
  --imag             (imaginary part instead of real)
  --both             (real + imaginary, 2 panels)

DOMAIN WINDOW
  --xlim -5 20       (default: full mesh extent)
  --ylim 0 1         (default: full mesh extent)

DIAGNOSTICS
  --check-mesh       (plot mesh nodes, no .pval needed)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Sweep eigen-modes 0-20 using TAU mesh
python plot_pval_eigfunction.py --modes 0-20 --dir RESULTS_eig/ \\
    --mesh MESH/BFS_h4_2D.taumesh

# Resolvent results (auto-detected from directory name)
python plot_pval_eigfunction.py --modes 0-5 --dir RESULTS_resolvent/ \\
    --mesh MESH/BFS_h4_2D.taumesh

# Only forcing + response, restrict x window
python plot_pval_eigfunction.py --modes 0-5 --dir RESULTS_resolvent/ \\
    --mesh MESH/BFS_h4_2D.taumesh --fields eigf eigr --xlim -2 15

# Single file, both real and imaginary
python plot_pval_eigfunction.py RESULTS_eig/eigf_3_1.2j.pval \\
    --mesh MESH/BFS_h4_2D.taumesh --both

# Sanity-check mesh geometry
python plot_pval_eigfunction.py --check-mesh --mesh MESH/BFS_h4_2D.taumesh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
from scipy.spatial import cKDTree
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — fastest for file output
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset

# ── defaults ───────────────────────────────────────────────────────────────────

DEFAULT_JAC    = 'JAC'
DEFAULT_COORDS = 'JAC/samg.matrix.coo'
DEFAULT_MESH   = None
XLIM           = None
YLIM           = None

# ── variable maps ──────────────────────────────────────────────────────────────

# Note: 'Umag' is computed as sqrt(u^2 + w^2), not read directly from file
VAR_MAP = {
    'rho':   ('rho',   'rho_i'),
    'u':     ('u',     'u_i'),
    'w':     ('w',     'w_i'),
    'e':     ('e',     'e_i'),
    'v':     ('v',     'v_i'),
    'turb1': ('turb1', 'turb1_i'),
    'turb2': ('turb2', 'turb2_i'),
}

LABELS = {
    'Umag': r'$|\hat{U}|$',
    'rho':   r'$\hat{\rho}$',
    'u':     r'$\hat{u}$',
    'w':     r'$\hat{w}$',
    'e':     r'$\hat{e}$',
    'v':     r'$\hat{v}$',
    'turb1': r'$\hat{\nu}_1$',
    'turb2': r'$\hat{\nu}_2$',
}

ALL_PREFIXES = [
    ('eigf',        'Forcing'),
    ('eigr',        'Response'),
    ('eiga',        'Adjoint'),
    ('sensitivity', 'Sensitivity'),
]

PREFIX_LABEL = dict(ALL_PREFIXES)

# Labels for single-file (non-resolvent) eigenmode plots
SINGLE_PREFIX_LABEL = {
    'eigf':        'Direct',
    'eiga':        'Adjoint',
    'eigr':        'Response',
    'sensitivity': 'Sensitivity',
}

SKIP_VARS_FOR = {
    'eiga':        {'rho', 'e'},
    'sensitivity': {'rho', 'e'},
}

# ── coordinate loader (.coo) ───────────────────────────────────────────────────

def peek_pval_gridpoints(path):
    """Read gridpoints from a .pval file without loading all data."""
    try:
        with Dataset(path, 'r') as ds:
            return len(ds.dimensions['no_of_points']) // 2
    except Exception:
        return None


def load_coo(path, neq):
    print(f"  Reading: {path}")
    with open(path, 'r') as fh:
        header        = fh.readline().split()
        npoints_total = int(header[0])
        ndim          = int(header[1])
        gridpoints    = npoints_total // neq

        print(f"    npoints_total = {npoints_total}   ndim = {ndim}   neq = {neq}")
        print(f"    -> unique nodes = {gridpoints}")

        if npoints_total % neq != 0:
            print(f"  WARNING: {npoints_total} % {neq} = {npoints_total % neq} "
                  "(not perfectly divisible; last partial node ignored)")

        x = np.empty(gridpoints, dtype=np.float64)
        y = np.empty(gridpoints, dtype=np.float64)

        node = 0
        for line_idx, line in enumerate(fh):
            if line_idx % neq == 0:
                vals    = line.split()
                x[node] = float(vals[0])
                y[node] = float(vals[1])
                node   += 1
                if node >= gridpoints:
                    break

    if node < gridpoints:
        print(f"  WARNING: header announces {gridpoints} nodes but the file "
              f"only contains {node}; using those {node}.")
        x = x[:node]
        y = y[:node]

    print(f"    Loaded {node} nodes   "
          f"x∈[{x.min():.4g}, {x.max():.4g}]   "
          f"y∈[{y.min():.4g}, {y.max():.4g}]")
    return x, y

# ── coordinate loader (TAU mesh) ───────────────────────────────────────────────

def _extract_body_contour(quads, bmark_quads, x_all, z_all, n):
    """
    Extract contour lines for boundary markers that represent physical walls
    (not the 2D solver-plane faces).

    Strategy: the solver-plane faces are the markers with ALL nodes < n.
    All other markers are physical boundaries. For those, we project their
    nodes onto the solver plane by taking their (xc, zc) coordinates directly
    — the contour lives in the xz plane regardless of which copy of the node
    is referenced.

    Returns a list of (N,2) arrays, one per physical boundary marker.
    """
    # Identify solver-plane markers (all nodes < n) — these are the domain faces
    plane_markers = set()
    for bm in np.unique(bmark_quads):
        idx   = np.where(bmark_quads == bm)[0]
        nodes = np.unique(quads[idx])
        if (nodes < n).all():
            plane_markers.add(bm)

    # Physical boundary markers = all others
    body_markers = [bm for bm in np.unique(bmark_quads)
                    if bm not in plane_markers]

    if not body_markers:
        return []

    contours    = []
    seen_coords = set()   # deduplicate identical contours
    for bm in body_markers:
        idx   = np.where(bmark_quads == bm)[0]
        faces = quads[idx]   # (n_faces, 4), indices into full x_all/z_all

        # For each face, keep only edges where both nodes are on the same
        # z-layer (either both < n or both >= n), then map to solver-plane coords
        # using x_all[node], z_all[node].
        edge_count = {}
        for f in faces:
            for i in range(4):
                a, b = int(f[i]), int(f[(i+1) % 4])
                # Only keep edges within the same z-layer
                if (a < n) == (b < n):
                    edge = (min(a,b), max(a,b))
                    edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear once
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            boundary_edges = list(edge_count.keys())
        if not boundary_edges:
            continue

        # Each boundary edge becomes an independent segment [pt_a, pt_b]
        # using NaN separators so matplotlib draws them as disconnected lines
        # in a single ax.plot() call — no walker needed, no spurious lines.
        seg_x = []
        seg_y = []
        for a, b in boundary_edges:
            seg_x += [x_all[a], x_all[b], np.nan]
            seg_y += [z_all[a], z_all[b], np.nan]

        if seg_x:
            coords = np.column_stack((seg_x, seg_y))
            # Deduplicate by edge count + bounding box
            key = (len(boundary_edges), bm)
            if key not in seen_coords:
                seen_coords.add(key)
                contours.append(coords)

    return contours


def _remap_triangulation(x_coo, y_coo, x_tau, y_tau, triang_tau):
    """
    Remap a TAU triangulation to the .coo node ordering.

    Handles meshes in different units (e.g. TAU in dimensional units,
    .coo in adimensional units) by auto-detecting a uniform scale factor
    from the bounding box ratio before building the KDTree.

    Returns a new Triangulation in .coo node space.
    """
    print(f"  Remapping TAU triangulation to .coo node order …", flush=True)

    # ── Auto-detect scale factor ──────────────────────────────────────────────
    range_coo_x = x_coo.max() - x_coo.min()
    range_tau_x = x_tau.max() - x_tau.min()
    range_coo_y = y_coo.max() - y_coo.min()
    range_tau_y = y_tau.max() - y_tau.min()

    scale_x = range_coo_x / range_tau_x if range_tau_x > 0 else 1.0
    scale_y = range_coo_y / range_tau_y if range_tau_y > 0 else 1.0

    # Use uniform scale if x and y ratios agree within 1%
    if abs(scale_x - scale_y) / max(scale_x, 1e-10) < 0.01:
        scale = (scale_x + scale_y) / 2.0
    else:
        scale = 1.0   # fallback: no scaling

    if abs(scale - 1.0) > 0.01:
        print(f"    Auto-scale TAU → .coo: {scale:.6g}")
        # Scale TAU coords to .coo space
        x_tau_s = x_tau * scale + (x_coo.min() - x_tau.min() * scale)
        y_tau_s = y_tau * scale + (y_coo.min() - y_tau.min() * scale)
    else:
        x_tau_s = x_tau
        y_tau_s = y_tau

    # ── KDTree remap ──────────────────────────────────────────────────────────
    tree = cKDTree(np.column_stack((x_tau_s, y_tau_s)))
    dists, tau_to_coo = tree.query(np.column_stack((x_coo, y_coo)))

    max_dist = dists.max()
    print(f"    Max mapping distance: {max_dist:.2e}")
    if max_dist > 1e-3:
        print(f"    WARNING: large mapping distance — meshes may be incompatible")

    n_tau = len(x_tau)
    tau2coo = np.full(n_tau, -1, dtype=np.int64)
    tau2coo[tau_to_coo] = np.arange(len(x_coo))

    tris_tau = triang_tau.triangles
    tris_coo = tau2coo[tris_tau]

    valid    = (tris_coo >= 0).all(axis=1)
    tris_coo = tris_coo[valid]
    print(f"    Triangles: {len(tris_tau)} → {len(tris_coo)} valid after remap")

    return tri.Triangulation(x_coo, y_coo, tris_coo)


def _detect_solver_marker(tris, quads, bmark_tris, bmark_quads, n):
    """
    Find the boundary marker whose faces cover exactly n unique nodes,
    all with index < n.  This is the solver-plane marker.
    """
    candidates = {}
    for bm in np.unique(np.concatenate([bmark_tris, bmark_quads])):
        ti = np.where(bmark_tris  == bm)[0]
        qi = np.where(bmark_quads == bm)[0]
        nodes = []
        if ti.size: nodes.append(tris[ti].ravel())
        if qi.size: nodes.append(quads[qi].ravel())
        all_nodes = np.unique(np.concatenate(nodes))
        if (all_nodes < n).all():
            candidates[bm] = len(all_nodes)

    if not candidates:
        raise ValueError("Could not auto-detect solver-plane marker. "
                         "Use --face-marker N to specify it manually.")

    # Pick the marker with the most nodes (= full solver plane)
    best = max(candidates, key=candidates.get)
    return best


def load_taumesh(path, face_marker=None):
    """
    Read coordinates and 2D connectivity from a TAU NetCDF mesh file.

    Coordinate mapping (confirmed empirically, Matlab view(0,0) = xz plane):
      solver x = points_xc,  solver y = points_zc

    Connectivity: boundary marker 2 contains both triangles (from prisms) and
    quads (from hexas) covering exactly the solver-plane nodes.
    Node indices are 1-based in the file → subtract 1 for 0-based Python.

    Returns
    -------
    x, y   : coordinate arrays (xc, zc), length = no_of_points // 2
    triang : matplotlib.tri.Triangulation from surface connectivity
    """
    print(f"  Reading TAU mesh: {path}  (marker={face_marker})")
    with Dataset(path, 'r') as ds:
        n_total = len(ds.dimensions['no_of_points'])
        x_all   = np.asarray(ds.variables['points_xc'][:], dtype=np.float64)
        z_all   = np.asarray(ds.variables['points_zc'][:], dtype=np.float64)
        bmark   = np.asarray(ds.variables['boundarymarker_of_surfaces'][:],
                             dtype=np.int32)
        # Surface triangles (optional — some meshes have none)
        if 'no_of_surfacetriangles' in ds.dimensions and            len(ds.dimensions['no_of_surfacetriangles']) > 0:
            n_stri = len(ds.dimensions['no_of_surfacetriangles'])
            tris   = np.asarray(ds.variables['points_of_surfacetriangles'][:],
                                dtype=np.int64)
        else:
            n_stri = 0
            tris   = np.empty((0, 3), dtype=np.int64)
        quads   = np.asarray(ds.variables['points_of_surfacequadrilaterals'][:],
                             dtype=np.int64)

    n = n_total // 2
    x = x_all[:n]
    y = z_all[:n]

    print(f"    no_of_points = {n_total}  →  using first {n} nodes")
    print(f"    x∈[{x.min():.4g}, {x.max():.4g}]   y∈[{y.min():.4g}, {y.max():.4g}]")

    # Split boundary marker array into triangle and quad parts
    bmark_tris  = bmark[:n_stri]
    bmark_quads = bmark[n_stri:]

    # Auto-detect the solver-plane marker if not specified
    if face_marker is None:
        face_marker = _detect_solver_marker(
            tris, quads, bmark_tris, bmark_quads, n)
        print(f"    Auto-detected solver-plane marker: {face_marker}")

    # Select faces on the requested marker
    tri_idx  = np.where(bmark_tris  == face_marker)[0]
    quad_idx = np.where(bmark_quads == face_marker)[0]

    if tri_idx.size == 0 and quad_idx.size == 0:
        available = np.unique(bmark).tolist()
        raise ValueError(
            f"Boundary marker {face_marker} not found. "
            f"Available markers: {available}."
        )

    face_tris  = tris[tri_idx]    # (n_tris_marker, 3)
    face_quads = quads[quad_idx]  # (n_quads_marker, 4)
    face_quads_solver = face_quads.copy() if len(face_quads) > 0 else None

    print(f"    Marker {face_marker}: {len(tri_idx)} tris + {len(quad_idx)} quads")

    # Split quads into 2 triangles each: [a,b,c,d] -> [a,b,c] and [a,c,d]
    quad_tris = np.vstack([face_quads[:, [0, 1, 2]],
                           face_quads[:, [0, 2, 3]]])

    # Combine all triangles
    parts = [a for a in [face_tris, quad_tris] if len(a) > 0]
    all_tris = np.vstack(parts) if parts else np.empty((0,3), dtype=np.int64)

    print(f"    Total elements: {len(all_tris)}", flush=True)
    triang = tri.Triangulation(x, y, all_tris)

    # Extract geometry contour: the boundary marker with fewest quads
    # whose nodes are all in the solver plane (these are solid walls)
    body_contour = _extract_body_contour(
        quads, bmark_quads, x_all, z_all, n)
    if body_contour:
        for c in body_contour:
            print(f"    Body contour: {len(c)} pts  "
                  f"x∈[{c[:,0].min():.3g}, {c[:,0].max():.3g}]  "
                  f"y∈[{c[:,1].min():.3g}, {c[:,1].max():.3g}]")
    else:
        print(f"    Body contour: NOT FOUND")

    return x, y, triang, body_contour, face_quads_solver



def load_sod2dmesh(path, bound_ids=None):
    """
    Read a SOD2D HDF5 mesh (datasets coords, connec, boundFaces, boundFacesId)
    and build the 2D triangulation of ONE z-plane (the lowest one).

    The mesh is high order: each hexahedron has (p+1)^3 nodes. The nodes of an
    element lying on the plane are triangulated locally (Delaunay of its
    (p+1)^2 nodes), so the result does not depend on the node ordering inside
    the element. The body contour is built from the boundary faces
    (boundFaces): their nodes on the plane form lines, grouped by boundFacesId.

    Returns the same tuple as load_taumesh:
      x, y, triang, body_contour, None
    with x, y the coordinates of the plane nodes, in the mesh ordering.
    """
    from scipy.spatial import Delaunay
    try:
        import h5py
    except ImportError:
        raise ImportError('Reading a SOD2D mesh requires h5py (pip install h5py)')

    print(f"  Reading SOD2D mesh: {path}")
    with h5py.File(path, 'r') as f:
        coords = np.asarray(f['coords'][:], dtype=np.float64)
        connec = np.asarray(f['connec'][:], dtype=np.int64)
        bfaces = (np.asarray(f['boundFaces'][:], dtype=np.int64)
                  if 'boundFaces' in f else None)
        bid    = (np.asarray(f['boundFacesId'][:], dtype=np.int64)
                  if 'boundFacesId' in f else None)

    nn  = len(coords)
    npe = connec.shape[1]
    # node numbering: 0-based or 1-based (Fortran)
    base = 1 if connec.max() >= nn else 0
    if base:
        connec = connec - 1
        if bfaces is not None:
            bfaces = bfaces - 1
    print(f"    nodes = {nn}   elements = {len(connec)}   nodes/element = {npe}   "
          f"index base = {base}")

    z    = coords[:, 2]
    zmin = float(z.min())
    ztol = 1e-6 * max(1.0, float(np.ptp(z)))
    on_plane = np.abs(z - zmin) <= ztol

    plane_nodes = np.flatnonzero(on_plane)
    new_index   = np.full(nn, -1, dtype=np.int64)
    new_index[plane_nodes] = np.arange(len(plane_nodes))
    x = coords[plane_nodes, 0]
    y = coords[plane_nodes, 1]
    print(f"    plane z = {zmin:.6g}: {len(plane_nodes)} nodes   "
          f"x∈[{x.min():.4g}, {x.max():.4g}]   y∈[{y.min():.4g}, {y.max():.4g}]")

    # nodes per element on the plane: (p+1)^2 = npe^(2/3)
    npl  = int(round(npe ** (2.0 / 3.0)))
    mask = on_plane[connec]
    elems = np.flatnonzero(mask.sum(axis=1) == npl)
    if elems.size == 0:
        raise ValueError(f"No element has {npl} nodes on the plane z={zmin}")
    print(f"    elements touching the plane: {elems.size}   "
          f"({npl} nodes each) -> triangulating", flush=True)

    tris = []
    for e in elems:
        ids = connec[e][mask[e]]
        loc = Delaunay(np.column_stack((coords[ids, 0], coords[ids, 1]))).simplices
        tris.append(new_index[ids][loc])
    tris = np.vstack(tris)

    # drop degenerate triangles
    a  = np.column_stack((x[tris[:, 1]] - x[tris[:, 0]], y[tris[:, 1]] - y[tris[:, 0]]))
    b  = np.column_stack((x[tris[:, 2]] - x[tris[:, 0]], y[tris[:, 2]] - y[tris[:, 0]]))
    ok = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-14
    tris = tris[ok]
    print(f"    Total triangles: {len(tris)}")
    triang = tri.Triangulation(x, y, tris)

    # ── body contour from the boundary faces ────────────────────────────────
    contours = []
    if bfaces is not None and bid is not None and len(bfaces) > 0:
        nline = int(round(npl ** 0.5))          # nodes per edge = p+1
        segs  = {}
        for face, code in zip(bfaces, bid):
            ids = face[on_plane[face]]
            if len(ids) != nline:               # face not crossing the plane
                continue
            pts = np.column_stack((coords[ids, 0], coords[ids, 1]))
            # order the nodes along the line (farthest pair gives the direction)
            d   = pts[:, None, :] - pts[None, :, :]
            i, j = np.unravel_index(np.argmax((d ** 2).sum(-1)), (nline, nline))
            t   = (pts - pts[i]) @ (pts[j] - pts[i])
            pts = pts[np.argsort(t)]
            segs.setdefault(int(code), []).append(
                np.vstack((pts, [np.nan, np.nan])))   # NaN breaks the polyline
        print("    Boundary ids on the plane (use --bound-ids to select):")
        for code in sorted(segs):
            c = np.vstack(segs[code])
            print(f"      id {code:4d}: {len(segs[code]):5d} faces   "
                  f"x∈[{np.nanmin(c[:,0]):.4g}, {np.nanmax(c[:,0]):.4g}]   "
                  f"y∈[{np.nanmin(c[:,1]):.4g}, {np.nanmax(c[:,1]):.4g}]")
            if bound_ids is None or code in bound_ids:
                contours.append(c)
    else:
        print("    Body contour: no boundFaces in the file")

    return x, y, triang, contours, None


def load_sod2dmesh_partitioned(path):
    """
    Read a partitioned SOD2D mesh (.hdf, e.g. back_facing_buffer-112.hdf, one
    piece per MPI rank) through its /VTKHDF group: linear hexahedra obtained
    by subdividing the high-order elements (isLinealOutput).

    The lowest z plane is extracted: every hexahedron face lying on it gives a
    quad, split into two triangles. The pieces of the different ranks are
    merged by coordinates (nodes shared between ranks are duplicated in the
    file). The body contour is the free boundary of that 2D triangulation
    (edges used by one triangle), so it includes the outer domain boundary:
    use --xlim/--ylim to look at the region of interest.

    Returns the same tuple as load_taumesh:
      x, y, triang, body_contour, None
    """
    try:
        import h5py
    except ImportError:
        raise ImportError('Reading a SOD2D mesh requires h5py (pip install h5py)')

    print(f"  Reading partitioned SOD2D mesh: {path}")
    with h5py.File(path, 'r') as f:
        if 'VTKHDF' not in f:
            raise ValueError("No /VTKHDF group in the file: it is not a "
                             "partitioned SOD2D mesh")
        g      = f['VTKHDF']
        pts    = np.asarray(g['Points'][:], dtype=np.float64)
        conn   = np.asarray(g['Connectivity'][:], dtype=np.int64)
        types  = np.asarray(g['Types'][:])
        npts_r = np.asarray(g['NumberOfPoints'][:], dtype=np.int64)
        ncel_r = np.asarray(g['NumberOfCells'][:], dtype=np.int64)
        ncon_r = np.asarray(g['NumberOfConnectivityIds'][:], dtype=np.int64)

    nranks = len(npts_r)
    print(f"    ranks = {nranks}   points = {len(pts)}   cells = {ncel_r.sum()}")
    if not (types == 12).all() or ncon_r.sum() != 8 * ncel_r.sum():
        raise ValueError("Only linear hexahedra (VTK type 12) are supported: "
                         "the mesh must be written with isLinealOutput = 1")

    pstart = np.concatenate(([0], np.cumsum(npts_r)))
    cstart = np.concatenate(([0], np.cumsum(ncel_r)))
    nstart = np.concatenate(([0], np.cumsum(ncon_r)))

    # connectivity ids are local to each piece (VTKHDF); tolerate global ids
    cells = np.empty((int(ncel_r.sum()), 8), dtype=np.int64)
    for r in range(nranks):
        c = conn[nstart[r]:nstart[r + 1]]
        if r > 0 and c.size and c.min() >= pstart[r]:
            off = 0                         # already global
        else:
            off = pstart[r]
        cells[cstart[r]:cstart[r + 1]] = (c + off).reshape(-1, 8)

    z    = pts[:, 2]
    zmin = float(z.min())
    ztol = 1e-6 * max(1.0, float(np.ptp(z)))
    on_plane = np.abs(z - zmin) <= ztol

    # faces of a VTK hexahedron (cyclic node order)
    faces = ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])
    quads = []
    for fc in faces:
        q = cells[:, fc]
        quads.append(q[on_plane[q].all(axis=1)])
    quads = np.vstack(quads)
    if quads.shape[0] == 0:
        raise ValueError(f"No hexahedron face lies on the plane z={zmin}")
    print(f"    plane z = {zmin:.6g}: {quads.shape[0]} quads")

    # merge the nodes shared between ranks (same x,y)
    used = np.unique(quads)
    xy   = pts[used, :2]
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    key  = np.round(xy / (1e-9 * span)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.ravel(inv)
    x = xy[first, 0]
    y = xy[first, 1]
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = inv
    quads = remap[quads]
    print(f"    plane nodes after merging ranks: {len(x)}   "
          f"x∈[{x.min():.4g}, {x.max():.4g}]   y∈[{y.min():.4g}, {y.max():.4g}]")

    tris = np.vstack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    a  = np.column_stack((x[tris[:, 1]] - x[tris[:, 0]], y[tris[:, 1]] - y[tris[:, 0]]))
    b  = np.column_stack((x[tris[:, 2]] - x[tris[:, 0]], y[tris[:, 2]] - y[tris[:, 0]]))
    tris = tris[np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-14]
    print(f"    Total triangles: {len(tris)}")
    triang = tri.Triangulation(x, y, tris)

    # body contour = free edges of the plane triangulation
    e = np.sort(np.vstack((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])), axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    free = ue[cnt == 1]
    print(f"    Free boundary edges: {len(free)}")
    nan = np.full(len(free), np.nan)
    cx = np.column_stack((x[free[:, 0]], x[free[:, 1]], nan)).ravel()
    cy = np.column_stack((y[free[:, 0]], y[free[:, 1]], nan)).ravel()
    return x, y, triang, [np.column_stack((cx, cy))], None

# ── pval reader ────────────────────────────────────────────────────────────────

def read_pval(path, vars_to_plot):
    data = {}
    with Dataset(path, 'r') as ds:
        nprob      = len(ds.dimensions['no_of_points'])
        gridpoints = nprob // 2
        avail      = set(ds.variables.keys())

        print(f"  File         : {path}")
        print(f"  no_of_points : {nprob}  ->  gridpoints = {gridpoints}")
        print(f"  Variables    : {sorted(avail)}")
        print()

        # Umag requires u and w — ensure they are loaded
        _vars = list(vars_to_plot)
        if 'Umag' in _vars:
            if 'u' not in _vars: _vars.append('u')
            if 'w' not in _vars: _vars.append('w')

        for vname in _vars:
            if vname == 'Umag':
                continue   # computed after loop
            if vname not in VAR_MAP:
                print(f"  [skip] Unknown variable '{vname}'. "
                      f"Valid names: {list(VAR_MAP.keys())}")
                continue
            rname, iname = VAR_MAP[vname]
            if rname not in avail:
                print(f"  [skip] '{rname}' not found in file.")
                continue

            real_part = np.asarray(ds.variables[rname][:gridpoints], dtype=np.float64)
            if iname and iname in avail:
                imag_part = np.asarray(ds.variables[iname][:gridpoints], dtype=np.float64)
            else:
                imag_part = np.zeros_like(real_part)

            nnan  = int(np.isnan(real_part).sum() + np.isnan(imag_part).sum())
            nzero = int((np.abs(real_part) + np.abs(imag_part) == 0).sum())
            print(f"  [{vname:6s}]  "
                  f"|real|_max = {np.abs(real_part).max():.3e}   "
                  f"|imag|_max = {np.abs(imag_part).max():.3e}   "
                  f"NaNs = {nnan}   zeros = {nzero}/{gridpoints}")

            data[vname] = real_part + 1j * imag_part

    # Compute Umag if requested
    if 'Umag' in vars_to_plot:
        compute_umag(data)
        # Remove u/w if they weren't originally requested
        for v in ['u', 'w']:
            if v not in vars_to_plot and v in data:
                del data[v]
        if 'Umag' in data:
            mag = data['Umag'].real
            print(f"  [{'Umag':6s}]  "
                  f"|mag|_max  = {mag.max():.3e}   "
                  f"NaNs = {int(np.isnan(mag).sum())}   "
                  f"zeros = {int((mag==0).sum())}/{gridpoints}")
    return gridpoints, data

# ── mesh check ─────────────────────────────────────────────────────────────────

def check_mesh(x, y, triang=None, body_contour=None, mesh_quads=None, mesh_name='mesh'):
    n = len(x)
    print(f"\n── Mesh check ──────────────────────────────────────────────────")
    print(f"  {n} nodes")
    print(f"  x ∈ [{x.min():.6g}, {x.max():.6g}]")
    print(f"  y ∈ [{y.min():.6g}, {y.max():.6g}]")

    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)

    if triang is not None:
        ax.triplot(triang, color='k', linewidth=0.2, alpha=0.5,
                   rasterized=True)
        n_elem = len(triang.triangles)
        ax.set_title(f'Mesh  —  {n} nodes  |  {n_elem} elements', fontsize=12)
    else:
        MAX_PTS = 300_000
        xs, ys = (x, y) if n <= MAX_PTS else (
            x[np.random.default_rng(0).choice(n, MAX_PTS, replace=False)],
            y[np.random.default_rng(0).choice(n, MAX_PTS, replace=False)],
        )
        ax.scatter(xs, ys, c=ys, cmap='viridis', s=1,
                   linewidths=0, rasterized=True)
        ax.set_title(f'Mesh  —  {n} nodes', fontsize=12)

    for bc in (body_contour or []):
        ax.plot(bc[:, 0], bc[:, 1], 'k-', linewidth=1.5)

    if XLIM:
        ax.set_xlim(*XLIM)
    if YLIM:
        ax.set_ylim(*YLIM)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x');  ax.set_ylabel('y')
    out = f'{os.path.splitext(mesh_name)[0]}_check.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {os.path.abspath(out)}')
    plt.close(fig)

# ── colour-map helpers ─────────────────────────────────────────────────────────

def symmetric_norm(data, pct=90):
    nonzero = np.abs(data[data != 0]) if (data != 0).any() else np.abs(data)
    vmax = float(np.nanpercentile(nonzero, pct))
    if vmax == 0:
        vmax = float(np.nanmax(np.abs(data)))
    if vmax == 0:
        vmax = 1.0
    return matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def positive_norm(data, pct=90):
    vmax = float(np.nanpercentile(data, pct))
    if vmax == 0:
        vmax = float(np.nanmax(data))
    if vmax == 0:
        vmax = 1.0
    # vmin: small positive value to avoid black background at zero
    vmin = 0.0
    return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# ── colorbar helper ────────────────────────────────────────────────────────────

def add_colorbar(ax, im):
    divider    = make_axes_locatable(ax)
    cax        = divider.append_axes('right', size='2%', pad=0.08)
    cb         = ax.get_figure().colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=8)
    cb.formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((-2, 2))   # scientific for |exp| > 2
    cb.update_ticks()
    return cb

# ── triangulation ──────────────────────────────────────────────────────────────

def build_triangulation(x, y):
    print("  Building Delaunay triangulation …", end=' ', flush=True)
    triang = tri.Triangulation(x, y)
    mask   = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio=0.01)
    triang.set_mask(mask)
    print(f"{(~mask).sum()} valid triangles.")
    return triang

# ── single-mode plot ───────────────────────────────────────────────────────────

def plot_modes(x, y, mode_data, cmap, output_stem, title_prefix,
               triang=None, clim_pct=90, plot_imag=False, plot_both=False,
               prefix=None, body_contour=None, eig_str='', omg_str='',
               field_label=None, mode_idx=None):
    if not mode_data:
        print("  Nothing to plot.")
        return

    if plot_both:
        parts  = [('Real', lambda c: c.real), ('Imag', lambda c: c.imag)]
        suffix = '_ri'
    elif plot_imag:
        parts  = [('Imag', lambda c: c.imag)]
        suffix = '_i'
    else:
        parts  = [('Real', lambda c: c.real)]
        suffix = ''

    for vname, cdata in mode_data.items():
        label    = LABELS.get(vname, vname)
        n_panels = len(parts)

        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(14, 3.2 * n_panels),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.4)

        for ax, (part_label, extractor) in zip(axes[:, 0], parts):
            arr = extractor(cdata)
            # Compute norm using only nodes within the visible window
            win_mask = ((x >= XLIM[0]) & (x <= XLIM[1]) &
                        (y >= YLIM[0]) & (y <= YLIM[1]))
            arr_win  = arr[win_mask] if win_mask.any() else arr
            # Exclude exact zeros (domain-reduced nodes) from norm computation
            arr_norm = arr_win[arr_win != 0] if (arr_win != 0).any() else arr_win
            if prefix == 'sensitivity' or vname == 'Umag':
                norm     = positive_norm(arr_norm, pct=clim_pct)
                cmap_use = 'viridis' if vname == 'Umag' else 'Greys'
            else:
                norm     = symmetric_norm(arr_norm, pct=clim_pct)
                cmap_use = cmap
            im = ax.tripcolor(triang, arr, cmap=cmap_use, norm=norm,
                              shading='gouraud', rasterized=True)

            for bc in (body_contour or []):
                ax.plot(bc[:, 0], bc[:, 1], 'k-', linewidth=0.8, zorder=5)
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)
            ax.set_aspect('equal', adjustable='box')
            add_colorbar(ax, im)
            _flabel = field_label or SINGLE_PREFIX_LABEL.get(prefix, '')
            _extra  = '   '.join(s for s in [eig_str, omg_str] if s)
            _title_parts = [p for p in [_flabel, f'{part_label} {label}', _extra] if p]
            _ax_title = ' - '.join(_title_parts)
            if mode_idx is not None:
                _ax_title += f' (mode {mode_idx})'
            ax.set_title(_ax_title, fontsize=11, loc='left')
            ax.set_xlabel('x/H', fontsize=9)
            ax.set_ylabel('y/H', fontsize=9)
            ax.tick_params(labelsize=8)

        out = f'{output_stem}_{vname}{suffix}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close(fig)

# ── resolvent plot ─────────────────────────────────────────────────────────────

def plot_resolvent(datasets, cmap, dir_path, idx_omega,
                   triang, clim_pct=90, plot_imag=False, plot_both=False,
                   body_contour=None, eig_str='', omg_str='', mode_idx=None):
    if not datasets:
        print("  Nothing to plot.")
        return

    if plot_both:
        parts  = [("Real", lambda c: c.real), ("Imag", lambda c: c.imag)]
        suffix = "_ri"
    elif plot_imag:
        parts  = [("Imag", lambda c: c.imag)]
        suffix = "_i"
    else:
        parts  = [("Real", lambda c: c.real)]
        suffix = ""

    for prefix, field_label, ddict, is_sensitivity in datasets:
        if not ddict:
            continue
        for vname, cdata in ddict.items():
            label    = LABELS.get(vname, vname)
            n_panels = len(parts)

            fig, axes = plt.subplots(
                n_panels, 1,
                figsize=(14, 3.2 * n_panels),
                squeeze=False,
            )
            fig.subplots_adjust(hspace=0.4)

            for ax, (part_label, extractor) in zip(axes[:, 0], parts):
                arr = extractor(cdata)
                win_mask = ((x >= XLIM[0]) & (x <= XLIM[1]) &
                            (y >= YLIM[0]) & (y <= YLIM[1]))
                arr_win  = arr[win_mask] if win_mask.any() else arr
                arr_norm = arr_win[arr_win != 0] if (arr_win != 0).any() else arr_win
                if prefix == 'sensitivity' or vname == 'Umag':
                    norm     = positive_norm(arr_norm, pct=clim_pct)
                    cmap_use = 'viridis' if vname == 'Umag' else 'Greys'
                else:
                    norm     = symmetric_norm(arr_norm, pct=clim_pct)
                    cmap_use = cmap
                im = ax.tripcolor(triang, arr, cmap=cmap_use, norm=norm,
                                  shading="gouraud", rasterized=True)

                for bc in (body_contour or []):
                    ax.plot(bc[:, 0], bc[:, 1], 'k-', linewidth=0.8, zorder=5)
                ax.set_xlim(*XLIM)
                ax.set_ylim(*YLIM)
                ax.set_aspect("equal", adjustable="box")
                add_colorbar(ax, im)
                _extra = "   ".join(s for s in [eig_str, omg_str] if s)
                _title_parts = [p for p in [field_label, f'{part_label} {label}', _extra] if p]
                _ax_title = ' - '.join(_title_parts)
                if mode_idx is not None:
                    _ax_title += f' (mode {mode_idx})'
                ax.set_title(_ax_title, fontsize=11, loc="left")
                ax.set_xlabel("x/H", fontsize=9)
                ax.set_ylabel("y/H", fontsize=9)
                ax.tick_params(labelsize=8)

            out = os.path.join(dir_path,
                               f"{prefix}_{idx_omega}_{vname}{suffix}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")
            plt.close(fig)

# ── helpers ────────────────────────────────────────────────────────────────────

def load_eigenvalues(search_dirs):
    """
    Load eigenvalues from eigv_DIR.dat or eigv_ADJ.dat.
    Searches each directory in search_dirs in order.
    Returns dict {index: complex} or empty dict if not found.
    """
    for d in search_dirs:
        for fname in ('eigv_DIR.dat', 'eigv_ADJ.dat'):
            path = os.path.join(d, fname)
            if not os.path.isfile(path):
                continue
            eigs = {}
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            idx = int(parts[0])
                            eigs[idx] = complex(float(parts[1]), float(parts[2]))
                        except ValueError:
                            pass
            if eigs:
                print(f"  Eigenvalues loaded from: {path} ({len(eigs)} modes)")
                return eigs
    return {}


def omega_str(idx_omega):
    """Extract omega value from idx_omega string like '0_4.102j' -> 'ω = 4.102'"""
    parts = idx_omega.split('_', 1)
    if len(parts) < 2:
        return ''
    omega_raw = parts[1].rstrip('j').rstrip('J')
    try:
        omega = float(omega_raw)
        return f"ω = {omega:.5g}"
    except ValueError:
        return f"ω = {parts[1]}"


def eigenvalue_str(eigs, idx):
    r"""Format eigenvalue for title: [-\sigma_r, \sigma_i] = [val, val]"""
    if idx not in eigs:
        return ''
    lam = eigs[idx]
    return rf"[$-\sigma_r, \sigma_i$] = [{-lam.real:.5g}, {lam.imag:.5g}]"


def compute_umag(data_dict):
    """Add Umag = sqrt(|u|^2 + |w|^2) to data_dict if u and w are present."""
    if 'Umag' in data_dict:
        return
    u = data_dict.get('u')
    w = data_dict.get('w')
    if u is not None and w is not None:
        data_dict['Umag'] = np.sqrt(np.abs(u)**2 + np.abs(w)**2).astype(np.complex128)
    elif u is not None:
        data_dict['Umag'] = np.abs(u).astype(np.complex128)


def load_volumes(vol_path, neq, n_nodes):
    """
    Load cell volumes from samg.matrix.vol and expand to DOF-level diagonal
    of mass matrix B.  Returns array of length n_nodes*neq.
    """
    with open(vol_path) as f:
        vols = np.array([float(line) for line in f], dtype=np.float64)
    # Repeat each volume neq times (one per DOF per node)
    return np.repeat(vols[:n_nodes], neq)


def compute_sensitivity(dir_data, adj_data, common, vol_path=None, neq=4):
    """
    Compute structural sensitivity S = |q_adj| * |q_dir| / |<q_adj, B q_dir>|
    following Giannetti & Luchini (2007).

    B is the diagonal mass matrix (cell volumes repeated neq times).
    If vol_path is None, uses identity (no mass weighting).

    The inner product is computed over all variables jointly (full DOF vector),
    matching eig_solver.py which uses the full Jacobian-sized vectors.

    Returns a data dict {'u': sens_array, 'w': sens_array} with real values.
    """
    # Build full DOF vectors for direct and adjoint modes
    # Interleave variables: [rho_0, u_0, w_0, e_0, rho_1, u_1, ...]
    var_order = [v for v in ['rho', 'u', 'w', 'e', 'turb1', 'turb2']
                 if v in dir_data and v in adj_data]
    if not var_order:
        return {}

    neq_actual = len(var_order)
    n_nodes    = common

    # Build interleaved arrays (node-major ordering)
    qd_full = np.zeros(n_nodes * neq_actual, dtype=np.complex128)
    qa_full = np.zeros(n_nodes * neq_actual, dtype=np.complex128)
    for k, vname in enumerate(var_order):
        qd_full[k::neq_actual] = dir_data[vname][:n_nodes]
        qa_full[k::neq_actual] = adj_data[vname][:n_nodes]

    # Mass matrix diagonal: volumes repeated neq times
    if vol_path and os.path.isfile(vol_path):
        b_diag = load_volumes(vol_path, neq_actual, n_nodes)
        print(f"  [sensitivity] using mass matrix from {vol_path}")
    else:
        b_diag = np.ones(n_nodes * neq_actual, dtype=np.float64)
        if vol_path:
            print(f"  [sensitivity] vol file not found ({vol_path}), using identity")
        else:
            print(f"  [sensitivity] no vol file provided, using identity")

    # <q_adj, B q_dir> = sum_i conj(qa_i) * b_i * qd_i
    Bqd        = b_diag * qd_full
    inner_prod = np.dot(np.conj(qa_full), Bqd)
    norm_ip    = float(np.abs(inner_prod))
    if norm_ip < 1e-30:
        print("  [sensitivity] WARNING: inner product near zero, not normalising")
        norm_ip = 1.0

    print(f"  [sensitivity] |<q+, B q>| = {norm_ip:.4e}")

    # Pointwise sensitivity: element-wise product of magnitudes / norm_ip
    # This matches eig_solver.py exactly:
    #   sensitivity = np.abs(adj_arr) * np.abs(dir_arr) / norm_ip
    # where adj_arr/dir_arr are the full DOF vectors (size n_nodes*neq_actual)
    # mode2pval stores the first gridpoints values -> DOF 0 of each node = rho
    # read_pval reads var by var: rho=DOF0, u=DOF1, w=DOF2, e=DOF3 per node
    s_dof = np.abs(qa_full) * np.abs(qd_full) / norm_ip  # (n_nodes*neq_actual,)

    # Extract per-variable sensitivity matching read_pval's VAR_MAP ordering
    # DOF ordering within each node: rho=0, u=1, w=2, e=3, turb1=4, turb2=5
    var_dof = {v: k for k, v in enumerate(var_order)}
    sens = {}
    for vname in ['u', 'w']:
        if vname in var_dof:
            dof_idx = var_dof[vname]
            sens[vname] = s_dof[dof_idx::neq_actual].astype(np.complex128)
    return sens


def vars_for_prefix(prefix, requested_vars):
    skip = SKIP_VARS_FOR.get(prefix, set())
    return [v for v in requested_vars if v not in skip]


def is_resolvent_dir(path):
    return 'resolvent' in os.path.basename(os.path.abspath(path)).lower()


def find_pval(directory, prefix, index):
    matches = sorted(glob.glob(os.path.join(directory, f'{prefix}_{index}_*.pval')))
    if not matches:
        matches = sorted(glob.glob(os.path.join(directory, f'{prefix}_{index}.pval')))
    return matches[0] if matches else None


def truncate(data_dict, common):
    return {k: v[:common] for k, v in data_dict.items()}


def parse_mode_indices(tokens):
    indices = []
    for token in tokens:
        if '-' in token and not token.lstrip('-').isdigit():
            start, end = token.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(token))
    return sorted(set(indices))


def tris_info(triang):
    mask = triang.mask
    n_valid = (~mask).sum() if mask is not None else len(triang.triangles)
    return f"{n_valid} triangles from mesh connectivity"


def print_usage_and_exit(error_msg=None):
    """Print a concise usage guide and exit."""
    print(__doc__)
    if error_msg:
        print(f"\n  ERROR: {error_msg}\n", file=sys.stderr)
    sys.exit(1 if error_msg else 0)

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Plot eigenmode .pval file on the 2D computational domain.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,   # we handle --help ourselves to show full usage
        epilog=__doc__)

    p.add_argument('pval', nargs='?', default=None,
                   help='Path to a single .pval file '
                        '(not required with --check-mesh or --modes)')

    p.add_argument('--modes', nargs='+', metavar='N', default=None,
                   help='Modes to plot: single (--modes 2), range (--modes 0-9), '
                        'or list (--modes 0 2 5)')

    p.add_argument('--dir', default='.',
                   help='Directory containing pval files (default: .)')

    p.add_argument('--jac', default='JAC',
                   help='Path to the JAC directory containing samg.matrix.coo '
                        '(default: JAC)')

    p.add_argument('--neq', type=int, default=None,
                   help='Number of equations per node in the .coo file '
                        '(default: auto-detected from pval gridpoints). '
                        'E.g. --neq 6 for SA turbulence models.')

    p.add_argument('--mesh', default=None,
                   help='Mesh file: TAU NetCDF (--mesh MESH/BFS_h4_2D.taumesh), '
                        'SOD2D serial mesh (--mesh MESH/back_facing.h5) or SOD2D '
                        'partitioned mesh (--mesh MESH/back_facing_buffer-112.hdf). '
                        'The type is picked by the extension (.h5 / .hdf)')

    p.add_argument('--bound-ids', nargs='+', type=int, default=None, metavar='ID',
                   help='SOD2D mesh only: boundary ids (boundFacesId) to draw as '
                        'body contour (default: all). The ids found are printed.')

    p.add_argument('--xlim', nargs=2, type=float, metavar=('XMIN', 'XMAX'),
                   default=None,
                   help='x-axis limits (default: full domain). E.g. --xlim -5 20')

    p.add_argument('--ylim', nargs=2, type=float, metavar=('YMIN', 'YMAX'),
                   default=None,
                   help='y-axis limits (default: full domain). E.g. --ylim 0 1')

    p.add_argument('--vars', nargs='+', default=['rho', 'u', 'w', 'e'],
                   help=f'Variables to plot (default: rho u w e). '
                        f'Available: {list(VAR_MAP.keys())} + Umag (computed)')

    p.add_argument('--fields', nargs='+',
                   choices=['eigf', 'eigr', 'eiga', 'sensitivity'],
                   default=['eigf', 'eigr', 'eiga', 'sensitivity'],
                   help='Resolvent fields to include (default: all).')

    p.add_argument('--clim', type=float, default=90,
                   help='Percentile for colorbar range [0-100] (default: 90). '
                        'E.g. --clim 95')

    p.add_argument('--imag', action='store_true',
                   help='Plot imaginary part instead of real part')

    p.add_argument('--both', action='store_true',
                   help='Plot real and imaginary parts (2 panels per figure)')

    p.add_argument('--check-mesh', action='store_true',
                   help='Plot mesh nodes only (sanity check)')

    p.add_argument('-h', '--help', action='store_true',
                   help='Show this help and exit')

    return p.parse_args()

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    global XLIM, YLIM
    args = parse_args()


    # Show help / usage when requested or when called with no arguments
    if args.help or (not args.pval and not args.modes
                     and not args.check_mesh):
        if(args.help):
            print_usage_and_exit()
        else:
            print_usage_and_exit(f"You must specify at least one of pval, modes, check_mesh. Values are:\n pval:{args.pval} modes:{args.modes} check_mesh:{args.check_mesh}")

    # ── 1. coordinates ─────────────────────────────────────────────────────────
    print(f"\n── Loading coordinates ─────────────────────────────────────────")
    mesh_triang  = None
    body_contour = []
    mesh_quads   = None

    # Always load .coo for coordinates (correct node ordering for .pval)
    coords_path = os.path.join(args.jac, 'samg.matrix.coo')
    if not os.path.isfile(coords_path):
        if args.mesh is None:
            print_usage_and_exit(
                f"coordinate file not found: '{coords_path}'\n"
                f"  Use --jac <dir> to specify the JAC directory, or "
                f"--mesh for a TAU mesh."
            )
    else:
        # Auto-detect neq from actual line count in .coo vs gridpoints
        _neq = args.neq
        if _neq is None:
            # Get gridpoints from pval or vol file
            _pval_path = (args.pval if args.pval else
                          next(iter([fp for fp in [
                              find_pval(args.dir, pfx, 0)
                              for pfx, _ in ALL_PREFIXES] if fp]), None))
            _gp = peek_pval_gridpoints(_pval_path) if _pval_path else None

            # Also check .vol file (most reliable)
            _vol_path = os.path.join(args.jac, 'samg.matrix.vol')
            if os.path.isfile(_vol_path):
                with open(_vol_path) as _fv:
                    _gp_vol = sum(1 for _ in _fv)
                if _gp is None or _gp_vol == _gp:
                    _gp = _gp_vol

            if _gp:
                # Count actual data lines in .coo
                with open(coords_path) as _f:
                    _f.readline()   # skip header
                    _nlines = sum(1 for _ in _f)
                # Pick neq so that _nlines // neq == _gp
                for _n in [4, 5, 6, 7, 8]:
                    if _nlines // _n == _gp:
                        _neq = _n
                        break
            if _neq is None:
                _neq = 4   # fallback
            if _neq != 4:
                print(f"  Auto-detected neq = {_neq}")
        x, y = load_coo(coords_path, neq=_neq)

    # Load TAU mesh for triangulation and body contour (optional)
    if args.mesh is not None:
        if not os.path.isfile(args.mesh):
            print_usage_and_exit(f"mesh file not found: '{args.mesh}'")
        if args.mesh.lower().endswith('.h5'):
            x_tau, y_tau, mesh_triang, body_contour, mesh_quads = \
                load_sod2dmesh(args.mesh, bound_ids=args.bound_ids)
        elif args.mesh.lower().endswith('.hdf'):
            x_tau, y_tau, mesh_triang, body_contour, mesh_quads = \
                load_sod2dmesh_partitioned(args.mesh)
        else:
            x_tau, y_tau, mesh_triang, body_contour, mesh_quads = load_taumesh(args.mesh)
        if not os.path.isfile(coords_path):
            # No .coo: use TAU coordinates directly
            x, y = x_tau, y_tau
        elif len(x) != len(x_tau) or not np.allclose(x[:5], x_tau[:5], atol=1e-6):
            # .coo and TAU have different ordering — remap triangulation
            mesh_triang = _remap_triangulation(x, y, x_tau, y_tau, mesh_triang)

            # Scale body contour from TAU coords to .coo coords
            if body_contour:
                range_coo_x = x.max() - x.min()
                range_tau_x = x_tau.max() - x_tau.min()
                range_coo_y = y.max() - y.min()
                range_tau_y = y_tau.max() - y_tau.min()
                sx = range_coo_x / range_tau_x if range_tau_x > 0 else 1.0
                sy = range_coo_y / range_tau_y if range_tau_y > 0 else 1.0
                scale = (sx + sy) / 2.0 if abs(sx - sy) / max(sx, 1e-10) < 0.01 else 1.0
                if abs(scale - 1.0) > 0.01:
                    ox = x.min() - x_tau.min() * scale
                    oy = y.min() - y_tau.min() * scale
                    scaled = []
                    for bc in body_contour:
                        bc_s = bc.copy()
                        bc_s[:, 0] = bc[:, 0] * scale + ox
                        bc_s[:, 1] = bc[:, 1] * scale + oy
                        scaled.append(bc_s)
                    body_contour = scaled
                    print(f"  Body contour scaled by {scale:.6g} to .coo space")
            # If remap failed (too few valid triangles), fall back to Delaunay
            if mesh_triang is not None and len(mesh_triang.triangles) < len(x) // 10:
                print(f"  WARNING: remap produced only {len(mesh_triang.triangles)} triangles "
                      f"— TAU mesh likely incompatible with .coo. Falling back to Delaunay.")
                mesh_triang = None

    # ── 2. set x limits ────────────────────────────────────────────────────────
    if args.xlim is not None:
        XLIM = tuple(args.xlim)
        print(f"  x limits: {XLIM[0]} to {XLIM[1]}  (from --xlim)")
    else:
        XLIM = (float(x.min()), float(x.max()))
        print(f"  x limits: {XLIM[0]:.4g} to {XLIM[1]:.4g}  (full domain)")

    if args.ylim is not None:
        YLIM = tuple(args.ylim)
        print(f"  y limits: {YLIM[0]} to {YLIM[1]}  (from --ylim)")
    else:
        YLIM = (float(y.min()), float(y.max()))
        print(f"  y limits: {YLIM[0]:.4g} to {YLIM[1]:.4g}  (full domain)")

    # ── 3. mesh-only sanity check ──────────────────────────────────────────────
    if args.check_mesh:
        check_mesh(x, y,
                   triang=mesh_triang,
                   body_contour=body_contour,
                   mesh_quads=mesh_quads,
                   mesh_name=os.path.basename(args.mesh))
        print("Done.\n")
        return

    # ── 3b. load eigenvalues ──────────────────────────────────────────────────
    # Also search in the directory containing the .pval file
    _pval_dir = os.path.dirname(os.path.abspath(args.pval)) if args.pval else None
    _search   = [d for d in [args.dir, _pval_dir, args.jac, '.'] if d]
    eigenvalues = load_eigenvalues(_search)

    # ── 4. detect resolvent directory ─────────────────────────────────────────────
    resolvent = is_resolvent_dir(args.dir)
    if resolvent:
        print(f"  Resolvent directory detected — plotting all available fields.")

    # ── 5. build list of paths to process ─────────────────────────────────────
    if args.modes is not None:
        indices   = parse_mode_indices(args.modes)
        work_list = []
        for i in indices:
            fp = find_pval(args.dir, 'eigf', i)
            if fp is None:
                for prefix, _ in ALL_PREFIXES:
                    if prefix not in args.fields:
                        continue
                    fp = find_pval(args.dir, prefix, i)
                    if fp is not None:
                        break
            if fp is None:
                print(f"  [skip] No file found for index {i} in '{args.dir}'")
            else:
                work_list.append(fp)
        if not work_list:
            print_usage_and_exit(
                f"no .pval files found for modes {args.modes} in '{args.dir}'"
            )
    elif args.pval is not None:
        if not os.path.isfile(args.pval):
            # If sensitivity file missing, try to compute from eigf + eiga
            fname_base = os.path.basename(args.pval)
            if fname_base.startswith('sensitivity_'):
                # Extract index from filename: sensitivity_181.pval -> 181
                _stem  = os.path.splitext(fname_base)[0]  # sensitivity_181
                _parts = _stem.split('_', 1)
                _idx_s = _parts[1] if len(_parts) > 1 else None
                _dir_s = os.path.dirname(args.pval) or '.'
                _eigf  = find_pval(_dir_s, 'eigf', _idx_s) if _idx_s else None
                _eiga  = find_pval(_dir_s, 'eiga', _idx_s) if _idx_s else None
                if _eigf and _eiga:
                    print(f"  sensitivity file not found — will compute from eigf + eiga")
                    work_list  = [('__compute_sensitivity__', _eigf, _eiga, _idx_s)]
                    resolvent  = False
                else:
                    print_usage_and_exit(f".pval file not found: '{args.pval}'")
            else:
                print_usage_and_exit(f".pval file not found: '{args.pval}'")
        else:
            work_list = [args.pval]
            resolvent = False
    else:
        print_usage_and_exit("provide a .pval file or use --modes.")

    # ── 6. process ────────────────────────────────────────────────────────────
    cached_triang  = None
    cached_n_nodes = None

    for item in work_list:

        if not resolvent:
            # ── special: compute sensitivity on-the-fly ────────────────────────
            if isinstance(item, tuple) and item[0] == '__compute_sensitivity__':
                _, _eigf_path, _eiga_path, _idx_s = item
                print(f"\n── Computing sensitivity from eigf + eiga ──────────────────")
                _vars_d = vars_for_prefix('eigf', args.vars)
                _vars_a = vars_for_prefix('eiga', args.vars)
                gridpoints, dir_data = read_pval(_eigf_path, _vars_d)
                _,          adj_data = read_pval(_eiga_path, _vars_a)
                common = min(len(x), gridpoints)
                xi, yi = x[:common], y[:common]
                dir_data = truncate(dir_data, common)
                adj_data = truncate(adj_data, common)
                _vol_path = os.path.join(args.jac, 'samg.matrix.vol')
                forc_data = compute_sensitivity(dir_data, adj_data, common,
                                                vol_path=_vol_path)
                if not forc_data:
                    print("  [skip] Could not compute sensitivity.")
                    continue
                if cached_n_nodes != common:
                    if mesh_triang is not None:
                        if len(mesh_triang.x) != common:
                            valid = (mesh_triang.triangles < common).all(axis=1)
                            t_trim = mesh_triang.triangles[valid]
                            cached_triang = tri.Triangulation(xi, yi, t_trim)
                        else:
                            cached_triang = mesh_triang
                    else:
                        cached_triang = build_triangulation(xi, yi)
                    cached_n_nodes = common
                out_stem = os.path.splitext(os.path.abspath(args.pval))[0]
                title    = os.path.basename(args.pval)
                _midx    = int(_idx_s) if _idx_s and _idx_s.isdigit() else None
                _eig_s   = eigenvalue_str(eigenvalues, _midx) if _midx is not None else ''
                print(f"\n── Plotting ────────────────────────────────────────────────────")
                plot_modes(xi, yi, forc_data, 'RdBu', out_stem, title,
                           triang=cached_triang, clim_pct=args.clim,
                           plot_imag=args.imag, plot_both=args.both,
                           prefix='sensitivity', body_contour=body_contour,
                           eig_str=_eig_s, mode_idx=_midx)
                continue

            forc_path   = item
            fname_base  = os.path.basename(forc_path)
            file_prefix = next((p for p, _ in ALL_PREFIXES
                                if fname_base.startswith(p + '_')), None)
            vars_to_read = vars_for_prefix(file_prefix, args.vars) \
                           if file_prefix else args.vars
            print(f"\n── Reading file ────────────────────────────────────────────────")
            gridpoints, forc_data = read_pval(forc_path, vars_to_read)
            if not forc_data:
                print(f"  [skip] No data in '{forc_path}'.")
                continue
            common = min(len(x), gridpoints)
            if len(x) != gridpoints:
                print(f"  WARNING: coord nodes ({len(x)}) != pval gridpoints "
                      f"({gridpoints}). Truncating to {common}: the plot is only "
                      f"correct if both files use the same node ordering "
                      f"(check the .coo matches the Jacobian/domain reduction).")
            xi, yi = x[:common], y[:common]
            forc_data = truncate(forc_data, common)
            if cached_n_nodes != common:
                if mesh_triang is not None:
                    # Trim triangulation to common nodes if needed
                    if len(mesh_triang.x) != common:
                        valid = (mesh_triang.triangles < common).all(axis=1)
                        t_trim = mesh_triang.triangles[valid]
                        plot_triang = tri.Triangulation(xi, yi, t_trim)
                        print(f"  Using mesh connectivity ({(~plot_triang.mask).sum() if plot_triang.mask is not None else len(plot_triang.triangles)} triangles).")
                    else:
                        plot_triang = mesh_triang
                        print(f"  Using mesh connectivity ({tris_info(mesh_triang)}).")
                    cached_triang = plot_triang
                else:
                    cached_triang = build_triangulation(xi, yi)
                cached_n_nodes = common
            out_stem   = os.path.splitext(os.path.abspath(forc_path))[0]
            fname_stem = os.path.splitext(os.path.basename(forc_path))[0]
            # Extract mode index from filename: eigf_3_1.2j -> 3
            _parts = fname_stem.split('_')
            _midx  = int(_parts[1]) if len(_parts) > 1 and _parts[1].lstrip('-').isdigit() else None
            _eig_s = eigenvalue_str(eigenvalues, _midx) if _midx is not None else ''
            title  = f"{os.path.basename(forc_path)}   {_eig_s}" if _eig_s else os.path.basename(forc_path)
            print(f"\n── Plotting ────────────────────────────────────────────────────")
            # Extract omega from filename if present (e.g. eigf_0_1.2j.pval)
            _omg_s_sf = omega_str('_'.join(_parts[1:])) if len(_parts) > 2 else ''
            plot_modes(xi, yi, forc_data, 'RdBu', out_stem, title,
                       triang=cached_triang, clim_pct=args.clim,
                       plot_imag=args.imag, plot_both=args.both,
                       prefix=file_prefix, body_contour=body_contour,
                       eig_str=_eig_s, omg_str=_omg_s_sf, mode_idx=_midx)
            continue

        # ── resolvent mode ─────────────────────────────────────────────────────
        ref_path   = item
        ref_fname  = os.path.splitext(os.path.basename(ref_path))[0]
        ref_prefix = next((p for p, _ in ALL_PREFIXES
                           if ref_fname.startswith(p + '_')), None)
        idx_omega  = ref_fname[len(ref_prefix) + 1:] if ref_prefix else ref_fname
        idx = int(idx_omega.split('_')[0])

        gridpoints, _ = read_pval(ref_path, ['u'])
        common = min(len(x), gridpoints)
        xi, yi = x[:common], y[:common]
        if len(x) != gridpoints:
            print(f"  WARNING: coord nodes ({len(x)}) ≠ gridpoints ({gridpoints}). "
                  f"Truncating to {common}.")

        if cached_n_nodes != common:
            if mesh_triang is not None:
                if len(mesh_triang.x) != common:
                    valid = (mesh_triang.triangles < common).all(axis=1)
                    t_trim = mesh_triang.triangles[valid]
                    cached_triang = tri.Triangulation(xi, yi, t_trim)
                else:
                    cached_triang = mesh_triang
                print(f"  Using mesh connectivity ({tris_info(cached_triang)}).")
            else:
                print(f"  Building Delaunay triangulation for {common} nodes …")
                cached_triang = build_triangulation(xi, yi)
            cached_n_nodes = common
        else:
            print(f"  Reusing cached triangulation ({common} nodes).")

        datasets = []
        for prefix, label in ALL_PREFIXES:
            if prefix not in args.fields:
                continue
            path = find_pval(args.dir, prefix, idx)
            if path is None:
                print(f"  [{label}] not found for index {idx} — skipping.")
                continue
            print(f"\n── Reading {label} file ────────────────────────────────────────")
            _, ddata = read_pval(path, vars_for_prefix(prefix, args.vars))
            ddata = truncate(ddata, common)
            datasets.append((prefix, label, ddata, prefix == 'sensitivity'))

        if not datasets:
            print(f"  [skip] No data loaded for index {idx}.")
            continue

        _eig_s = eigenvalue_str(eigenvalues, idx)
        _omg_s = omega_str(idx_omega)
        print(f"\n── Plotting ────────────────────────────────────────────────────")
        plot_resolvent(datasets, 'RdBu',
                       dir_path=os.path.abspath(args.dir),
                       idx_omega=idx_omega,
                       triang=cached_triang, clim_pct=args.clim,
                       plot_imag=args.imag, plot_both=args.both,
                       body_contour=body_contour,
                       eig_str=_eig_s, omg_str=_omg_s, mode_idx=idx)

    print("Done.\n")


if __name__ == '__main__':
    main()