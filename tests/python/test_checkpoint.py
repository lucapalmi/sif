# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""checkpoint= and prior= on the exodus finders, read_checkpoint(), and
Catalogue.to_cartesian(): that each reaches the C side and comes back as it
says. Resuming bit for bit is tested in C (tests/test_finders.c); here a
checkpoint is written from its documented layout alone, which is what anyone
outside sif would have to go by."""

import struct
import zlib

import numpy as np
import pysif
import pytest

BOX, N_GRID, N = 200.0, 32, 60_000
HOLES = [(50, 50, 50, 30), (150, 60, 120, 25), (80, 150, 160, 20),
         (160, 160, 40, 16)]
RADII = np.array([30, 25, 20, 16, 13], dtype=pysif.real)
THRESHOLD = -0.7


@pytest.fixture(scope="module")
def tracers():
    rng = np.random.default_rng(3)
    pts = rng.random((3, 3 * N)) * BOX
    keep = np.ones(pts.shape[1], dtype=bool)
    for x, y, z, r in HOLES:
        keep &= (pts[0] - x)**2 + (pts[1] - y)**2 + (pts[2] - z)**2 >= r * r
    return pts[:, keep][:, :N].astype(pysif.real)


@pytest.fixture
def grid(tracers):
    g = pysif.Grid(n_cells=N_GRID, box_length=BOX)
    g.assign_cic(field=pysif.field_from_numpy(*tracers))
    g.to_density_contrast()
    return g


@pytest.fixture(scope="module")
def mesh(tracers):
    cells = pysif.finders.suggest_mesh_cells(N, BOX, max_radius=float(RADII[0]))
    return pysif.ChainMesh(cells, BOX, pysif.field_from_numpy(*tracers))


def columns(cat):
    return np.vstack([cat.centres.T, cat.radii])


def run(grid, mesh, radii=RADII, **kw):
    return pysif.finders.exodus(grid, mesh, radii, THRESHOLD, **kw)


def write_checkpoint(path, cat, rungs_done, grid_cells=N_GRID):
    """A checkpoint of a box run over RADII, from the layout in
    exodus_finder.h: the 192-byte header, the ladder largest first, then x,
    y, z and r as four columns."""
    real = np.dtype(pysif.real)
    ladder = np.sort(RADII)[::-1].astype(real).tobytes()
    voids = b"".join(np.ascontiguousarray(c, dtype=real).tobytes()
                     for c in columns(cat))

    def header(crc):
        return struct.pack(
            "=4s9I3dQdQd3dQIIQQ40s",
            b"XCKP", 1, 0x01020304, int(real.itemsize == 8), 0, 0, 0,
            len(RADII), rungs_done, grid_cells,
            BOX, float(np.float32(THRESHOLD)) if real.itemsize == 4
            else THRESHOLD, 0.0,
            N, float(N), 0, 0.0, 0.0, 0.0, 0.0, 0, 0, crc,
            cat.n_voids, 0, b"")

    assert len(header(0)) == 192
    crc = zlib.crc32(header(0) + ladder + voids)
    path.write_bytes(header(crc) + ladder + voids)


def test_plain_run_unchanged(grid, mesh, tmp_path, monkeypatch):
    # Every way of saying "checkpoint" finds the same voids, and leaves
    # nothing behind once the run completes.
    monkeypatch.chdir(tmp_path)
    ref = run(grid, mesh, checkpoint=False)
    assert ref.n_voids >= len(HOLES)
    for kw in ({}, {"checkpoint": None}, {"checkpoint": tmp_path / "a.ckpt"},
               {"checkpoint": str(tmp_path / "b.ckpt")}):
        np.testing.assert_array_equal(columns(run(grid, mesh, **kw)),
                                      columns(ref))
    assert list(tmp_path.iterdir()) == []


def test_resume_from_a_checkpoint_written_by_hand(grid, mesh, tmp_path):
    ref = run(grid, mesh, checkpoint=False)
    head = run(grid, mesh, RADII[:2], checkpoint=False)
    path = tmp_path / "run.ckpt"
    write_checkpoint(path, head, rungs_done=2)

    voids, info = pysif.finders.read_checkpoint(path)
    np.testing.assert_array_equal(columns(voids), columns(head))
    assert info["mode"] == "box" and info["rungs_done"] == 2
    np.testing.assert_array_equal(info["radii"], np.sort(RADII)[::-1])
    assert info["n_particles"] == N and info["grid_n_cells"] == N_GRID

    np.testing.assert_array_equal(columns(run(grid, mesh, checkpoint=path)),
                                  columns(ref))
    assert not path.exists()


def test_another_runs_checkpoint_is_refused(grid, mesh, tmp_path):
    head = run(grid, mesh, RADII[:2], checkpoint=False)
    path = tmp_path / "run.ckpt"
    write_checkpoint(path, head, rungs_done=2, grid_cells=64)
    with pytest.raises(ValueError, match=r"not this run: grid of 64 cells"):
        run(grid, mesh, checkpoint=path)
    assert path.exists()

    path.write_bytes(path.read_bytes()[:-1] + b"\0")
    with pytest.raises(OSError, match="corrupt"):
        pysif.finders.read_checkpoint(path)
    with pytest.raises(FileNotFoundError):
        pysif.finders.read_checkpoint(tmp_path / "missing.ckpt")


def test_prior(grid, mesh):
    # The larger rungs' voids, extended with the rest of the ladder: one run.
    ref = run(grid, mesh, checkpoint=False)
    large = run(grid, mesh, RADII[:2], checkpoint=False)
    extended = run(grid, mesh, RADII[2:], checkpoint=False, prior=large)
    np.testing.assert_array_equal(columns(extended), columns(ref))


@pytest.mark.parametrize("kw, error, message", [
    ({"checkpoint": True}, ValueError, "checkpoint: True"),
    ({"checkpoint": 3}, TypeError, None),
    ({"prior": [1, 2]}, TypeError, "prior: expected a Catalogue"),
])
def test_bad_extras(grid, mesh, kw, error, message):
    with pytest.raises(error, match=message):
        run(grid, mesh, **kw)


def test_to_cartesian_undoes_to_sky():
    real = pysif.real
    cat = pysif.catalogue_from_numpy(
        cx=np.array([300.0, -80.0], dtype=real),
        cy=np.array([-120.0, 400.0], dtype=real),
        cz=np.array([50.0, -200.0], dtype=real),
        r=np.array([10.0, 10.0], dtype=real))
    before = columns(cat)
    cat.to_sky(0.31)
    assert cat.units == "sky"
    cat.to_cartesian(0.31)
    assert cat.units == "cartesian"
    np.testing.assert_allclose(columns(cat), before, rtol=1e-5)
    with pytest.raises(ValueError, match="already Cartesian"):
        cat.to_cartesian(0.31)
