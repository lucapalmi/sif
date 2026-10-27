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
    """Builds the grid afresh for every run. A finder hands its grid back
    restored to float precision but not to the bit, and centres placed
    between the nodes follow the field's last bits, so runs compared bit for
    bit each start from a new grid -- as a resumed run in a new process
    does."""
    def make(scheme="cic", interlace=False):
        g = pysif.Grid(n_cells=N_GRID, box_length=BOX)
        g.assign(pysif.field_from_numpy(*tracers), scheme, interlace=interlace)
        g.to_density_contrast()
        return g
    return make


@pytest.fixture(scope="module")
def mesh(tracers):
    cells = pysif.finders.suggest_mesh_cells(N, BOX, max_radius=float(RADII[0]))
    return pysif.ChainMesh(cells, BOX, pysif.field_from_numpy(*tracers))


def columns(cat):
    return np.vstack([cat.centres.T, cat.radii])


def run(grid, mesh, radii=RADII, **kw):
    return pysif.finders.exodus_raw(grid(), mesh, radii, THRESHOLD, **kw)


CIC = 2  # sif_assignment_t


def write_checkpoint(path, cat, rungs_done, grid_cells=N_GRID, version=2,
                     options=0, assignment=CIC, overlap=0.0):
    """A checkpoint of a box run over RADII, from the layout in
    exodus_finder.h: the 192-byte header, the ladder largest first, then x,
    y, z and r as four columns. Options 0 are the defaults: search factor
    1.5, seeds='cells', the window divided out in full."""
    real = np.dtype(pysif.real)
    ladder = np.sort(RADII)[::-1].astype(real).tobytes()
    voids = b"".join(np.ascontiguousarray(c, dtype=real).tobytes()
                     for c in columns(cat))

    def header(crc):
        return struct.pack(
            "=4s9I3dQdQd3dQIIQQI36s",
            b"XCKP", version, 0x01020304, int(real.itemsize == 8), 0, 0,
            options,
            len(RADII), rungs_done, grid_cells,
            BOX, float(np.float32(THRESHOLD)) if real.itemsize == 4
            else THRESHOLD, overlap,
            N, float(N), 0, 0.0, 0.0, 0.0, 0.0, 0, 0, crc,
            cat.n_voids, 0, assignment, b"")

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


def test_checkpoint_records_the_assignment(grid, mesh, tmp_path):
    # An interlaced TSC grid, deconvolved allowing for aliasing.
    aliasing, tsc_interlaced = 1 << 8, 3 | 1 << 8
    def g():
        return grid("tsc", interlace=True)
    ref = run(g, mesh, checkpoint=False, deconvolution="aliasing")
    head = run(g, mesh, RADII[:2], checkpoint=False, deconvolution="aliasing")
    path = tmp_path / "run.ckpt"
    write_checkpoint(path, head, rungs_done=2, options=aliasing,
                     assignment=tsc_interlaced)

    _, info = pysif.finders.read_checkpoint(path)
    assert info["deconvolution"] == "aliasing"
    assert info["assignment"] == "tsc" and info["interlaced"] is True
    assert info["seeds"] == "cells"

    with pytest.raises(ValueError, match=r"not this run: window deconvolved"):
        run(g, mesh, checkpoint=path)
    with pytest.raises(ValueError, match=r"grid assigned with TSC, interlaced"):
        run(grid, mesh, checkpoint=path, deconvolution="aliasing")
    assert path.exists()

    resumed = run(g, mesh, checkpoint=path, deconvolution="aliasing")
    np.testing.assert_array_equal(columns(resumed), columns(ref))


def test_version_1_checkpoint(grid, mesh, tmp_path):
    # Written before off-grid centres and assignment schemes: every cell on
    # its node, on a CIC grid -- seeds='cells' -- which it resumes.
    ref = run(grid, mesh, checkpoint=False)
    head = run(grid, mesh, RADII[:2], checkpoint=False)
    path = tmp_path / "run.ckpt"
    write_checkpoint(path, head, rungs_done=2, version=1, assignment=0)
    _, info = pysif.finders.read_checkpoint(path)
    assert info["assignment"] == "cic" and info["interlaced"] is False
    assert info["seeds"] == "cells"
    assert info["deconvolution"] == "full"
    with pytest.raises(ValueError,
                       match="not this run: centres every cell, on its node"):
        run(grid, mesh, checkpoint=path, seeds="minima")
    resumed = run(grid, mesh, checkpoint=path)
    np.testing.assert_array_equal(columns(resumed), columns(ref))


def test_unknown_version_is_refused(grid, mesh, tmp_path):
    head = run(grid, mesh, RADII[:2], checkpoint=False)
    path = tmp_path / "run.ckpt"
    write_checkpoint(path, head, rungs_done=2, version=3)
    with pytest.raises(OSError, match=r"version 3 \(expected 1 to 2\)"):
        pysif.finders.read_checkpoint(path)


def test_variant_checkpoints(grid, mesh, tmp_path):
    # A variant always checkpoints; where to, and whether to keep it, are
    # still the caller's.
    path = tmp_path / "run.ckpt"
    ref = run(grid, mesh, checkpoint=False, seeds="minima", overlap_fraction=1)
    cat = pysif.finders.exodus(grid(), mesh, RADII, THRESHOLD,
                               variant="excursion-peak", checkpoint=path,
                               keep_checkpoint=True)
    np.testing.assert_array_equal(columns(cat), columns(ref))
    assert path.exists()
    _, info = pysif.finders.read_checkpoint(path)
    assert info["rungs_done"] == len(RADII) and info["overlap_fraction"] == 1
    with pytest.raises(ValueError, match="a variant always checkpoints"):
        pysif.finders.exodus(grid(), mesh, RADII, THRESHOLD, checkpoint=False)


def test_prior(grid, mesh):
    # The larger rungs' voids, extended with the rest of the ladder: one run.
    ref = run(grid, mesh, checkpoint=False)
    large = run(grid, mesh, RADII[:2], checkpoint=False)
    extended = run(grid, mesh, RADII[2:], checkpoint=False, prior=large)
    np.testing.assert_array_equal(columns(extended), columns(ref))


@pytest.fixture
def scratch(tmp_path):
    """scratch_directory for this test only."""
    path = tmp_path / "scratch"
    pysif.set_setting("scratch_directory", str(path), persist=False)
    yield path
    pysif.set_setting("scratch_directory", None, persist=False)


def test_save_memory(grid, mesh, tmp_path, scratch):
    # The same voids, a checkpoint that resumes either way, and nothing left
    # in the scratch directory.
    ref = run(grid, mesh, checkpoint=False)
    lean = run(grid, mesh, checkpoint=False, save_memory=True)
    np.testing.assert_array_equal(columns(lean), columns(ref))

    path = tmp_path / "half.ckpt"
    write_checkpoint(path, run(grid, mesh, RADII[:2], checkpoint=False), 2)
    resumed = run(grid, mesh, checkpoint=path, save_memory=True)
    np.testing.assert_array_equal(columns(resumed), columns(ref))
    assert list(scratch.iterdir()) == []


def test_save_memory_unusable_scratch(grid, mesh, tmp_path):
    (tmp_path / "file").write_text("")
    pysif.set_setting("scratch_directory", str(tmp_path / "file" / "x"),
                      persist=False)
    try:
        with pytest.raises(OSError, match="file"):
            run(grid, mesh, checkpoint=False, save_memory=True)
    finally:
        pysif.set_setting("scratch_directory", None, persist=False)


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
