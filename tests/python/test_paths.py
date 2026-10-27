# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Every reader and writer takes a path the way open() does: a str, bytes,
or anything os.PathLike (pathlib.Path), positionally or by keyword.

Each function is driven once per form, round-tripping where it writes, so
that a binding still parsing its path as a str fails here by name.
"""

import os
from pathlib import Path

import numpy as np
import pysif
import pytest
from conftest import GADGET, requires_hdf5

FORMS = {
    "pathlib": Path,
    "bytes": os.fsencode,
    "str": str,
}


@pytest.fixture(params=FORMS)
def form(request):
    return FORMS[request.param]


def field():
    x = np.linspace(1.0, 9.0, 16)
    return x, x[::-1].copy(), np.full(16, 5.0)


def catalogue():
    c = np.array([2.0, 5.0, 8.0])
    return pysif.catalogue_from_numpy(c, c, c, np.array([1.0, 1.5, 2.0]))


def test_xfield(tmp_path, form):
    p = form(tmp_path / "f.xfield")
    pysif.io.write_field(p, 10.0, *field())
    assert pysif.io.read_field_header(p)["n_particles"] == 16
    assert pysif.io.read_field(p).n_particles == 16
    assert pysif.io.read_field(filepath=p, wrap=True).n_particles == 16


def test_xgrid(tmp_path, form):
    p = form(tmp_path / "g.xgrid")
    grid = pysif.Grid(8, 10.0)
    grid.assign_cic(pysif.field_from_numpy(*field()))
    pysif.io.write_grid(p, grid)
    assert pysif.io.read_grid(filepath=p).assignment == "cic"


def test_field_ascii_and_binary(tmp_path, form):
    text = tmp_path / "f.txt"
    np.savetxt(text, np.column_stack(field()))
    assert pysif.io.read_field_ascii(form(text)).n_particles == 16
    raw = tmp_path / "f.bin"
    np.column_stack(field()).astype(np.float32).tofile(raw)
    assert pysif.io.read_field_binary(filepath=form(raw)).n_particles == 16


def test_ascii_products(tmp_path, form):
    p = form(tmp_path / "c.txt")
    cat = catalogue()
    pysif.io.write_catalogue_ascii(p, cat)
    assert pysif.io.read_catalogue_ascii(p).n_voids == 3

    mesh = pysif.ChainMesh(2, 10.0, pysif.field_from_numpy(*field()))
    profiles = pysif.measure.profiles(cat, mesh, 4)
    q = form(tmp_path / "p.txt")
    pysif.io.write_profiles_ascii(q, profiles, cat)
    back, _ = pysif.io.read_profiles_ascii(filepath=q)
    assert back.n_voids == 3


@requires_hdf5
def test_hdf5_products(tmp_path, form):
    p = form(tmp_path / "products.h5")
    cat = catalogue()
    pysif.io.write_catalogue_hdf5(p, cat)
    assert pysif.io.read_catalogue_hdf5(p).n_voids == 3

    mesh = pysif.ChainMesh(2, 10.0, pysif.field_from_numpy(*field()))
    pysif.io.write_profiles_hdf5(p, pysif.measure.profiles(cat, mesh, 4))
    assert pysif.io.read_profiles_hdf5(p).n_voids == 3

    vsf = pysif.measure.size_function_catalogue(cat, 10.0, 2)
    pysif.io.write_size_function_hdf5(p, vsf)
    assert pysif.io.read_size_function_hdf5(p).n_bins == 2

    pysif.io.set_hdf5_attr(p, "origin", "test")
    assert pysif.io.get_hdf5_attr(filepath=p, key="origin") == "test"
    assert pysif.io.get_hdf5_attrs(p)["origin"] == "test"


def test_gadget(form, capsys):
    p = form(GADGET / "f1_legacy" / "snap_005")
    assert pysif.io.gadget_header(p)["format"] == 1
    pysif.io.inspect_gadget(path=p)
    assert "snap_005" in capsys.readouterr().out
    field, _ = pysif.io.read_gadget(p, length="kpc")
    assert field.n_particles == 100


@pytest.mark.parametrize("call", [
    lambda: pysif.io.read_field(42),
    lambda: pysif.io.read_grid(filepath=None),
    lambda: pysif.io.read_gadget(3.5),
])
def test_not_a_path(call):
    with pytest.raises(TypeError, match="str, bytes or os.PathLike"):
        call()


def test_missing_path_argument():
    with pytest.raises(TypeError, match="filepath"):
        pysif.io.read_field()
