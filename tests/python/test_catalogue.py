# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.catalogue_from_numpy, Catalogue.to_numpy and the catalogue's metadata,
through the text and HDF5 formats (FITS in test_fits.py).

The formats themselves are tested in C (tests/test_io.c, tests/test_hdf5.c).
Here: that the keyword arguments and the NumPy side map onto them, and that
each kind of failure raises the exception it is documented to.
"""

import numpy as np
import pysif
import pytest
from conftest import requires_hdf5

N = 25
rng = np.random.default_rng(4)
CX, CY, CZ = (rng.random(N) * 100 for _ in range(3))
R = 5 + 10 * rng.random(N)


def catalogue():
    return pysif.catalogue_from_numpy(CX, CY, CZ, R)


def test_from_and_to_numpy():
    cat = catalogue()
    assert cat.n_voids == N and cat.units == "cartesian"
    rec = cat.to_numpy()
    assert rec.dtype.names == ("cx", "cy", "cz", "r")
    for name, want in zip(rec.dtype.names, (CX, CY, CZ, R)):
        np.testing.assert_array_equal(rec[name], want.astype(pysif.real))

    sky = pysif.catalogue_from_numpy(ra=CX, dec=CY - 50, z=CZ / 100, r=R,
                                   footprint=np.ones(N), footprint_shell=np.ones(N))
    assert sky.units == "sky"
    assert sky.to_numpy().dtype.names == ("ra", "dec", "z", "r", "footprint",
                                          "footprint_shell")


@pytest.mark.parametrize("kwargs, error", [
    (dict(cx=CX, cy=CY, cz=CZ), ValueError),                  # no radius
    (dict(cx=CX, cy=CY, z=CZ, r=R), ValueError),              # mixed names
    (dict(cx=CX, cy=CY, cz=CZ[:-1], r=R), ValueError),        # lengths differ
    (dict(cx=CX, cy=CY, cz=CZ, r=R, footprint=R), ValueError),  # half a footprint
])
def test_from_numpy_refusals(kwargs, error):
    with pytest.raises(error):
        pysif.catalogue_from_numpy(**kwargs)


def test_metadata():
    cat = catalogue()
    assert cat.metadata == {}
    cat.set_metadata("Finder", "exodus")
    cat.set_metadata("threshold", -0.7)
    cat.set_metadata("n_tracers", 2**40)
    assert cat.metadata == {"finder": "exodus", "threshold": -0.7,
                            "n_tracers": 2**40}
    cat.set_metadata("finder", None)
    assert "finder" not in cat.metadata

    for key, value in [("n_voids", 1), ("naxis2", 1), ("two words", 1),
                       ("x", float("nan")), ("x", 'a "quote"')]:
        with pytest.raises(ValueError):
            cat.set_metadata(key, value)
    with pytest.raises(TypeError):
        cat.set_metadata("x", [1])


def with_metadata():
    cat = catalogue()
    cat.set_metadata("finder", "exodus")
    cat.set_metadata("threshold", -0.7)
    cat.set_metadata("n_tracers", 2**40)
    cat.set_metadata("label", "10")
    return cat


def test_metadata_through_text(tmp_path):
    cat = with_metadata()
    path = tmp_path / "voids.txt"
    pysif.io.write_catalogue_ascii(str(path), cat)
    assert pysif.io.read_catalogue_ascii(str(path)).metadata == cat.metadata
    # The rows are plain numbers, the header comments.
    assert np.loadtxt(path).shape == (N, 4)


@requires_hdf5
def test_metadata_through_hdf5(tmp_path):
    cat = with_metadata()
    path = str(tmp_path / "voids.h5")
    pysif.io.write_catalogue_hdf5(path, cat)
    back = pysif.io.read_catalogue_hdf5(path)
    assert back.metadata == cat.metadata
    np.testing.assert_array_equal(back.to_numpy(), cat.to_numpy())


def test_foreign_text_catalogue(tmp_path):
    # Another finder's layout: an ID, the centre, a volume, the radius, a tag.
    path = tmp_path / "other.txt"
    path.write_text("# ID x_centre y_centre z_centre volume R_eff type\n"
                    "7 1.5 2.5 3.5 900 12 main\n"
                    "8 4.5 5.5 6.5 100 6 sub\n")
    cat = pysif.io.read_catalogue_ascii(str(path), format="* x y z * r")
    np.testing.assert_array_equal(cat.to_numpy()["r"], [12, 6])
    np.testing.assert_array_equal(cat.to_numpy()["cz"], [3.5, 6.5])

    with pytest.raises(OSError):
        pysif.io.read_catalogue_ascii(str(path))
    # A format without the radius is the request's mistake, not the file's.
    with pytest.raises(ValueError, match="needs the centre"):
        pysif.io.read_catalogue_ascii(str(path), format="x y z")
