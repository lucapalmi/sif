# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.io.read_fits and inspect_fits, on the fixtures in tests/data/fits --
see make_fixtures.py there for what they hold.

The reader itself is tested in C (tests/test_fits.c). Here: that the keyword
arguments map onto the right C options, that the values reach Python intact
from files cfitsio did not write, and that each kind of failure raises the
exception it is documented to.
"""

import numpy as np
import pysif
import pytest
from conftest import FITS, HAS_FITS

pytestmark = pytest.mark.skipif(not HAS_FITS, reason="pysif built without FITS")

N_ROWS = 100
I = np.arange(N_ROWS)
RA = 3.5 * I
DEC = -45 + 0.75 * I
Z = 0.1 + 0.005 * I
W1 = 1 + (I % 4) / 4
W2 = 2 + I % 3

SKY = dict(ra="RA", dec="DEC", z="Z")


def path(name):
    return str(FITS / name)


def read(name, **kwargs):
    return pysif.io.read_fits(path(name), **kwargs)


def test_columns():
    field = read("catalogue.fits", **SKY)
    assert field.n_particles == N_ROWS and field.units == "sky"
    np.testing.assert_array_equal(field.x, RA.astype(field.x.dtype))
    np.testing.assert_array_equal(field.y, DEC.astype(field.y.dtype))
    np.testing.assert_allclose(field.z, Z, rtol=1e-6)
    assert field.weights is None and field.vx is None


def test_cartesian_is_the_default():
    assert read("catalogue.fits", x="RA", y="DEC", z="Z").units == "cartesian"


def test_expressions():
    field = read("catalogue.fits", w="W1 * W2", **SKY)
    np.testing.assert_array_equal(field.weights, (W1 * W2).astype(field.weights.dtype))

    vec = read("catalogue.fits", x="POS[1]", y="POS[2]", z="POS[3]",
               vx="POS[1]", vy="POS[2]", vz="POS[3]")
    np.testing.assert_array_equal(vec.x, 3.0 * I)
    np.testing.assert_array_equal(vec.vz, 3.0 * I + 2)


def test_filter_and_subsample():
    keep = (Z > 0.2) & (Z < 0.4)
    field = read("catalogue.fits", where="Z > 0.2 && Z < 0.4", **SKY)
    np.testing.assert_array_equal(field.x, RA[keep].astype(field.x.dtype))

    sub = read("catalogue.fits", where="Z > 0.2 && Z < 0.4", fraction=0.5,
               seed=3, **SKY)
    assert sub.n_particles == round(0.5 * keep.sum())
    assert np.isin(sub.x, field.x).all()
    again = read("catalogue.fits", where="Z > 0.2 && Z < 0.4", fraction=0.5,
                 seed=3, **SKY)
    np.testing.assert_array_equal(sub.x, again.x)


def test_several_files():
    halves = [path("part_a.fits"), path("part_b.fits")]
    whole = read("catalogue.fits", fraction=0.4, seed=11, **SKY)
    joined = pysif.io.read_fits(halves, fraction=0.4, seed=11, **SKY)
    np.testing.assert_array_equal(whole.x, joined.x)


def test_hdu_and_paths():
    for hdu in ("OTHER", "other", 2):
        field = read("catalogue.fits", x="X", y="X", z="X", hdu=hdu)
        np.testing.assert_array_equal(field.x, 1000.0 + np.arange(5))
    # os.PathLike, and a compressed file.
    assert pysif.io.read_fits(FITS / "catalogue.fits.gz", **SKY).n_particles == N_ROWS


def test_undefined_values():
    with pytest.raises(ValueError):
        read("catalogue.fits", x="RA", y="DEC", z="ZNULL")
    field = read("catalogue.fits", x="RA", y="DEC", z="ZNULL",
                 where="!ISNULL(ZNULL)")
    assert field.n_particles == N_ROWS - 10


@pytest.mark.parametrize("kwargs", [
    dict(x="RA", y="DEC", z="NOPE"),             # no such column
    dict(x="RA", y="DEC", z="NAME"),             # a string column
    dict(x="RA", y="DEC", z="POS"),              # a whole vector column
    dict(x="RA", y="DEC", z="Z +*"),             # does not parse
    dict(where="Z * 2", **SKY),                  # not a boolean
    dict(where="Z > 5", **SKY),                  # keeps nothing
    dict(hdu="RANDOMS", **SKY),                  # no such HDU
    dict(hdu=0, **SKY),                          # the primary image
    dict(fraction=0.0, **SKY),
    dict(fraction=1.5, **SKY),
    dict(x="RA", dec="DEC", z="Z"),              # positions and sky mixed
    dict(vx="RA", vy="DEC", **SKY),              # half the velocities
])
def test_bad_requests(kwargs):
    with pytest.raises(ValueError):
        read("catalogue.fits", **kwargs)


def test_missing_positions():
    with pytest.raises(TypeError):
        read("catalogue.fits", x="RA", y="DEC")
    with pytest.raises(TypeError):
        read("catalogue.fits", ra="RA", dec="DEC")


def test_options_are_keyword_only():
    with pytest.raises(TypeError):
        pysif.io.read_fits(path("catalogue.fits"), "RA", "DEC", "Z")


def test_io_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read("no_such_file.fits", **SKY)
    with pytest.raises(FileNotFoundError):
        pysif.io.read_fits([path("part_a.fits"), path("no_such_part.fits")], **SKY)

    garbage = tmp_path / "garbage.fits"
    garbage.write_bytes(b"not a FITS file" * 200)
    with pytest.raises(OSError):
        pysif.io.read_fits(garbage, **SKY)


def test_inspect(capsys):
    pysif.io.inspect_fits(path("catalogue.fits"))
    out = capsys.readouterr().out
    assert "[1] GALAXIES" in out and "100 rows, 8 columns" in out
    assert "vector: POS[1] ... POS[3]" in out and "not readable" in out
    with pytest.raises(FileNotFoundError):
        pysif.io.inspect_fits(path("no_such_file.fits"))


# --- catalogues and keywords ------------------------------------------------

COSMOLOGY = dict(omega_m=0.31)


def make_catalog(tmp_path, n=50):
    """A catalogue, through the text format: centres 500-2000 Mpc/h out."""
    rng = np.random.default_rng(5)
    centers = rng.normal(size=(n, 3))
    distance = 500 + 1500 * rng.random(n)
    centers *= (distance / np.linalg.norm(centers, axis=1))[:, None]
    radii = 10 + 20 * rng.random(n)
    path = tmp_path / "voids.txt"
    with open(path, "w") as f:
        f.write(f"{n}\n")
        np.savetxt(f, np.column_stack([centers, radii]))
    return pysif.io.read_catalog_ascii(str(path))


def test_catalog_to_sky(tmp_path):
    cat = make_catalog(tmp_path)
    assert cat.units == "cartesian"
    cartesian = np.array(cat.centers)
    radii = np.array(cat.radii)

    cat.to_sky(**COSMOLOGY)
    assert cat.units == "sky"
    ra, dec, z = np.array(cat.centers).T
    assert ((ra >= 0) & (ra < 360)).all() and (np.abs(dec) <= 90).all()
    np.testing.assert_array_equal(cat.radii, radii)

    # Back through a field: the same positions, within single precision.
    field = pysif.field_from_numpy(ra=ra, dec=dec, z=z)
    field.convert_sky_coordinates(**COSMOLOGY)
    back = np.column_stack([field.x, field.y, field.z])
    np.testing.assert_allclose(back, cartesian, rtol=0, atol=2e-3)

    with pytest.raises(ValueError):
        cat.to_sky(**COSMOLOGY)
    with pytest.raises(ValueError):
        cat.translate((1.0, 1.0, 1.0))


@pytest.mark.parametrize("sky", [False, True])
def test_catalog_round_trips(tmp_path, sky):
    cat = make_catalog(tmp_path)
    if sky:
        cat.to_sky(**COSMOLOGY)

    fits = tmp_path / "voids.fits"
    pysif.io.write_catalog_fits(fits, cat)
    back = pysif.io.read_catalog_fits(fits)
    assert back.units == cat.units
    np.testing.assert_array_equal(back.centers, cat.centers)
    np.testing.assert_array_equal(back.radii, cat.radii)

    # The text format keeps the coordinates too.
    txt = tmp_path / "again.txt"
    pysif.io.write_catalog_ascii(str(txt), cat)
    assert pysif.io.read_catalog_ascii(str(txt)).units == cat.units


def test_catalog_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        pysif.io.read_catalog_fits(tmp_path / "missing.fits")
    with pytest.raises(ValueError):
        pysif.io.read_catalog_fits(path("catalogue.fits"))  # galaxies, not voids


def test_keywords(tmp_path):
    fits = tmp_path / "voids.fits"
    pysif.io.write_catalog_fits(fits, make_catalog(tmp_path))

    pysif.io.set_fits_key(fits, "NTRACER", 2**60 + 1)
    pysif.io.set_fits_key(fits, "search_factor", 1.5)
    pysif.io.set_fits_key(fits, "INPUT", "x" * 200)
    pysif.io.set_fits_key(fits, "THRESH", -0.7, hdu="VOIDS")

    assert pysif.io.fits_key(fits, "ntracer") == 2**60 + 1  # every digit
    assert pysif.io.fits_key(fits, "SEARCH_FACTOR") == 1.5
    assert pysif.io.fits_key(fits, "INPUT") == "x" * 200
    assert pysif.io.fits_key(fits, "SIMPLE") is True
    assert pysif.io.fits_key(fits, "NOPE") is None
    assert pysif.io.fits_key(fits, "THRESH") is None  # not in the primary
    assert pysif.io.fits_key(fits, "THRESH", hdu="VOIDS") == -0.7
    assert pysif.io.fits_key(fits, "THRESH", hdu=1) == -0.7
    assert pysif.io.fits_key(fits, "COORDS", hdu="VOIDS") == "cartesian"

    with pytest.raises(FileNotFoundError):
        pysif.io.fits_key(tmp_path / "missing.fits", "X")
    with pytest.raises(ValueError):
        pysif.io.set_fits_key(fits, "X", 1, hdu="NOHDU")
    with pytest.raises(ValueError):
        pysif.io.set_fits_key(fits, "X", float("nan"))
    with pytest.raises(TypeError):
        pysif.io.set_fits_key(fits, "X", [1, 2])
