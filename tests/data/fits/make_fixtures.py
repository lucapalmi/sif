# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Write the small synthetic FITS catalogues the pysif tests read.

Written from the FITS standard (version 4.0, sections 3, 4 and 7.3) with numpy
alone, independently of cfitsio, so a test that passes means sif's reader and
the standard agree, not merely that cfitsio reads back what cfitsio wrote. The
files are committed, so the tests need nothing but pysif; rerun this only to
change them:

    python tests/data/fits/make_fixtures.py

One catalogue of 100 rows, whole and split over two files. Row i (counting
across files) has

    RA  = 3.5 i            (deg, float64)
    DEC = -45 + 0.75 i     (deg, float32)
    Z   = 0.1 + 0.005 i    (float64)
    W1  = 1 + (i % 4) / 4  (float32)
    W2  = 2 + i % 3        (int32)
    POS = (3i, 3i + 1, 3i + 2)                       (float64 vector)
    ZNULL = Z, or NaN in rows 7, 17, 27, ...         (float64)
    NAME = "gal"           (string, which the reader refuses)

all exact in single precision except Z, which the tests compare with a
tolerance. The table is extension 1, named GALAXIES, after an empty primary
HDU; extension 2 is a second table, OTHER, of 5 rows with one column X = 1000
+ i.
"""

import gzip
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

N_ROWS = 100
SPLIT = 60
BLOCK = 2880


def card(key, value=None, comment=""):
    """One 80-character header card: fixed format, as section 4.2 has it."""
    if value is None:
        text = f"{key:<8}"
    elif isinstance(value, bool):
        text = f"{key:<8}= {'T' if value else 'F':>20}"
    elif isinstance(value, (int, np.integer)):
        text = f"{key:<8}= {int(value):>20}"
    else:
        quoted = "'" + str(value).replace("'", "''").ljust(8) + "'"
        text = f"{key:<8}= {quoted:<20}"
    if comment:
        text += f" / {comment}"
    assert len(text) <= 80, text
    return text.ljust(80)


def header(cards):
    """Cards, then END, padded with spaces to a whole block."""
    text = "".join(cards) + "END".ljust(80)
    return (text + " " * (-len(text) % BLOCK)).encode("ascii")


def data(raw):
    """Data, padded with zeros to a whole block."""
    return raw + b"\0" * (-len(raw) % BLOCK)


def primary():
    return header([card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0),
                   card("EXTEND", True)])


def bintable(name, columns, n_rows):
    """A binary table: columns is a list of (TTYPE, TFORM, TUNIT, array),
    array of shape (n_rows,) or (n_rows, repeat), written big-endian (section
    7.3.3) row by row."""
    dtype = []
    for ttype, tform, _, arr in columns:
        code = tform.lstrip("0123456789")
        repeat = int(tform[: len(tform) - len(code)] or 1)
        base = {"D": ">f8", "E": ">f4", "J": ">i4", "A": f"S{repeat}"}[code]
        shape = () if code == "A" or repeat == 1 else (repeat,)
        dtype.append((ttype, base, shape))
    rows = np.zeros(n_rows, dtype=dtype)
    for ttype, _, _, arr in columns:
        rows[ttype] = arr

    cards = [card("XTENSION", "BINTABLE"), card("BITPIX", 8), card("NAXIS", 2),
             card("NAXIS1", rows.dtype.itemsize), card("NAXIS2", n_rows),
             card("PCOUNT", 0), card("GCOUNT", 1),
             card("TFIELDS", len(columns))]
    for n, (ttype, tform, tunit, _) in enumerate(columns, start=1):
        cards.append(card(f"TTYPE{n}", ttype))
        cards.append(card(f"TFORM{n}", tform))
        if tunit:
            cards.append(card(f"TUNIT{n}", tunit))
    cards.append(card("EXTNAME", name))
    return header(cards) + data(rows.tobytes())


def catalogue(first, n):
    i = np.arange(first, first + n)
    z = 0.1 + 0.005 * i
    znull = np.where(i % 10 == 7, np.nan, z)
    pos = np.stack([3.0 * i, 3.0 * i + 1, 3.0 * i + 2], axis=1)
    columns = [
        ("RA", "1D", "deg", 3.5 * i),
        ("DEC", "1E", "deg", -45 + 0.75 * i),
        ("Z", "1D", "", z),
        ("W1", "1E", "", 1 + (i % 4) / 4),
        ("W2", "1J", "", 2 + i % 3),
        ("POS", "3D", "Mpc/h", pos),
        ("ZNULL", "1D", "", znull),
        ("NAME", "8A", "", np.full(n, b"gal")),
    ]
    other = [("X", "1D", "", 1000.0 + np.arange(5))]
    return (primary() + bintable("GALAXIES", columns, n)
            + bintable("OTHER", other, 5))


def main():
    whole = catalogue(0, N_ROWS)
    files = {
        "catalogue.fits": whole,
        "part_a.fits": catalogue(0, SPLIT),
        "part_b.fits": catalogue(SPLIT, N_ROWS - SPLIT),
    }
    for name, raw in files.items():
        with open(os.path.join(HERE, name), "wb") as f:
            f.write(raw)
    # mtime=0, so rerunning gives the same bytes.
    with open(os.path.join(HERE, "catalogue.fits.gz"), "wb") as f:
        f.write(gzip.compress(whole, mtime=0))


if __name__ == "__main__":
    main()
