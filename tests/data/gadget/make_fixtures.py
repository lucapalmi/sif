# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Write the small synthetic GADGET snapshots test_gadget reads.

Written from the GADGET-4 manual ("Snapshot file format") with numpy and h5py,
independently of sif's reader, so a test that passes means the two agree on the
format and not merely that the reader reads back what it wrote. The files are
committed, so the tests need neither numpy nor h5py; rerun this only to change
them:

    python tests/data/gadget/make_fixtures.py

Every snapshot holds the same particles, split differently over its files.
Particle g of type t (g counting across files, in file order) has

    x  = 100 t + 0.5 g + 0.125     y = x + 1000     z = x + 2000
    vx = 10 t + 0.25 g             vy = -vx         vz = vx + 1
    m  = 1 + g / 1024              (types 0 and 2; type 1 has a table mass)

all exact in single precision, so the test compares with ==. Lengths are in
kpc/h, the box is 10 000 kpc/h and the scale factor is 0.25.
"""

import os
import shutil

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

N_TYPES = 6
N_TOTAL = np.array([40, 100, 30, 0, 0, 0], dtype=np.int64)
MASS_TABLE = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
TIME = 0.25
REDSHIFT = 1.0 / TIME - 1.0
BOX = 10000.0
OMEGA0, OMEGA_L, HUBBLE = 0.3, 0.7, 0.7

# Per-file counts: uneven, and with a file holding no type-2 particles, so the
# reader has to get the per-type offsets right in every file.
SPLIT_3 = np.array([[15, 30, 0, 0, 0, 0],
                    [5, 45, 18, 0, 0, 0],
                    [20, 25, 12, 0, 0, 0]], dtype=np.int64)
SPLIT_2 = np.array([[25, 60, 10, 0, 0, 0],
                    [15, 40, 20, 0, 0, 0]], dtype=np.int64)
SPLIT_1 = N_TOTAL[None, :]


def particles(t):
    g = np.arange(N_TOTAL[t], dtype=np.float64)
    x = 100.0 * t + 0.5 * g + 0.125
    pos = np.stack([x, x + 1000.0, x + 2000.0], axis=1)
    vx = 10.0 * t + 0.25 * g
    vel = np.stack([vx, -vx, vx + 1.0], axis=1)
    mass = 1.0 + g / 1024.0
    return pos, vel, mass


def file_slices(split):
    """For each file, for each type, the [lo, hi) range of global indices."""
    starts = np.vstack([np.zeros(N_TYPES, dtype=np.int64),
                        np.cumsum(split, axis=0)[:-1]])
    return [[(starts[i, t], starts[i, t] + split[i, t]) for t in range(N_TYPES)]
            for i in range(len(split))]


def file_data(split, i):
    """Positions, velocities, IDs, masses and gas energies of file i, ordered
    by type as the binary formats store them."""
    pos, vel, ids, mass, u = [], [], [], [], []
    for t, (lo, hi) in enumerate(file_slices(split)[i]):
        p, v, m = particles(t)
        pos.append(p[lo:hi])
        vel.append(v[lo:hi])
        ids.append(np.arange(lo, hi) + 1_000_000 * t)
        if MASS_TABLE[t] == 0:
            mass.append(m[lo:hi])
        if t == 0:
            u.append(np.full(hi - lo, 42.0))
    return (np.concatenate(pos), np.concatenate(vel), np.concatenate(ids),
            np.concatenate(mass), np.concatenate(u))


# --- the binary formats ---

def record(payload, bo):
    n = np.array([len(payload)], dtype=bo + "u4").tobytes()
    return n + payload + n


def legacy_header(npart, n_files, bo):
    """The 256-byte GADGET-2/3 header."""
    h = b""
    h += np.asarray(npart, dtype=bo + "i4").tobytes()                # npart
    h += MASS_TABLE.astype(bo + "f8").tobytes()                      # mass
    h += np.array([TIME, REDSHIFT], dtype=bo + "f8").tobytes()       # time, z
    h += np.array([0, 0], dtype=bo + "i4").tobytes()                 # flags
    h += (N_TOTAL & 0xFFFFFFFF).astype(bo + "u4").tobytes()          # nall
    h += np.array([0, n_files], dtype=bo + "i4").tobytes()           # cooling, files
    h += np.array([BOX, OMEGA0, OMEGA_L, HUBBLE], dtype=bo + "f8").tobytes()
    h += np.array([0, 0], dtype=bo + "i4").tobytes()                 # flags
    h += (N_TOTAL >> 32).astype(bo + "u4").tobytes()                 # high word
    assert len(h) == 192
    return h + b"\0" * (256 - len(h))


def g4_header(npart, n_files, bo, npart_width):
    """The GADGET-4 header as one fwrite() of the C struct writes it: per-file
    counts, 64-bit totals, masses, time, redshift, box, file count, padded to
    8 bytes."""
    h = np.asarray(npart, dtype=bo + ("i8" if npart_width == 8 else "u4")).tobytes()
    h += b"\0" * (-len(h) % 8)
    h += N_TOTAL.astype(bo + "i8").tobytes()
    h += MASS_TABLE.astype(bo + "f8").tobytes()
    h += np.array([TIME, REDSHIFT, BOX], dtype=bo + "f8").tobytes()
    h += np.array([n_files], dtype=bo + "i4").tobytes()
    return h + b"\0" * (-len(h) % 8)


def write_binary(path, split, i, *, fmt, header, bo="<", real="f4"):
    pos, vel, ids, mass, u = file_data(split, i)
    n_files = len(split)
    head = (legacy_header(split[i], n_files, bo) if header == "legacy"
            else g4_header(split[i], n_files, bo, 8 if header == "g4" else 4))

    blocks = [("HEAD", head),
              ("POS ", pos.astype(bo + real).tobytes()),
              ("VEL ", vel.astype(bo + real).tobytes()),
              ("ID  ", ids.astype(bo + "u4").tobytes())]
    if len(mass):
        blocks.append(("MASS", mass.astype(bo + real).tobytes()))
    if len(u):
        blocks.append(("U   ", u.astype(bo + real).tobytes()))

    with open(path, "wb") as f:
        for label, payload in blocks:
            if fmt == 2:
                nxt = np.array([len(payload) + 8], dtype=bo + "i4").tobytes()
                f.write(record(label.encode() + nxt, bo))
            f.write(record(payload, bo))


# --- HDF5 ---

def write_hdf5(path, split, i, *, style, real="f4"):
    """style "g4": PartType groups, 64-bit counts, cosmology and units in
    /Parameters. style "legacy": ParticleType groups (as the manual names
    them), 32-bit counts with a high word, cosmology in /Header, units in
    /Units."""
    n_files = len(split)
    with h5py.File(path, "w") as f:
        hd = f.create_group("Header")
        if style == "g4":
            hd.attrs["NumPart_ThisFile"] = split[i].astype(np.int64)
            hd.attrs["NumPart_Total"] = N_TOTAL.astype(np.uint64)
        else:
            hd.attrs["NumPart_ThisFile"] = split[i].astype(np.int32)
            hd.attrs["NumPart_Total"] = (N_TOTAL & 0xFFFFFFFF).astype(np.uint32)
            hd.attrs["NumPart_Total_HighWord"] = (N_TOTAL >> 32).astype(np.uint32)
        hd.attrs["MassTable"] = MASS_TABLE
        hd.attrs["Time"] = TIME
        hd.attrs["Redshift"] = REDSHIFT
        hd.attrs["BoxSize"] = BOX
        hd.attrs["NumFilesPerSnapshot"] = np.int32(n_files)

        cosmo = {"Omega0": OMEGA0, "OmegaLambda": OMEGA_L, "HubbleParam": HUBBLE}
        if style == "g4":
            par = f.create_group("Parameters")
            par.attrs.update(cosmo)
            par.attrs["UnitLength_in_cm"] = 3.085678e21
            par.attrs["ComovingIntegrationOn"] = np.int32(1)
            f.create_group("Config")
        else:
            hd.attrs.update(cosmo)
            f.create_group("Units").attrs["UnitLength_in_cm"] = 3.085678e21

        prefix = "PartType" if style == "g4" else "ParticleType"
        for t, (lo, hi) in enumerate(file_slices(split)[i]):
            if hi == lo:
                continue
            p, v, m = particles(t)
            g = f.create_group(f"{prefix}{t}")
            g.create_dataset("Coordinates", data=p[lo:hi].astype(real))
            g.create_dataset("Velocities", data=v[lo:hi].astype(real))
            g.create_dataset("ParticleIDs",
                             data=(np.arange(lo, hi) + 1_000_000 * t).astype(np.uint64))
            if MASS_TABLE[t] == 0:
                g.create_dataset("Masses", data=m[lo:hi].astype(real))


def fresh(*parts):
    d = os.path.join(HERE, *parts)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def write_ntypes2(path, counts, n_files):
    """A GADGET-4 run compiled with NTYPES=2, as dark-matter-only runs often
    are: the header has two entries per array rather than six, only type 1 is
    populated, its mass is in the table, and so there is no MASS block --
    HEAD, POS, VEL, ID and nothing else. Single precision, 64-bit counts. Its
    particles are the type-1 particles of every other fixture."""
    bo = "<"
    total = np.array([0, N_TOTAL[1]], dtype=np.int64)
    mass = np.array([0.0, MASS_TABLE[1]])

    h = np.array([0, counts[1]], dtype=bo + "i8").tobytes()
    h += total.astype(bo + "i8").tobytes()
    h += mass.astype(bo + "f8").tobytes()
    h += np.array([TIME, REDSHIFT, BOX], dtype=bo + "f8").tobytes()
    h += np.array([n_files], dtype=bo + "i4").tobytes()
    h += b"\0" * (-len(h) % 8)

    lo, hi = counts[0], counts[0] + counts[1]  # counts[0]: type-1 offset
    p, v, _ = particles(1)
    with open(path, "wb") as f:
        for payload in [h,
                        p[lo:hi].astype(bo + "f4").tobytes(),
                        v[lo:hi].astype(bo + "f4").tobytes(),
                        np.arange(lo, hi, dtype=bo + "u4").tobytes()]:
            f.write(record(payload, bo))


def main():
    # SnapFormat 1, GADGET-2 header, single precision, three files side by side.
    d = fresh("f1_legacy")
    for i in range(3):
        write_binary(os.path.join(d, f"snap_005.{i}"), SPLIT_3, i,
                     fmt=1, header="legacy")

    # SnapFormat 1, GADGET-4 header, double precision, in a snapdir.
    d = fresh("f1_g4")
    os.makedirs(os.path.join(d, "snapdir_005"))
    for i in range(2):
        write_binary(os.path.join(d, "snapdir_005", f"snap_005.{i}"), SPLIT_2,
                     i, fmt=1, header="g4", real="f8")

    # SnapFormat 1, GADGET-4 header with 32-bit per-file counts (as the manual
    # tabulates them), one file with an unnumbered name.
    d = fresh("f1_g4_u32")
    write_binary(os.path.join(d, "snap_005"), SPLIT_1, 0, fmt=1,
                 header="g4_u32")

    # SnapFormat 2, GADGET-2 header, big-endian: the byte-swapped case.
    d = fresh("f2_swapped")
    for i in range(3):
        write_binary(os.path.join(d, f"snap_005.{i}"), SPLIT_3, i, fmt=2,
                     header="legacy", bo=">")

    # SnapFormat 1, GADGET-4 header with NTYPES=2, dark matter only, two
    # files in a snapdir. The shape of a real dark-matter-only GADGET-4 run.
    d = fresh("f1_g4_ntypes2")
    os.makedirs(os.path.join(d, "snapdir_005"))
    for i, (offset, n) in enumerate([(0, 60), (60, 40)]):
        write_ntypes2(os.path.join(d, "snapdir_005", f"snap_005.{i}"),
                      (offset, n), 2)

    # HDF5, GADGET-4 layout, three files.
    d = fresh("hdf5_g4")
    for i in range(3):
        write_hdf5(os.path.join(d, f"snap_005.{i}.hdf5"), SPLIT_3, i,
                   style="g4")

    # HDF5, older layout, double precision, a single file.
    d = fresh("hdf5_legacy")
    write_hdf5(os.path.join(d, "snap_005.hdf5"), SPLIT_1, 0, style="legacy",
               real="f8")


if __name__ == "__main__":
    main()
