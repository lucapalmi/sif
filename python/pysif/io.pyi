# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The types of pysif.io; see __init__.pyi.

"""Reading and writing sif's on-disk formats.

The .xfield and .xgrid binary formats load without parsing or
copying, at the cost of being portable only between machines
that agree on endianness and on the precision sif was built
with; both record a checksum and refuse a file that fails it.
ASCII is the portable path, and far slower; read_field_binary()
reads raw binary files written by anything else, given their layout.

read_gadget() loads GADGET snapshots, in all three formats.

The ``*_hdf5`` functions keep a catalogue and what was measured from
it in one HDF5 file, readable with h5py without pysif.
"""

from collections.abc import Sequence
from os import PathLike
from typing import Any, Literal, TypedDict

from numpy.typing import ArrayLike

from . import Catalogue, Field, Grid, SizeFunction
from .measure import Profiles

_Path = str | bytes | PathLike[str] | PathLike[bytes]
_GadgetFormat = Literal["auto", "hdf5", 1, 2]
_Value = int | float | str

class _FieldHeader(TypedDict):
    n_particles: int
    box_length: float
    has_weights: bool
    has_velocities: bool
    version: int

def write_field(filepath: _Path, box_length: float, x: ArrayLike,
                y: ArrayLike, z: ArrayLike, vx: ArrayLike | None = None,
                vy: ArrayLike | None = None, vz: ArrayLike | None = None,
                weights: ArrayLike | None = None) -> None:
    """Write particle arrays to an .xfield binary.

    Velocity and weight blocks are written only if given, and the header
    records which. A checksum of the payload goes in the header, so a
    reader can tell a truncated or corrupted file from a good one.

    Args:
        filepath: Output path, truncated if it exists.
        box_length: Box size to record. Not carried by the arrays, so it
            has to be supplied here.
        x, y, z: Position components.
        vx, vy, vz: Velocity components, or None.
        weights: Per-particle weights, or None for an unweighted field.
    """

def read_field(filepath: _Path, wrap: bool = False) -> Field:
    """Read an .xfield binary into a Field.

    Args:
        filepath: Input path.
        wrap: Fold every coordinate into [0, box_length) using the box the
            file declares. The short way to clear the single-precision
            rounding that puts a handful of particles exactly on the box
            edge. Off by default, since folding is only correct for a
            genuinely periodic field; use Field.wrap() instead when you
            want the counts back.

    Returns:
        Field: The loaded field.

    Raises:
        IOError: If the file is missing, malformed, written at a different
            precision, or fails its checksum.
    """

def read_field_header(filepath: _Path) -> _FieldHeader:
    """Read only the 64-byte .xfield header.

    Costs one small read rather than the whole file, so it can size a grid
    or a chain mesh before committing to loading the tracers. box_length
    lives in the file and not in the Field, so this is the only way to
    recover it without reading everything.

    Args:
        filepath: Input path.

    Returns:
        dict: n_particles, box_length, has_weights, has_velocities,
        version.
    """

def write_grid(filepath: _Path, grid: Grid) -> None:
    """Write a grid to an .xgrid binary.

    Records whether the cells hold a density or a density contrast, along with a
    checksum of the payload.

    Args:
        filepath: Output path, truncated if it exists.
        grid: Grid to write.
    """

def read_grid(filepath: _Path) -> Grid:
    """Read an .xgrid binary into a Grid.

    Args:
        filepath: Input path.

    Returns:
        Grid: The loaded grid, tagged with what its cells hold.

    Raises:
        IOError: If the file is missing, malformed, written at a different
            precision, or fails its checksum.
    """

def read_field_ascii(filepath: _Path, format: str = "x y z",
                     delimiter: str = " ", skip_lines: int = 0) -> Field:
    """Read a particle field from an ASCII table.

    Far slower than the binary path, and the portable one.

    Args:
        filepath: Input path.
        format: The columns, in order: 'x', 'y', 'z' for positions, or
            'ra', 'dec', 'z' for sky coordinates (a Field with units='sky');
            'vx', 'vy', 'vz' for velocities, 'w' for the per-particle
            weight, '*' to skip a column. Case does not matter, and spaces
            or commas between names are optional: 'x y z * w', 'x,y,z,*,w'
            and 'xyz*w' are the same. Anything else, a name given twice, or a
            position or velocity named in part, is a ValueError.
        delimiter: Column separator; ' ' for whitespace.
        skip_lines: Header lines to skip.

    Returns:
        Field: The loaded field.

    Raises:
        ValueError: For a malformed format.
        OSError: For a file that is missing or holds no data row, or a
            malformed row -- fewer columns than the format, or a value it
            reads that is not a number -- named by its line.
    """

def read_field_binary(filepath: _Path, format: str = "x y z", *,
                      layout: Literal["rows", "blocks"] = "rows",
                      precision: Literal["float32", "float64"] = "float32",
                      byteorder: Literal["native", "little", "big"] = "native",
                      header_bytes: int = 0, n_particles: int = 0,
                      field: Field | None = None) -> Field:
    """Read a raw binary file -- a header of known length, then the values.

    For files no other reader knows: the header is skipped without being
    read, and the arguments say everything about the data.

    Args:
        filepath: Input path.
        format: The columns, as for read_field_ascii(). In a binary file a
            skipped column may carry its width in bytes, '*8', for a column
            of another type -- a 64-bit ID among float32 values, say. So
            'x y z *8 w'. '* * * vx vy vz' reads only the velocities.
        layout: 'rows', one record per particle (x0 y0 z0 x1 y1 z1 ...,
            what an array of structs gives), or 'blocks', one column at a
            time (x0 x1 ... y0 y1 ..., what one array after another gives).
        precision: 'float32' or 'float64', for every value column.
        byteorder: 'native', 'little' or 'big'.
        header_bytes: Bytes to skip before the data.
        n_particles: Particles to read; 0 takes the count from the file,
            which then has to hold a whole number of them.
        field: A Field to read into instead of a new one: blocks it already
            has are kept, so a second file can add columns to the first.
            Its particle count is used, and it must not have been
            Morton-sorted unless the format reloads the positions.

    Returns:
        Field: The loaded field -- the one passed as field, if any.

    Raises:
        ValueError: For a malformed format or an unknown option.
        OSError: For a file that is missing, too short, or does not divide
            into whole particles -- a sign the format, precision or header
            length is not the file's.
    """

def read_hdf5(paths: _Path | Sequence[_Path], *, x: str | None = None,
              y: str | None = None, z: str | None = None,
              ra: str | None = None, dec: str | None = None,
              vx: str | None = None, vy: str | None = None,
              vz: str | None = None, w: str | None = None,
              length_scale: float = 1.0, fraction: float = 1.0,
              seed: int = 0) -> Field:
    """Read particles out of any HDF5 file -- a simulation snapshot, a halo
    catalogue, a mock -- into a Field.

    Each part of the field is a dataset, named by its path; a dataset of
    two dimensions, a row per particle, is read a column at a time, counted
    from 0 as numpy counts. The dark matter of an IllustrisTNG snapshot, in
    Mpc/h:

        field = pysif.io.read_hdf5(
            ["snap_099.0.hdf5", "snap_099.1.hdf5"],
            x="PartType1/Coordinates[0]", y="PartType1/Coordinates[1]",
            z="PartType1/Coordinates[2]", length_scale=1e-3)

    Several files read as one, their rows one after another. A GADGET
    snapshot has read_gadget(), which finds its files and units by itself.

    Args:
        paths: A path, or a list of them.
        x, y, z: Positions; or ra, dec, z for sky coordinates (a Field
            with units='sky').
        vx, vy, vz: Velocities: all three, or none.
        w: Per-particle weight, or None.
        length_scale: What Cartesian positions are multiplied by: 1e-3 from
            kpc/h to Mpc/h. Velocities and weights are read as they are.
        fraction: Share of the rows to keep, in (0, 1], over all the files
            together.
        seed: Seed for the subsample.

    Returns:
        Field: The loaded field.

    Raises:
        ValueError: For a dataset a file does not have or cannot give, row
            counts that disagree, or a value that is not finite.
        OSError: For a file that is missing or not HDF5.
        RuntimeError: If pysif was built without HDF5.
    """

def read_gadget(path: _Path, *, ptype: int = 1,
                velocities: Literal["raw", "peculiar"] | None = None,
                masses: bool = False,
                length: Literal["auto", "kpc", "mpc"] = "auto",
                fraction: float = 1.0, seed: int = 0,
                format: _GadgetFormat = "auto") -> tuple[Field, float]:
    """Read one particle type of a GADGET snapshot into a Field.

    Reads all three snapshot formats: the legacy binaries (SnapFormat 1
    and 2, with either the GADGET-2 or the GADGET-4 header, in either byte
    order and precision) and HDF5. A snapshot split over several files is
    read as one: name any of its files, or the base name, and the rest are
    found -- snap_010.N, snap_010.N.hdf5, and both inside snapdir_010/.

    Args:
        path: A file of the snapshot, or its base name.
        ptype: Particle type, 0 to 5. One per call; 1 is GADGET's dark
            matter.
        velocities: None to skip them; 'raw' for u as GADGET stores it;
            'peculiar' for v = u * sqrt(a), a being the header's Time --
            which is the scale factor only in a cosmological run.
        masses: Read per-particle masses into the weights. A type with a
            mass in the header's table has one mass for every particle, and
            is left unweighted.
        length: The snapshot's length unit: 'auto', the default, reads
            UnitLength_in_cm from an HDF5 file; a binary file records no
            unit, so it has to be named -- 'kpc' (kpc/h, GADGET's default)
            or 'mpc' (Mpc/h). Positions and box come out in Mpc/h.
        fraction: Share of the particles to keep, in (0, 1]. Exactly
            round(fraction * N) are kept, uniformly at random, in file
            order.
        seed: Seed for the subsample: the same seed and files give the
            same particles.
        format: 'auto', 1, 2 or 'hdf5'. Anything but 'auto' is checked
            against the file.

    Returns:
        tuple: (Field, box_length), the box in Mpc/h. Returned alongside
        the field because it is converted with it, and the header's
        box_size is in the file's own unit.

    Raises:
        ValueError: For an argument out of range, a type with no
            particles, a fraction that keeps none, or length='auto' (the
            default) on a file that records no unit -- every binary one.
        OSError: For a missing, truncated or inconsistent snapshot.
        RuntimeError: For an HDF5 snapshot, if pysif was built without
            HDF5.
    """

def gadget_header(path: _Path, format: _GadgetFormat = "auto") -> dict[str, Any]:
    """Read the header of one GADGET snapshot file.

    Only the first file of a multi-file snapshot is read, so n_part_file
    is that file's and n_part_total the snapshot's.

    Returns:
        dict: format (1, 2 or 'hdf5'), swapped and legacy_header (None for
        HDF5), precision (bytes per position component), n_types,
        n_part_file, n_part_total, mass_table, time, redshift, box_size
        (in the file's unit), n_files, cosmology (a dict, or None if the
        file has none), unit_length_in_cm (None if not recorded),
        n_blocks, blocks (names, or None for SnapFormat 1).
    """

def inspect_gadget(path: _Path, format: _GadgetFormat = "auto") -> None:
    """Print a GADGET snapshot's header: format, precision, particle counts
    per type, masses, time, box, cosmology and blocks. For a look at a
    snapshot before deciding how to read it; gadget_header() returns the
    same as a dict.
    """

def read_fits(paths: _Path | Sequence[_Path], *, x: str | None = None,
              y: str | None = None, z: str | None = None,
              ra: str | None = None, dec: str | None = None,
              vx: str | None = None, vy: str | None = None,
              vz: str | None = None, w: str | None = None,
              where: str | None = None, hdu: int | str | None = None,
              fraction: float = 1.0, seed: int = 0) -> Field:
    """Read a catalogue of one or more FITS tables into a Field.

    Each part of the field is named by a column of the table or by an
    arithmetic expression over columns, in cfitsio's syntax -- how a
    weight built from several columns, or an element of a vector column,
    is read:

        field = pysif.io.read_fits(
            ["galaxies_NGC.fits", "galaxies_SGC.fits"],
            ra="RA", dec="DEC", z="Z",
            w="WEIGHT_SYSTOT * (WEIGHT_NOZ + WEIGHT_CP - 1)",
            where="Z > 0.43 && Z < 0.7")

    Several files read as one table, their rows one after another. Rows
    the filter drops are never read, and the subsample is drawn from the
    rows it keeps, over all the files together -- so data and randoms read
    with the same filter are cut identically, whatever share of each is
    kept.

    Args:
        paths: A path, or a list of them. Compressed files (.fits.gz) are
            read, uncompressed in memory.
        x, y, z: Positions, read as they are.
        ra, dec: Sky coordinates in place of x and y -- right ascension and
            declination in degrees, with z the redshift. The Field then has
            units='sky', to be turned into positions with
            Field.convert_sky_coordinates().
        vx, vy, vz: Velocities: all three, or none.
        w: Per-row weight, or None for an unweighted field.
        where: Row filter, a boolean expression: 'Z > 0.43 && Z < 0.7',
            '!ISNULL(Z)'.
        hdu: The table in each file: its EXTNAME, or its extension number
            (1 for the first). None reads the first table.
        fraction: Share of the filtered rows to keep, in (0, 1]. Exactly
            round(fraction * N) are kept, uniformly at random, in file
            order.
        seed: Seed for the subsample: the same seed and files give the
            same rows.

    Returns:
        Field: The loaded field.

    Raises:
        ValueError: For a request the files cannot satisfy: a column, HDU
            or expression they do not have, a filter that keeps nothing,
            an undefined value (NaN, TNULL) in a row the filter keeps.
        OSError: For a file that is missing or is not readable as FITS.
        RuntimeError: If pysif was built without FITS support.
    """

def inspect_fits(path: _Path) -> None:
    """Print what a FITS file holds: every HDU, and for each table its row
    count and its columns, with their type, their unit, and a note for a
    vector column (read as NAME[1] ... NAME[n]) or one read_fits() cannot
    use. For finding the names to pass to read_fits().
    """

def write_catalogue_fits(filepath: _Path, catalogue: Catalogue) -> None:
    """Write a catalogue as a FITS table: an empty primary HDU, then a binary
    table named VOIDS, one row per void.

    The columns follow Catalogue.units: CX, CY, CZ, R for Cartesian centres;
    RA, DEC (degrees), Z (redshift), R (comoving, Mpc/h) for a
    catalogue on the sky -- see Catalogue.to_sky(). FOOTPRINT and
    FOOTPRINT_SHELL follow when the catalogue has them. The primary header
    is left for set_fits_key(), to record what the catalogue came from.

    Args:
        filepath: Output path, replaced if it exists.
        catalogue: Catalogue to write.
    """

def read_catalogue_fits(filepath: _Path) -> Catalogue:
    """Read a catalogue written by write_catalogue_fits(): on the sky if its
    table has RA, DEC and Z columns, Cartesian if CX, CY and CZ (or X, Y
    and Z). R or RADIUS is the radius.

    Returns:
        Catalogue: The loaded catalogue.

    Raises:
        FileNotFoundError: For a missing file.
        ValueError: For a file with no VOIDS table of those columns.
    """

def write_profiles_fits(filepath: _Path, profiles: Profiles) -> None:
    """Write stacked profiles into DENSITY_PROFILES and VELOCITY_PROFILES:
    a row per void, the profile as one vector column, the shape and the
    bin edges (EDGE0, EDGE1, ...) as keywords. Only the sets the Profiles
    carries are written; the other is left in the file as it was.

    A FITS file holds a catalogue and its products one table each, as the
    HDF5 file does a group each: every writer creates the file if it is
    missing and replaces only its own table, and a FITS file sif did not
    write is never written into.

    Args:
        filepath: The file.
        profiles: Profiles to write.
    """

def read_profiles_fits(filepath: _Path) -> Profiles:
    """Read the profiles write_profiles_fits() wrote: whichever sets the file
    holds.

    Returns:
        Profiles: The loaded profiles.

    Raises:
        FileNotFoundError: For a missing file.
        ValueError: For a file that holds no profiles.
    """

def write_size_function_fits(filepath: _Path,
                             size_function: SizeFunction) -> None:
    """Write a size function into SIZE_FUNCTION: a row per bin -- R_LOW,
    R_HIGH, R_CENTRE, COUNT, VSF, ERR -- with R_MIN, R_MAX and BINNING
    ('ln' or 'linear', what VSF is per unit of) as keywords. As for
    write_profiles_fits(), only its own table is replaced.

    Args:
        filepath: The file.
        size_function: Size function to write, measured or modelled.
    """

def read_size_function_fits(filepath: _Path) -> SizeFunction:
    """Read the size function write_size_function_fits() wrote.

    Returns:
        SizeFunction: The loaded size function.

    Raises:
        FileNotFoundError: For a missing file.
        ValueError: For a file that holds none.
    """

def fits_key(path: _Path, key: str,
             hdu: int | str | None = None) -> _Value | bool | None:
    """A keyword's value from a FITS header, or None if it is not there.

    The value comes back as the header has it: int, float, bool (a FITS
    logical) or str. Names are matched without regard to case; a name
    longer than eight characters is a HIERARCH keyword.

    Args:
        path: The file.
        key: The keyword's name.
        hdu: None for the primary header, where a file keeps what describes
            it as a whole; otherwise an extension number or an EXTNAME.
            (For read_fits(), None is the first table instead.)

    Raises:
        FileNotFoundError: For a missing file. An HDU the file does not have
            reads as None, and is logged.
    """

def set_fits_key(path: _Path, key: str, value: _Value,
                 hdu: int | str | None = None) -> None:
    """Write a keyword into a FITS header, replacing one of the same name.

    Args:
        path: The file, which has to exist.
        key: The keyword's name; longer than eight characters, it is
            written as a HIERARCH keyword.
        value: An int, a float or a str. A bool is written as 1 or 0; a
            string longer than a header card holds goes over several.
        hdu: As for fits_key(): None for the primary header.

    Raises:
        ValueError: For an HDU the file does not have, a name FITS does not
            allow, or a value that is not finite.
        OSError: If the file could not be written.
    """

def write_catalogue_ascii(filepath: _Path, catalogue: Catalogue) -> None:
    """Write a catalogue as text: a two-line header behind '#', then a row
    per void.

        #n=1024
        #cx cy cz r footprint footprint_shell
        101.25 250.5 33.125 12.5 1 0.84

    The first line is the count, the second names the columns: 'ra dec z
    r' for a catalogue on the sky (Catalogue.to_sky()), and the footprint
    columns only when the catalogue carries one, as one from
    finders.exodus_survey() does. numpy.loadtxt(filepath) reads the rows
    as they are.

    Written with enough significant digits to recover the stored values
    exactly, so a write/read round trip is lossless.

    Args:
        filepath: Output path, truncated if it exists.
        catalogue: Catalogue to write.
    """

def read_catalogue_ascii(filepath: _Path, format: str | None = None) -> Catalogue:
    """Read a catalogue written by write_catalogue_ascii().

    Lines starting with '#' are comments. Before the first row, 'n=N'
    gives the count -- without it the rows are counted first -- and a
    comment naming the columns places them, in any order: 'cx cy cz' (or
    'x y z') or 'ra dec z' for the centres, 'r' (or 'radius'), and
    optionally 'footprint footprint_shell'; columns with other names are
    skipped. Without names, rows are 'cx cy cz r', and six columns add the
    footprint. Files written before the header, whose first line is the
    count alone, read as they always did.

    A catalogue another finder wrote is read with a format, which places
    the columns in the language of the field readers: 'x y z' (or
    'cx cy cz') or 'ra dec z' for the centre, 'r' for the radius, '*' for
    a column not read -- '* x y z * r' for rows of an ID, the centre, a
    volume and the radius. Names in the file are then ignored. Columns may
    be separated by blanks or commas, and a skipped column, or one past
    the last named, may hold anything.

    Args:
        filepath: Input path.
        format: The columns, or None to take them from the file.

    Returns:
        Catalogue: The loaded catalogue.
    """

def write_profiles_ascii(filepath: _Path, profiles: Profiles,
                         catalogue: Catalogue) -> None:
    """Write stacked profiles as text, one row per void.

    A header behind '#', then a row per void -- the void it belongs to,
    then its profile:

        #n=1024
        #n_bins=20
        #ext=3
        #differential=1
        #r_edges=0 0.15 0.3 ... 3
        #cx cy cz r density_0 ... density_19 v_rad_0 ... v_rad_19
        101.25 250.5 33.125 12.5 -0.91 ... 12.3

    The rows are plain numbers, so numpy.loadtxt(path) reads them as they
    are, and the last header line names every column: the centres are
    'ra dec z' for a catalogue on the sky (Catalogue.to_sky()), and only the
    profile blocks the Profiles carries are there. Bin edges are in units
    of each void's own radius; multiply by the radius in the row for
    physical units.

    Written with enough significant digits to recover the stored values
    exactly, so a write/read round trip is lossless.

    Args:
        filepath: Output path, truncated if it exists.
        profiles: Profiles to write.
        catalogue: The catalogue they were measured from, which is where the
            per-void columns come from. Row i is void i, so it has to be
            that catalogue and not another of the same length.
    """

def read_profiles_ascii(filepath: _Path) -> tuple[Profiles, Catalogue]:
    """Read profiles written by write_profiles_ascii().

    Args:
        filepath: Input path.

    Returns:
        tuple: (Profiles, Catalogue). The Profiles carries whichever blocks
        the file holds; has_density and has_velocity say which.
    """

def write_catalogue_hdf5(filepath: _Path, catalogue: Catalogue) -> None:
    """Write a catalogue into /catalogue of an HDF5 file.

    The file can hold any subset of catalogue, density_profiles,
    velocity_profiles and size_function, each in its own group; this
    replaces only its own and creates the file if it is missing. It reads
    back with h5py without pysif. An existing file sif did not write is
    never overwritten.

    In a pysif built without HDF5 the data is saved as plain text in
    <filepath>.<product>.txt instead, with a RuntimeWarning saying so.

    Args:
        filepath: The file.
        catalogue: Catalogue to write, with its footprint if it has one.

    Raises:
        OSError: If the file could not be written, or is not sif's.
    """

def read_catalogue_hdf5(filepath: _Path) -> Catalogue:
    """Read /catalogue of an HDF5 file, footprint included when present.

    Returns:
        Catalogue: The catalogue.

    Raises:
        OSError: If the file or the group is missing or malformed.
        RuntimeError: If pysif was built without HDF5.
    """

def write_profiles_hdf5(filepath: _Path, profiles: Profiles) -> None:
    """Write stacked profiles into /density_profiles and /velocity_profiles.

    Only the sets the Profiles holds are written; one it does not hold
    stays in the file as it was. A row count that disagrees with the
    catalogue in the file is warned about in the log, not refused.

    The file can hold any subset of catalogue, density_profiles,
    velocity_profiles and size_function, each in its own group; this
    replaces only its own and creates the file if it is missing. It reads
    back with h5py without pysif. An existing file sif did not write is
    never overwritten.

    In a pysif built without HDF5 the data is saved as plain text in
    <filepath>.<product>.txt instead, with a RuntimeWarning saying so.

    Args:
        filepath: The file.
        profiles: Profiles to write.
    """

def read_profiles_hdf5(filepath: _Path) -> Profiles:
    """Read whichever profile sets an HDF5 file holds.

    Returns:
        Profiles: has_density and has_velocity say which came back.

    Raises:
        ValueError: If the file holds no profiles.
        OSError: If the file is missing or malformed.
        RuntimeError: If pysif was built without HDF5.
    """

def write_size_function_hdf5(filepath: _Path,
                             size_function: SizeFunction) -> None:
    """Write a size function, measured or modelled, into /size_function.

    The binning goes in as the attribute 'binning' ('ln' or 'linear'),
    since it decides whether vsf is per unit ln R or per unit R.

    The file can hold any subset of catalogue, density_profiles,
    velocity_profiles and size_function, each in its own group; this
    replaces only its own and creates the file if it is missing. It reads
    back with h5py without pysif. An existing file sif did not write is
    never overwritten.

    In a pysif built without HDF5 the data is saved as plain text in
    <filepath>.<product>.txt instead, with a RuntimeWarning saying so.

    Args:
        filepath: The file.
        size_function: SizeFunction to write.
    """

def read_size_function_hdf5(filepath: _Path) -> SizeFunction:
    """Read /size_function of an HDF5 file.

    Returns:
        SizeFunction: The size function.

    Raises:
        OSError: If the file or the group is missing or malformed.
        RuntimeError: If pysif was built without HDF5.
    """

def set_hdf5_attr(filepath: _Path, key: str, value: _Value,
                  group: str | None = None) -> None:
    """Attach a named value to an HDF5 file, or to one of its products.

    The file's own header -- how the catalogue was made, which simulation,
    which snapshot -- the way FITS keywords are. On the root (group None)
    an entry describes the file and survives every rewrite; on a product
    ('catalogue', 'size_function', ...) it describes that product and goes
    when the product is rewritten.

    sif's own attributes (n_voids, n_bins, ... and every root name
    beginning ``sif_``) cannot be set. Without HDF5 the entry is appended to
    <filepath>.attributes.txt instead, with a RuntimeWarning.

    Args:
        filepath: The file; for the root it is created if missing.
        key: The entry's name.
        value: An int (or bool), float or str.
        group: None for the file itself, or a product that is already
            in it.

    Raises:
        ValueError: For a missing product or a name sif keeps.
        TypeError: For a value that is not int, float or str.
    """

def get_hdf5_attr(filepath: _Path, key: str, group: str | None = None) -> _Value:
    """Read one named value from an HDF5 file or one of its products.

    Returns:
        int, float or str.

    Raises:
        OSError: If the file, the group or the entry is missing.
        TypeError: If the entry is an array or something else these
            functions do not read.
        RuntimeError: If pysif was built without HDF5.
    """

def get_hdf5_attrs(filepath: _Path,
                   group: str | None = None) -> dict[str, _Value | None]:
    """Every named value of an HDF5 file, or of one of its products.

    sif's own entries are included. One of a kind these functions do not
    read (an array another tool put there) is listed as None.

    Returns:
        dict: name -> int, float, str or None.

    Raises:
        OSError: If the file or the group is missing.
        RuntimeError: If pysif was built without HDF5.
    """
