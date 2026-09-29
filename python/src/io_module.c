/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#define Py_MODULE_HEAD_UNIFIED
#include "io/py_io.h"

static PyMethodDef io_methods[] = {
  {"write_field", (PyCFunction)pysif_write_field, METH_VARARGS | METH_KEYWORDS,
    "write_field(filepath, box_length, x, y, z, vx=None, vy=None, vz=None, "
    "weights=None)\n"
    "--\n\n"
    "Write particle arrays to an .xfield binary.\n\n"
    "Velocity and weight blocks are written only if given, and the header\n"
    "records which. A checksum of the payload goes in the header, so a\n"
    "reader can tell a truncated or corrupted file from a good one.\n\n"
    "Args:\n"
    "    filepath: Output path, truncated if it exists.\n"
    "    box_length: Box size to record. Not carried by the arrays, so it\n"
    "        has to be supplied here.\n"
    "    x, y, z: Position components.\n"
    "    vx, vy, vz: Velocity components, or None.\n"
    "    weights: Per-particle weights, or None for an unweighted field."},

  {"read_field", (PyCFunction)pysif_read_field, METH_VARARGS | METH_KEYWORDS,
    "read_field(filepath, wrap=False)\n"
    "--\n\n"
    "Read an .xfield binary into a Field.\n\n"
    "Args:\n"
    "    filepath: Input path.\n"
    "    wrap: Fold every coordinate into [0, box_length) using the box the\n"
    "        file declares. The short way to clear the single-precision\n"
    "        rounding that puts a handful of particles exactly on the box\n"
    "        edge. Off by default, since folding is only correct for a\n"
    "        genuinely periodic field; use Field.wrap() instead when you\n"
    "        want the counts back.\n\n"
    "Returns:\n"
    "    Field: The loaded field.\n\n"
    "Raises:\n"
    "    IOError: If the file is missing, malformed, written at a different\n"
    "        precision, or fails its checksum."},

  {"read_field_header", (PyCFunction)pysif_read_field_header,
    METH_VARARGS | METH_KEYWORDS,
    "read_field_header(filepath)\n"
    "--\n\n"
    "Read only the 64-byte .xfield header.\n\n"
    "Costs one small read rather than the whole file, so it can size a grid\n"
    "or a chain mesh before committing to loading the tracers. box_length\n"
    "lives in the file and not in the Field, so this is the only way to\n"
    "recover it without reading everything.\n\n"
    "Args:\n"
    "    filepath: Input path.\n\n"
    "Returns:\n"
    "    dict: n_particles, box_length, has_weights, has_velocities,\n"
    "    version."},

  {"write_grid", (PyCFunction)pysif_write_grid, METH_VARARGS | METH_KEYWORDS,
    "write_grid(filepath, grid)\n"
    "--\n\n"
    "Write a grid to an .xgrid binary.\n\n"
    "Records whether the cells hold a density or a density contrast, along "
    "with a\n"
    "checksum of the payload.\n\n"
    "Args:\n"
    "    filepath: Output path, truncated if it exists.\n"
    "    grid: Grid to write."},

  {"read_grid", (PyCFunction)pysif_read_grid, METH_VARARGS | METH_KEYWORDS,
    "read_grid(filepath)\n"
    "--\n\n"
    "Read an .xgrid binary into a Grid.\n\n"
    "Args:\n"
    "    filepath: Input path.\n\n"
    "Returns:\n"
    "    Grid: The loaded grid, tagged with what its cells hold.\n\n"
    "Raises:\n"
    "    IOError: If the file is missing, malformed, written at a different\n"
    "        precision, or fails its checksum."},

  {"read_field_ascii", (PyCFunction)pysif_read_field_ascii,
    METH_VARARGS | METH_KEYWORDS,
    "read_field_ascii(filepath, format='x y z', delimiter=' ', skip_lines=0)\n"
    "--\n\n"
    "Read a particle field from an ASCII table.\n\n"
    "Far slower than the binary path, and the portable one.\n\n"
    "Args:\n"
    "    filepath: Input path.\n"
    "    format: The columns, in order: 'x', 'y', 'z' for positions, or\n"
    "        'ra', 'dec', 'z' for sky coordinates (a Field with units='sky');\n"
    "        'vx', 'vy', 'vz' for velocities, 'w' for the per-particle\n"
    "        weight, '*' to skip a column. Case does not matter, and spaces\n"
    "        or commas between names are optional: 'x y z * w', 'x,y,z,*,w'\n"
    "        and 'xyz*w' are the same. Anything else, a name given twice, or "
    "a\n"
    "        position or velocity named in part, is a ValueError.\n"
    "    delimiter: Column separator; ' ' for whitespace.\n"
    "    skip_lines: Header lines to skip.\n\n"
    "Returns:\n"
    "    Field: The loaded field.\n\n"
    "Note:\n"
    "    A column that does not parse as a number reads as 0.0 rather than\n"
    "    raising, so check that format matches the file."},

  {"read_field_binary", (PyCFunction)pysif_read_field_binary,
    METH_VARARGS | METH_KEYWORDS,
    "read_field_binary(filepath, format='x y z', *, layout='rows', "
    "precision='float32', byteorder='native', header_bytes=0, n_particles=0, "
    "field=None)\n"
    "--\n\n"
    "Read a raw binary file -- a header of known length, then the values.\n\n"
    "For files no other reader knows: the header is skipped without being\n"
    "read, and the arguments say everything about the data.\n\n"
    "Args:\n"
    "    filepath: Input path.\n"
    "    format: The columns, as for read_field_ascii(). In a binary file a\n"
    "        skipped column may carry its width in bytes, '*8', for a column\n"
    "        of another type -- a 64-bit ID among float32 values, say. So\n"
    "        'x y z *8 w'. '* * * vx vy vz' reads only the velocities.\n"
    "    layout: 'rows', one record per particle (x0 y0 z0 x1 y1 z1 ...,\n"
    "        what an array of structs gives), or 'blocks', one column at a\n"
    "        time (x0 x1 ... y0 y1 ..., what one array after another gives).\n"
    "    precision: 'float32' or 'float64', for every value column.\n"
    "    byteorder: 'native', 'little' or 'big'.\n"
    "    header_bytes: Bytes to skip before the data.\n"
    "    n_particles: Particles to read; 0 takes the count from the file,\n"
    "        which then has to hold a whole number of them.\n"
    "    field: A Field to read into instead of a new one: blocks it already\n"
    "        has are kept, so a second file can add columns to the first.\n"
    "        Its particle count is used, and it must not have been\n"
    "        Morton-sorted unless the format reloads the positions.\n\n"
    "Returns:\n"
    "    Field: The loaded field -- the one passed as field, if any.\n\n"
    "Raises:\n"
    "    ValueError: For a malformed format or an unknown option.\n"
    "    OSError: For a file that is missing, too short, or does not divide\n"
    "        into whole particles -- a sign the format, precision or header\n"
    "        length is not the file's."},

  {"read_hdf5", (PyCFunction)pysif_read_hdf5, METH_VARARGS | METH_KEYWORDS,
    "read_hdf5(paths, *, x=None, y=None, z=None, ra=None, dec=None, "
    "vx=None, vy=None, vz=None, w=None, length_scale=1.0, fraction=1.0, "
    "seed=0)\n"
    "--\n\n"
    "Read particles out of any HDF5 file -- a simulation snapshot, a halo\n"
    "catalogue, a mock -- into a Field.\n\n"
    "Each part of the field is a dataset, named by its path; a dataset of\n"
    "two dimensions, a row per particle, is read a column at a time, counted\n"
    "from 0 as numpy counts. The dark matter of an IllustrisTNG snapshot, in\n"
    "Mpc/h:\n\n"
    "    field = pysif.io.read_hdf5(\n"
    "        [\"snap_099.0.hdf5\", \"snap_099.1.hdf5\"],\n"
    "        x=\"PartType1/Coordinates[0]\", y=\"PartType1/Coordinates[1]\",\n"
    "        z=\"PartType1/Coordinates[2]\", length_scale=1e-3)\n\n"
    "Several files read as one, their rows one after another. A GADGET\n"
    "snapshot has read_gadget(), which finds its files and units by itself.\n\n"
    "Args:\n"
    "    paths: A path, or a list of them.\n"
    "    x, y, z: Positions; or ra, dec, z for sky coordinates (a Field\n"
    "        with units='sky').\n"
    "    vx, vy, vz: Velocities: all three, or none.\n"
    "    w: Per-particle weight, or None.\n"
    "    length_scale: What Cartesian positions are multiplied by: 1e-3 from\n"
    "        kpc/h to Mpc/h. Velocities and weights are read as they are.\n"
    "    fraction: Share of the rows to keep, in (0, 1], over all the files\n"
    "        together.\n"
    "    seed: Seed for the subsample.\n\n"
    "Returns:\n"
    "    Field: The loaded field.\n\n"
    "Raises:\n"
    "    ValueError: For a dataset a file does not have or cannot give, row\n"
    "        counts that disagree, or a value that is not finite.\n"
    "    OSError: For a file that is missing or not HDF5.\n"
    "    RuntimeError: If pysif was built without HDF5."},

  {"read_gadget", (PyCFunction)pysif_read_gadget, METH_VARARGS | METH_KEYWORDS,
    "read_gadget(path, *, ptype=1, velocities=None, masses=False, "
    "length='auto', fraction=1.0, seed=0, format='auto')\n"
    "--\n\n"
    "Read one particle type of a GADGET snapshot into a Field.\n\n"
    "Reads all three snapshot formats: the legacy binaries (SnapFormat 1\n"
    "and 2, with either the GADGET-2 or the GADGET-4 header, in either byte\n"
    "order and precision) and HDF5. A snapshot split over several files is\n"
    "read as one: name any of its files, or the base name, and the rest are\n"
    "found -- snap_010.N, snap_010.N.hdf5, and both inside snapdir_010/.\n\n"
    "Args:\n"
    "    path: A file of the snapshot, or its base name.\n"
    "    ptype: Particle type, 0 to 5. One per call; 1 is GADGET's dark\n"
    "        matter.\n"
    "    velocities: None to skip them; 'raw' for u as GADGET stores it;\n"
    "        'peculiar' for v = u * sqrt(a), a being the header's Time --\n"
    "        which is the scale factor only in a cosmological run.\n"
    "    masses: Read per-particle masses into the weights. A type with a\n"
    "        mass in the header's table has one mass for every particle, and\n"
    "        is left unweighted.\n"
    "    length: The snapshot's length unit: 'auto', the default, reads\n"
    "        UnitLength_in_cm from an HDF5 file; a binary file records no\n"
    "        unit, so it has to be named -- 'kpc' (kpc/h, GADGET's default)\n"
    "        or 'mpc' (Mpc/h). Positions and box come out in Mpc/h.\n"
    "    fraction: Share of the particles to keep, in (0, 1]. Exactly\n"
    "        round(fraction * N) are kept, uniformly at random, in file\n"
    "        order.\n"
    "    seed: Seed for the subsample: the same seed and files give the\n"
    "        same particles.\n"
    "    format: 'auto', 1, 2 or 'hdf5'. Anything but 'auto' is checked\n"
    "        against the file.\n\n"
    "Returns:\n"
    "    tuple: (Field, box_length), the box in Mpc/h. Returned alongside\n"
    "    the field because it is converted with it, and the header's\n"
    "    box_size is in the file's own unit.\n\n"
    "Raises:\n"
    "    ValueError: For an argument out of range, a type with no\n"
    "        particles, a fraction that keeps none, or length='auto' (the\n"
    "        default) on a file that records no unit -- every binary one.\n"
    "    OSError: For a missing, truncated or inconsistent snapshot.\n"
    "    RuntimeError: For an HDF5 snapshot, if pysif was built without\n"
    "        HDF5."},

  {"gadget_header", (PyCFunction)pysif_gadget_header,
    METH_VARARGS | METH_KEYWORDS,
    "gadget_header(path, format='auto')\n"
    "--\n\n"
    "Read the header of one GADGET snapshot file.\n\n"
    "Only the first file of a multi-file snapshot is read, so n_part_file\n"
    "is that file's and n_part_total the snapshot's.\n\n"
    "Returns:\n"
    "    dict: format (1, 2 or 'hdf5'), swapped and legacy_header (None for\n"
    "    HDF5), precision (bytes per position component), n_types,\n"
    "    n_part_file, n_part_total, mass_table, time, redshift, box_size\n"
    "    (in the file's unit), n_files, cosmology (a dict, or None if the\n"
    "    file has none), unit_length_in_cm (None if not recorded),\n"
    "    n_blocks, blocks (names, or None for SnapFormat 1)."},

  {"inspect_gadget", (PyCFunction)pysif_inspect_gadget,
    METH_VARARGS | METH_KEYWORDS,
    "inspect_gadget(path, format='auto')\n"
    "--\n\n"
    "Print a GADGET snapshot's header: format, precision, particle counts\n"
    "per type, masses, time, box, cosmology and blocks. For a look at a\n"
    "snapshot before deciding how to read it; gadget_header() returns the\n"
    "same as a dict."},

  {"read_fits", (PyCFunction)pysif_read_fits, METH_VARARGS | METH_KEYWORDS,
    "read_fits(paths, *, x=None, y=None, z=None, ra=None, dec=None, "
    "vx=None, vy=None, vz=None, w=None, where=None, hdu=None, fraction=1.0, "
    "seed=0)\n"
    "--\n\n"
    "Read a catalogue of one or more FITS tables into a Field.\n\n"
    "Each part of the field is named by a column of the table or by an\n"
    "arithmetic expression over columns, in cfitsio's syntax -- how a\n"
    "weight built from several columns, or an element of a vector column,\n"
    "is read:\n\n"
    "    field = pysif.io.read_fits(\n"
    "        [\"galaxies_NGC.fits\", \"galaxies_SGC.fits\"],\n"
    "        ra=\"RA\", dec=\"DEC\", z=\"Z\",\n"
    "        w=\"WEIGHT_SYSTOT * (WEIGHT_NOZ + WEIGHT_CP - 1)\",\n"
    "        where=\"Z > 0.43 && Z < 0.7\")\n\n"
    "Several files read as one table, their rows one after another. Rows\n"
    "the filter drops are never read, and the subsample is drawn from the\n"
    "rows it keeps, over all the files together -- so data and randoms read\n"
    "with the same filter are cut identically, whatever share of each is\n"
    "kept.\n\n"
    "Args:\n"
    "    paths: A path, or a list of them. Compressed files (.fits.gz) are\n"
    "        read, uncompressed in memory.\n"
    "    x, y, z: Positions, read as they are.\n"
    "    ra, dec: Sky coordinates in place of x and y -- right ascension and\n"
    "        declination in degrees, with z the redshift. The Field then has\n"
    "        units='sky', to be turned into positions with\n"
    "        Field.convert_sky_coordinates().\n"
    "    vx, vy, vz: Velocities: all three, or none.\n"
    "    w: Per-row weight, or None for an unweighted field.\n"

    "    where: Row filter, a boolean expression: 'Z > 0.43 && Z < 0.7',\n"
    "        '!ISNULL(Z)'.\n"
    "    hdu: The table in each file: its EXTNAME, or its extension number\n"
    "        (1 for the first). None reads the first table.\n"
    "    fraction: Share of the filtered rows to keep, in (0, 1]. Exactly\n"
    "        round(fraction * N) are kept, uniformly at random, in file\n"
    "        order.\n"
    "    seed: Seed for the subsample: the same seed and files give the\n"
    "        same rows.\n\n"
    "Returns:\n"
    "    Field: The loaded field.\n\n"
    "Raises:\n"
    "    ValueError: For a request the files cannot satisfy: a column, HDU\n"
    "        or expression they do not have, a filter that keeps nothing,\n"
    "        an undefined value (NaN, TNULL) in a row the filter keeps.\n"
    "    OSError: For a file that is missing or is not readable as FITS.\n"
    "    RuntimeError: If pysif was built without FITS support."},

  {"inspect_fits", (PyCFunction)pysif_inspect_fits,
    METH_VARARGS | METH_KEYWORDS,
    "inspect_fits(path)\n"
    "--\n\n"
    "Print what a FITS file holds: every HDU, and for each table its row\n"
    "count and its columns, with their type, their unit, and a note for a\n"
    "vector column (read as NAME[1] ... NAME[n]) or one read_fits() cannot\n"
    "use. For finding the names to pass to read_fits()."},

  {"write_catalog_fits", (PyCFunction)pysif_write_catalog_fits,
    METH_VARARGS | METH_KEYWORDS,
    "write_catalog_fits(filepath, catalog)\n"
    "--\n\n"
    "Write a catalogue as a FITS table: an empty primary HDU, then a binary\n"
    "table named VOIDS, one row per void.\n\n"
    "The columns follow Catalog.units: CX, CY, CZ, R for Cartesian centres;\n"
    "RA, DEC (degrees), Z (redshift), R (comoving, Mpc/h) for a\n"
    "catalogue on the sky -- see Catalog.to_sky(). FOOTPRINT and\n"
    "FOOTPRINT_SHELL follow when the catalogue has them. The primary header\n"
    "is left for set_fits_key(), to record what the catalogue came from.\n\n"
    "Args:\n"
    "    filepath: Output path, replaced if it exists.\n"
    "    catalog: Catalogue to write."},

  {"read_catalog_fits", (PyCFunction)pysif_read_catalog_fits,
    METH_VARARGS | METH_KEYWORDS,
    "read_catalog_fits(filepath)\n"
    "--\n\n"
    "Read a catalogue written by write_catalog_fits(): on the sky if its\n"
    "table has RA, DEC and Z columns, Cartesian if CX, CY and CZ (or X, Y\n"
    "and Z). R or RADIUS is the radius.\n\n"
    "Returns:\n"
    "    Catalog: The loaded catalogue.\n\n"
    "Raises:\n"
    "    FileNotFoundError: For a missing file.\n"
    "    ValueError: For a file with no VOIDS table of those columns."},

  {"write_profiles_fits", (PyCFunction)pysif_write_profiles_fits,
    METH_VARARGS | METH_KEYWORDS,
    "write_profiles_fits(filepath, profiles)\n"
    "--\n\n"
    "Write stacked profiles into DENSITY_PROFILES and VELOCITY_PROFILES:\n"
    "a row per void, the profile as one vector column, the shape and the\n"
    "bin edges (EDGE0, EDGE1, ...) as keywords. Only the sets the Profiles\n"
    "carries are written; the other is left in the file as it was.\n\n"
    "A FITS file holds a catalogue and its products one table each, as the\n"
    "HDF5 file does a group each: every writer creates the file if it is\n"
    "missing and replaces only its own table, and a FITS file sif did not\n"
    "write is never written into.\n\n"
    "Args:\n"
    "    filepath: The file.\n"
    "    profiles: Profiles to write."},

  {"read_profiles_fits", (PyCFunction)pysif_read_profiles_fits,
    METH_VARARGS | METH_KEYWORDS,
    "read_profiles_fits(filepath)\n"
    "--\n\n"
    "Read the profiles write_profiles_fits() wrote: whichever sets the file\n"
    "holds.\n\n"
    "Returns:\n"
    "    Profiles: The loaded profiles.\n\n"
    "Raises:\n"
    "    FileNotFoundError: For a missing file.\n"
    "    ValueError: For a file that holds no profiles."},

  {"write_size_function_fits", (PyCFunction)pysif_write_size_function_fits,
    METH_VARARGS | METH_KEYWORDS,
    "write_size_function_fits(filepath, size_function)\n"
    "--\n\n"
    "Write a size function into SIZE_FUNCTION: a row per bin -- R_LOW,\n"
    "R_HIGH, R_CENTER, COUNT, VSF, ERR -- with R_MIN, R_MAX and BINNING\n"
    "('ln' or 'linear', what VSF is per unit of) as keywords. As for\n"
    "write_profiles_fits(), only its own table is replaced.\n\n"
    "Args:\n"
    "    filepath: The file.\n"
    "    size_function: Size function to write, measured or modelled."},

  {"read_size_function_fits", (PyCFunction)pysif_read_size_function_fits,
    METH_VARARGS | METH_KEYWORDS,
    "read_size_function_fits(filepath)\n"
    "--\n\n"
    "Read the size function write_size_function_fits() wrote.\n\n"
    "Returns:\n"
    "    SizeFunction: The loaded size function.\n\n"
    "Raises:\n"
    "    FileNotFoundError: For a missing file.\n"
    "    ValueError: For a file that holds none."},

  {"fits_key", (PyCFunction)pysif_fits_key, METH_VARARGS | METH_KEYWORDS,
    "fits_key(path, key, hdu=None)\n"
    "--\n\n"
    "A keyword's value from a FITS header, or None if it is not there.\n\n"
    "The value comes back as the header has it: int, float, bool (a FITS\n"
    "logical) or str. Names are matched without regard to case; a name\n"
    "longer than eight characters is a HIERARCH keyword.\n\n"
    "Args:\n"
    "    path: The file.\n"
    "    key: The keyword's name.\n"
    "    hdu: None for the primary header, where a file keeps what describes\n"
    "        it as a whole; otherwise an extension number or an EXTNAME.\n"
    "        (For read_fits(), None is the first table instead.)\n\n"
    "Raises:\n"
    "    FileNotFoundError: For a missing file. An HDU the file does not have\n"
    "        reads as None, and is logged."},

  {"set_fits_key", (PyCFunction)pysif_set_fits_key,
    METH_VARARGS | METH_KEYWORDS,
    "set_fits_key(path, key, value, hdu=None)\n"
    "--\n\n"
    "Write a keyword into a FITS header, replacing one of the same name.\n\n"
    "Args:\n"
    "    path: The file, which has to exist.\n"
    "    key: The keyword's name; longer than eight characters, it is\n"
    "        written as a HIERARCH keyword.\n"
    "    value: An int, a float or a str. A bool is written as 1 or 0; a\n"
    "        string longer than a header card holds goes over several.\n"
    "    hdu: As for fits_key(): None for the primary header.\n\n"
    "Raises:\n"
    "    ValueError: For an HDU the file does not have, a name FITS does not\n"
    "        allow, or a value that is not finite.\n"
    "    OSError: If the file could not be written."},

  {"write_catalog_ascii", (PyCFunction)pysif_write_catalog_ascii,
    METH_VARARGS | METH_KEYWORDS,
    "write_catalog_ascii(filepath, catalog)\n"
    "--\n\n"
    "Write a catalogue as text: a two-line header behind '#', then a row\n"
    "per void.\n\n"
    "    #n=1024\n"
    "    #cx cy cz r footprint footprint_shell\n"
    "    101.25 250.5 33.125 12.5 1 0.84\n\n"
    "The first line is the count, the second names the columns: 'ra dec z\n"
    "r' for a catalogue on the sky (Catalog.to_sky()), and the footprint\n"
    "columns only when the catalogue carries one, as one from\n"
    "finders.exodus_survey() does. numpy.loadtxt(filepath) reads the rows\n"
    "as they are.\n\n"
    "Written with enough significant digits to recover the stored values\n"
    "exactly, so a write/read round trip is lossless.\n\n"
    "Args:\n"
    "    filepath: Output path, truncated if it exists.\n"
    "    catalog: Catalogue to write."},

  {"read_catalog_ascii", (PyCFunction)pysif_read_catalog_ascii,
    METH_VARARGS | METH_KEYWORDS,
    "read_catalog_ascii(filepath, format=None)\n"
    "--\n\n"
    "Read a catalogue written by write_catalog_ascii().\n\n"
    "Lines starting with '#' are comments. Before the first row, 'n=N'\n"
    "gives the count -- without it the rows are counted first -- and a\n"
    "comment naming the columns places them, in any order: 'cx cy cz' (or\n"
    "'x y z') or 'ra dec z' for the centres, 'r' (or 'radius'), and\n"
    "optionally 'footprint footprint_shell'; columns with other names are\n"
    "skipped. Without names, rows are 'cx cy cz r', and six columns add the\n"
    "footprint. Files written before the header, whose first line is the\n"
    "count alone, read as they always did.\n\n"
    "A catalogue another finder wrote is read with a format, which places\n"
    "the columns in the language of the field readers: 'x y z' (or\n"
    "'cx cy cz') or 'ra dec z' for the centre, 'r' for the radius, '*' for\n"
    "a column not read -- '* x y z * r' for rows of an ID, the centre, a\n"
    "volume and the radius. Names in the file are then ignored. Columns may\n"
    "be separated by blanks or commas, and a skipped column, or one past\n"
    "the last named, may hold anything.\n\n"
    "Args:\n"
    "    filepath: Input path.\n"
    "    format: The columns, or None to take them from the file.\n\n"
    "Returns:\n"
    "    Catalog: The loaded catalogue."},

  {"write_profiles_ascii", (PyCFunction)pysif_write_profiles_ascii,
    METH_VARARGS | METH_KEYWORDS,
    "write_profiles_ascii(filepath, profiles, catalog)\n"
    "--\n\n"
    "Write stacked profiles as text, one row per void.\n\n"
    "A header behind '#', then a row per void -- the void it belongs to,\n"
    "then its profile:\n\n"
    "    #n=1024\n"
    "    #n_bins=20\n"
    "    #ext=3\n"
    "    #differential=1\n"
    "    #r_edges=0 0.15 0.3 ... 3\n"
    "    #cx cy cz r density_0 ... density_19 v_rad_0 ... v_rad_19\n"
    "    101.25 250.5 33.125 12.5 -0.91 ... 12.3\n\n"
    "The rows are plain numbers, so numpy.loadtxt(path) reads them as they\n"
    "are, and the last header line names every column: the centres are\n"
    "'ra dec z' for a catalogue on the sky (Catalog.to_sky()), and only the\n"
    "profile blocks the Profiles carries are there. Bin edges are in units\n"
    "of each void's own radius; multiply by the radius in the row for\n"
    "physical units.\n\n"
    "Written with enough significant digits to recover the stored values\n"
    "exactly, so a write/read round trip is lossless.\n\n"
    "Args:\n"
    "    filepath: Output path, truncated if it exists.\n"
    "    profiles: Profiles to write.\n"
    "    catalog: The catalogue they were measured from, which is where the\n"
    "        per-void columns come from. Row i is void i, so it has to be\n"
    "        that catalogue and not another of the same length."},

  {"read_profiles_ascii", (PyCFunction)pysif_read_profiles_ascii,
    METH_VARARGS | METH_KEYWORDS,
    "read_profiles_ascii(filepath)\n"
    "--\n\n"
    "Read profiles written by write_profiles_ascii().\n\n"
    "Args:\n"
    "    filepath: Input path.\n\n"
    "Returns:\n"
    "    tuple: (Profiles, Catalog). The Profiles carries whichever blocks\n"
    "    the file holds; has_density and has_velocity say which."},

  {"write_catalog_hdf5", (PyCFunction)pysif_write_catalog_hdf5,
    METH_VARARGS | METH_KEYWORDS,
    "write_catalog_hdf5(filepath, catalog)\n"
    "--\n\n"
    "Write a catalogue into /catalog of an HDF5 file.\n\n"
    "The file can hold any subset of catalog, density_profiles,\n"
    "velocity_profiles and size_function, each in its own group; this\n"
    "replaces only its own and creates the file if it is missing. It reads\n"
    "back with h5py without pysif. An existing file sif did not write is\n"
    "never overwritten.\n\n"
    "In a pysif built without HDF5 the data is saved as plain text in\n"
    "<filepath>.<product>.txt instead, with a RuntimeWarning saying so.\n\n"
    "Args:\n"
    "    filepath: The file.\n"
    "    catalog: Catalogue to write, with its footprint if it has one.\n\n"
    "Raises:\n"
    "    OSError: If the file could not be written, or is not sif's."},

  {"read_catalog_hdf5", (PyCFunction)pysif_read_catalog_hdf5,
    METH_VARARGS | METH_KEYWORDS,
    "read_catalog_hdf5(filepath)\n"
    "--\n\n"
    "Read /catalog of an HDF5 file, footprint included when present.\n\n"
    "Returns:\n"
    "    Catalog: The catalogue.\n\n"
    "Raises:\n"
    "    OSError: If the file or the group is missing or malformed.\n"
    "    RuntimeError: If pysif was built without HDF5."},

  {"write_profiles_hdf5", (PyCFunction)pysif_write_profiles_hdf5,
    METH_VARARGS | METH_KEYWORDS,
    "write_profiles_hdf5(filepath, profiles)\n"
    "--\n\n"
    "Write stacked profiles into /density_profiles and /velocity_profiles.\n\n"
    "Only the sets the Profiles holds are written; one it does not hold\n"
    "stays in the file as it was. A row count that disagrees with the\n"
    "catalogue in the file is warned about in the log, not refused.\n\n"
    "The file can hold any subset of catalog, density_profiles,\n"
    "velocity_profiles and size_function, each in its own group; this\n"
    "replaces only its own and creates the file if it is missing. It reads\n"
    "back with h5py without pysif. An existing file sif did not write is\n"
    "never overwritten.\n\n"
    "In a pysif built without HDF5 the data is saved as plain text in\n"
    "<filepath>.<product>.txt instead, with a RuntimeWarning saying so.\n\n"
    "Args:\n"
    "    filepath: The file.\n"
    "    profiles: Profiles to write."},

  {"read_profiles_hdf5", (PyCFunction)pysif_read_profiles_hdf5,
    METH_VARARGS | METH_KEYWORDS,
    "read_profiles_hdf5(filepath)\n"
    "--\n\n"
    "Read whichever profile sets an HDF5 file holds.\n\n"
    "Returns:\n"
    "    Profiles: has_density and has_velocity say which came back.\n\n"
    "Raises:\n"
    "    ValueError: If the file holds no profiles.\n"
    "    OSError: If the file is missing or malformed.\n"
    "    RuntimeError: If pysif was built without HDF5."},

  {"write_size_function_hdf5", (PyCFunction)pysif_write_size_function_hdf5,
    METH_VARARGS | METH_KEYWORDS,
    "write_size_function_hdf5(filepath, size_function)\n"
    "--\n\n"
    "Write a size function, measured or modelled, into /size_function.\n\n"
    "The binning goes in as the attribute 'binning' ('ln' or 'linear'),\n"
    "since it decides whether vsf is per unit ln R or per unit R.\n\n"
    "The file can hold any subset of catalog, density_profiles,\n"
    "velocity_profiles and size_function, each in its own group; this\n"
    "replaces only its own and creates the file if it is missing. It reads\n"
    "back with h5py without pysif. An existing file sif did not write is\n"
    "never overwritten.\n\n"
    "In a pysif built without HDF5 the data is saved as plain text in\n"
    "<filepath>.<product>.txt instead, with a RuntimeWarning saying so.\n\n"
    "Args:\n"
    "    filepath: The file.\n"
    "    size_function: SizeFunction to write."},

  {"read_size_function_hdf5", (PyCFunction)pysif_read_size_function_hdf5,
    METH_VARARGS | METH_KEYWORDS,
    "read_size_function_hdf5(filepath)\n"
    "--\n\n"
    "Read /size_function of an HDF5 file.\n\n"
    "Returns:\n"
    "    SizeFunction: The size function.\n\n"
    "Raises:\n"
    "    OSError: If the file or the group is missing or malformed.\n"
    "    RuntimeError: If pysif was built without HDF5."},

  {"set_hdf5_attr", (PyCFunction)pysif_set_hdf5_attr,
    METH_VARARGS | METH_KEYWORDS,
    "set_hdf5_attr(filepath, key, value, group=None)\n"
    "--\n\n"
    "Attach a named value to an HDF5 file, or to one of its products.\n\n"
    "The file's own header -- how the catalogue was made, which simulation,\n"
    "which snapshot -- the way FITS keywords are. On the root (group None)\n"
    "an entry describes the file and survives every rewrite; on a product\n"
    "('catalog', 'size_function', ...) it describes that product and goes\n"
    "when the product is rewritten.\n\n"
    "sif's own attributes (n_voids, n_bins, ... and every root name\n"
    "beginning ``sif_``) cannot be set. Without HDF5 the entry is appended to\n"
    "<filepath>.attributes.txt instead, with a RuntimeWarning.\n\n"
    "Args:\n"
    "    filepath: The file; for the root it is created if missing.\n"
    "    key: The entry's name.\n"
    "    value: An int (or bool), float or str.\n"
    "    group: None for the file itself, or a product that is already\n"
    "        in it.\n\n"
    "Raises:\n"
    "    ValueError: For a missing product or a name sif keeps.\n"
    "    TypeError: For a value that is not int, float or str."},

  {"get_hdf5_attr", (PyCFunction)pysif_get_hdf5_attr,
    METH_VARARGS | METH_KEYWORDS,
    "get_hdf5_attr(filepath, key, group=None)\n"
    "--\n\n"
    "Read one named value from an HDF5 file or one of its products.\n\n"
    "Returns:\n"
    "    int, float or str.\n\n"
    "Raises:\n"
    "    OSError: If the file, the group or the entry is missing.\n"
    "    TypeError: If the entry is an array or something else these\n"
    "        functions do not read.\n"
    "    RuntimeError: If pysif was built without HDF5."},

  {"get_hdf5_attrs", (PyCFunction)pysif_get_hdf5_attrs,
    METH_VARARGS | METH_KEYWORDS,
    "get_hdf5_attrs(filepath, group=None)\n"
    "--\n\n"
    "Every named value of an HDF5 file, or of one of its products.\n\n"
    "sif's own entries are included. One of a kind these functions do not\n"
    "read (an array another tool put there) is listed as None.\n\n"
    "Returns:\n"
    "    dict: name -> int, float, str or None.\n\n"
    "Raises:\n"
    "    OSError: If the file or the group is missing.\n"
    "    RuntimeError: If pysif was built without HDF5."},

  {NULL, NULL, 0, NULL}};

static struct PyModuleDef io_module = {PyModuleDef_HEAD_INIT,
  .m_name = "pysif.io",
  .m_doc =
    "Reading and writing sif's on-disk formats.\n\n"
    "The .xfield and .xgrid binary formats load without parsing or\n"
    "copying, at the cost of being portable only between machines\n"
    "that agree on endianness and on the precision sif was built\n"
    "with; both record a checksum and refuse a file that fails it.\n"
    "ASCII is the portable path, and far slower; read_field_binary()\n"
    "reads raw binary files written by anything else, given their layout.\n\n"
    "read_gadget() loads GADGET snapshots, in all three formats.\n\n"
    "The ``*_hdf5`` functions keep a catalogue and what was measured from\n"
    "it in one HDF5 file, readable with h5py without pysif.",
  .m_size = -1, .m_methods = io_methods};

/* Submodule exporter called from the parent module initialization routing */
PyObject* py_sif_init_io(void) { return PyModule_Create(&io_module); }
