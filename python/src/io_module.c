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
    "read_field_ascii(filepath, format, delimiter=' ', skip_lines=0)\n"
    "--\n\n"
    "Read a particle field from an ASCII table.\n\n"
    "Far slower than the binary path, and the portable one.\n\n"
    "Args:\n"
    "    filepath: Input path.\n"
    "    format: One character per column: 'x', 'y', 'z' for positions,\n"
    "        'u', 'v', 'w' for velocities, 'm' for the per-particle weight, "
    "'*' or '/' to skip\n"
    "        a column. So 'xyz*m' reads position, skips one, then the weight.\n"
    "    delimiter: Column separator; ' ' for whitespace.\n"
    "    skip_lines: Header lines to skip.\n\n"
    "Returns:\n"
    "    Field: The loaded field.\n\n"
    "Note:\n"
    "    A column that does not parse as a number reads as 0.0 rather than\n"
    "    raising, so check that format matches the file."},

  {"write_catalog_ascii", (PyCFunction)pysif_write_catalog_ascii,
    METH_VARARGS | METH_KEYWORDS,
    "write_catalog_ascii(filepath, catalog)\n"
    "--\n\n"
    "Write a catalogue as text: a count line, then 'cx cy cz radius' per\n"
    "void -- and 'footprint footprint_shell' after them when the catalogue\n"
    "carries a footprint, as one from finders.exodus_survey() does.\n\n"
    "Written with enough significant digits to recover the stored values\n"
    "exactly, so a write/read round trip is lossless.\n\n"
    "Args:\n"
    "    filepath: Output path, truncated if it exists.\n"
    "    catalog: Catalogue to write."},

  {"read_catalog_ascii", (PyCFunction)pysif_read_catalog_ascii,
    METH_VARARGS | METH_KEYWORDS,
    "read_catalog_ascii(filepath)\n"
    "--\n\n"
    "Read a catalogue written by write_catalog_ascii().\n\n"
    "The leading count sizes the allocation, so the file must carry it.\n"
    "The footprint columns come back when the file has them, and files\n"
    "written without them read as they always did.\n\n"
    "Args:\n"
    "    filepath: Input path.\n\n"
    "Returns:\n"
    "    Catalog: The loaded catalogue."},

  {"write_profiles_ascii", (PyCFunction)pysif_write_profiles_ascii,
    METH_VARARGS | METH_KEYWORDS,
    "write_profiles_ascii(filepath, profiles, catalog)\n"
    "--\n\n"
    "Write stacked profiles as text, one row per void.\n\n"
    "A row is the void it belongs to and then its profile, so it stands on\n"
    "its own:\n\n"
    "    n_voids n_bins ext has_density has_velocity\n"
    "    r_edge[0] ... r_edge[n_bins]\n"
    "    cx cy cz radius  density[...]  v_rad[...]\n\n"
    "Only the two leading lines are ragged, so the table proper loads with\n"
    "numpy.loadtxt(path, skiprows=2). Bin edges are in units of each void's\n"
    "own radius; multiply by the radius in the row for physical units.\n"
    "Whichever of the two profile blocks the Profiles carries is written.\n\n"
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
    "beginning 'sif_') cannot be set. Without HDF5 the entry is appended to\n"
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
  .m_doc = "Reading and writing sif's on-disk formats.\n\n"
           "The .xfield and .xgrid binary formats load without parsing or\n"
           "copying, at the cost of being portable only between machines\n"
           "that agree on endianness and on the precision sif was built\n"
           "with; both record a checksum and refuse a file that fails it.\n"
           "ASCII is the portable path, and far slower.\n\n"
           "The *_hdf5 functions keep a catalogue and what was measured from\n"
           "it in one HDF5 file, readable with h5py without pysif.",
  .m_size = -1, .m_methods = io_methods};

/* Submodule exporter called from the parent module initialization routing */
PyObject* py_sif_init_io(void) { return PyModule_Create(&io_module); }
