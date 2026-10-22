# Fields

How to create a field from any source

In **sif**, a set of particles is represented by a field object, which stores
positions and, optionally, velocities and weights. This page shows every way
to make one, and what has to be done to it before a grid or a chain mesh can
be built from it. 

## Contents of a field

A field holds:

- **positions**, always: three arrays `x`, `y` and `z`. They are either
  Cartesian positions or sky coordinates (see [](#sky-coordinates)), 
  specified by `units`.
- **velocities**, optionally: three arrays `vx`, `vy` and `vz`. Currently, 
  they are only used in the computation of the velocity profiles.
- **weights**, optionally: one array `w`. If not specified, all particles
  have the same weight.

The arrays are stored one after the other (Structure of Arrays), and the field
always owns them: every reader, and every function that builds a field from
arrays in memory, copies the data in.

## Reading from files

Each format has its own reader:

| Format | Python | C | sif-exodus `format` | Box length |
| --- | --- | --- | --- | --- |
| `.xfield` | `pysif.io.read_field` | `sif_field_read` | `"xfield"` | from the file |
| GADGET snapshot | `pysif.io.read_gadget` | `sif_field_read_gadget` | `"gadget"` | from the file |
| HDF5 (any) | `pysif.io.read_hdf5` | `sif_field_read_hdf5` | `"hdf5"` | to be given |
| FITS table | `pysif.io.read_fits` | `sif_field_read_fits` | `"fits"` | to be given |
| ASCII table | `pysif.io.read_field_ascii` | `sif_field_read_ascii` | `"ascii"` | to be given |
| Raw binary | `pysif.io.read_field_binary` | `sif_field_read_binary` | `"binary"` | to be given |

The box length matters only for a simulation box; a survey is placed in a box
of its own (see [](#preparing-a-survey)). In **sif-exodus**, `box_length` goes
in the `input` section, next to `path` and `format`, and overrides the one a
file records.

The C readers return `NULL` on failure, with the reason in the log; the Python
ones raise. Every field a C reader returns is released with
`sif_field_free()`. In the C snippets below, `field` and `box_length` are
declared as follows, and `goto done` jumps to the cleanup at the end of the
function:

```c
sif_field_t* field = NULL;
double box_length = 0.0;
```

### .xfield files

The `.xfield` format is **sif**'s own: a 64-byte header followed by the
arrays, exactly as the field holds them in memory. It is the fastest format to
read, useful when the same data is read many times. Its header records the box
length, which weights and velocities follow, and a checksum of the data.

:::::{tab-set}

::::{tab-item} Python
```python
import pysif

# the header alone, without reading the particles: the box length is
# recorded here, and not in the field
header = pysif.io.read_field_header("path/to/tracers.xfield")
box_length = header["box_length"]

# the particles, with whatever velocities and weights the file has.
# 'wrap=True' folds the positions into the box the header records, as
# Field.wrap() does (see below)
field = pysif.io.read_field("path/to/tracers.xfield")
```
::::

::::{tab-item} C
```c
#include "sif/io/field_io.h"

// the box length comes back through the last argument (NULL to skip it)
field = sif_field_read("path/to/tracers.xfield", &box_length);
if (!field)
  goto done;
```
::::

::::{tab-item} sif-exodus
```lua
input = {
  path = "path/to/tracers.xfield",
  -- the .xfield extension is enough to tell the format
}
```
::::

:::::

Any field can be written to an `.xfield` file

:::::{tab-set}

::::{tab-item} Python
```python
# weights and velocities are written only if given
pysif.io.write_field(
    "path/to/tracers.xfield", box_length,
    field.x, field.y, field.z, weights=field.weights
)
```
::::

::::{tab-item} C
```c
// weight and velocity blocks are written only if the field carries them
if (sif_field_write("path/to/tracers.xfield", field, box_length) != SIF_OK)
  goto done;
```
::::

:::::

:::{note}
An `.xfield` file can be read by library builds that have the same byte order
and same precision.
:::

### GADGET snapshots

Both Gadget binary formats and the HDF5 format are supported (for HDF5, the
library needs to be compiled with HDF5 support). A snapshot split over several
files is read as one by providing the name of one of its files, or its base 
name (`snap_010` for `snap_010.0`, `snap_010.1`, ..., also inside 
`snapdir_010/`).

The parser reads one particle type, and it requires the length units: `kpc` for
kpc/h or `mpc` for Mpc/h; with an HDF5 snapshot, use `auto` to read it from the
file. Positions and box length always come out in Mpc/h.

:::::{tab-set}

::::{tab-item} Python
```python
import pysif

# a look at the header first: particle counts per type, mass table,
# box (in the file's own unit), redshift, cosmology. gadget_header()
# returns the same as a dict
pysif.io.inspect_gadget("path/to/snapdir_010/snap_010")

# the dark matter (ptype=1) of a snapshot written in kpc/h.
# velocities: None skips them, 'raw' keeps them as GADGET stores them
# (u = v / sqrt(a)), 'peculiar' converts them to peculiar velocities.
# masses=True reads per-particle masses into the weights; a type whose
# mass is in the header's mass table is left unweighted.
# the box length is returned with the field, in Mpc/h
field, box_length = pysif.io.read_gadget(
    "path/to/snapdir_010/snap_010",
    ptype=1,
    velocities="peculiar",
    masses=False,
    length="kpc",
)
```
::::

::::{tab-item} C
```c
#include "sif/io/gadget_io.h"

// a look at the header first, printed to stdout
sif_gadget_inspect("path/to/snapdir_010/snap_010");

// every choice is an enumeration of its own, so a value in the wrong
// place is refused. the last three arguments are the subsampling
// fraction, its seed and where the box length (in Mpc/h) goes
field = sif_field_read_gadget("path/to/snapdir_010/snap_010",
  SIF_GADGET_FORMAT_AUTO, SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_PECULIAR,
  SIF_GADGET_MASS_SKIP, SIF_GADGET_LENGTH_KPC, 1.0, 0, &box_length);
if (!field)
  goto done;
```
::::

::::{tab-item} sif-exodus
```lua
input = {
  path = "path/to/snapdir_010/snap_010",
  format = "gadget",
  snapformat = "auto",  -- 1 | 2 | 3 (HDF5) | "auto"
  ptype = 1,
  length = "kpc",       -- kpc | mpc | auto (HDF5 files only)
  masses = false,       -- true: particle masses as weights
}
```

**sif-exodus** never reads the velocities.
::::

:::::

### HDF5 files

Any custom catalogue in HDF5 format can be parsed, and multiple files can be read
as one. The `length_scale` parameter multiplies the positions (not the 
velocities or the weights), to convert them to Mpc/h. The `box_length` needs to 
be manually specified.

:::::{tab-set}

::::{tab-item} Python
```python
import pysif

# the dark matter of a generic HDF5 snapshot, in two files, from
# kpc/h to Mpc/h
files = ["path/to/snap_099.0.hdf5", "path/to/snap_099.1.hdf5"]
field = pysif.io.read_hdf5(
    files,
    x="PartType1/Coordinates[0]",
    y="PartType1/Coordinates[1]",
    z="PartType1/Coordinates[2]",
    length_scale=1e-3,
)
box_length = 205.0  # Mpc/h, from the simulation's documentation

# a halo catalogue with one dataset per coordinate, weighted by mass
halos = pysif.io.read_hdf5(
    "path/to/halos.hdf5", x="pos_x", y="pos_y", z="pos_z", w="mass"
)
```
::::

::::{tab-item} C
```c
#include "sif/io/hdf5_io.h"

// which dataset fills which part of the field: anything left out of
// the initialiser is NULL, and not read
const char* files[] = {"path/to/snap_099.0.hdf5", "path/to/snap_099.1.hdf5"};
const sif_field_columns_t columns = {
  .x = "PartType1/Coordinates[0]",
  .y = "PartType1/Coordinates[1]",
  .z = "PartType1/Coordinates[2]",
};

// the arguments after the columns are the length scale, the subsampling
// fraction and its seed
field = sif_field_read_hdf5(files, 2, &columns, 1e-3, 1.0, 0);
box_length = 205.0; // Mpc/h
if (!field)
  goto done;
```
::::

::::{tab-item} sif-exodus
```lua
input = {
  path = { "path/to/snap_099.0.hdf5", "path/to/snap_099.1.hdf5" },
  format = "hdf5",
  columns = {
    x = "PartType1/Coordinates[0]",
    y = "PartType1/Coordinates[1]",
    z = "PartType1/Coordinates[2]",
  },
  length_scale = 1e-3,
  box_length = 205,
}
```
::::

:::::

:::{note}
Reading HDF5 files needs a build of **sif** with HDF5 support.
:::

### FITS tables

Each column can be given as a column name or as an arithmetic expression over
columns, in cfitsio syntax (e.g. `"WEIGHT_SYSTOT * WEIGHT_CP"`, or `"POS[1]"`
for the first element of a vector column). Passing `ra` and `dec` instead of
`x` and `y` reads sky coordinates, with `z` as the redshift (see
[](#sky-coordinates)). Multiple files are read as one table, `where` filters
the rows before they are read, and `hdu` selects the table (by default, the
first one).

:::::{tab-set}

::::{tab-item} Python
```python
import pysif

# tables and columns of the file
pysif.io.inspect_fits("path/to/galaxies_NGC.fits")

# sky coordinates, weights from two columns, redshift cut
galaxies = pysif.io.read_fits(
    ["path/to/galaxies_NGC.fits", "path/to/galaxies_SGC.fits"],
    ra="RA", dec="DEC", z="Z",
    w="WEIGHT_SYSTOT * WEIGHT_CP",
    where="Z > 0.43 && Z < 0.7",
)

# cartesian positions from a vector column, in the table named MOCK
mock = pysif.io.read_fits(
    "path/to/mock.fits",
    x="POS[1]", y="POS[2]", z="POS[3]",
    hdu="MOCK",
)
```
::::

::::{tab-item} C
```c
#include "sif/io/fits_io.h"

// tables and columns of the file
sif_fits_inspect("path/to/galaxies_NGC.fits");

const char* files[] = {
  "path/to/galaxies_NGC.fits", "path/to/galaxies_SGC.fits"};
const sif_field_columns_t columns = {
  .ra = "RA", .dec = "DEC", .z = "Z", .w = "WEIGHT_SYSTOT * WEIGHT_CP"};

// files, number of files, hdu (NULL for the first table), columns,
// filter, subsampling fraction and seed
field = sif_field_read_fits(
  files, 2, NULL, &columns, "Z > 0.43 && Z < 0.7", 1.0, 0);
if (!field)
  goto done;
```
::::

::::{tab-item} sif-exodus
```lua
-- survey (input.data or input.random), sky coordinates
data = {
  path = { "path/to/galaxies_NGC.fits", "path/to/galaxies_SGC.fits" },
  format = "fits",
  columns = { ra = "RA", dec = "DEC", z = "Z", w = "WEIGHT_SYSTOT * WEIGHT_CP" },
  where = "Z > 0.43 && Z < 0.7",
}

-- box, cartesian positions
input = {
  path = "path/to/mock.fits",
  columns = { x = "POS[1]", y = "POS[2]", z = "POS[3]" },
  hdu = "MOCK",
  box_length = 1000,
}
```
::::

:::::

:::{note}
Undefined values (NaN or FITS null) in the selected rows raise an error;
exclude them with `where` (e.g. `"!ISNULL(Z)"`). Reading FITS files needs a
build of **sif** with cfitsio.
:::

### ASCII tables

The columns are specified by a format string, naming them in order:

| Name | Column |
| --- | --- |
| `x` `y` `z` | positions |
| `ra` `dec` `z` | sky coordinates (degrees, redshift) |
| `vx` `vy` `vz` | velocities |
| `w` | weight |
| `*` | skipped |

Names are case-insensitive and separators are optional: `"x y z * w"`,
`"x,y,z,*,w"` and `"xyz*w"` are equivalent. Blank lines and lines starting
with `#` or `;` are skipped; a malformed row raises an error with its line
number.

:::::{tab-set}

::::{tab-item} Python
```python
import pysif

# comma-separated, one header line, ID column first
field = pysif.io.read_field_ascii(
    "path/to/halos.csv", format="* x y z w", delimiter=",", skip_lines=1
)
```
::::

::::{tab-item} C
```c
#include "sif/io/field_io.h"

// format, delimiter (' ' for any whitespace), header lines to skip
field = sif_field_read_ascii("path/to/halos.csv", "* x y z w", ',', 1);
if (!field)
  goto done;
```
::::

::::{tab-item} sif-exodus
```lua
input = {
  path = "path/to/halos.csv",
  format = "ascii",
  columns = "* x y z w",
  delimiter = ",",
  skip_header = 1,
  box_length = 1000,
}
```
::::

:::::

### Raw binary files

Raw binary files are read by giving their layout:

- the column format, as for ASCII tables; `*N` skips a column of `N` bytes
  (e.g. `*8` for a 64-bit ID);
- `layout`: `rows` (`x0 y0 z0 x1 y1 z1 ...`) or `blocks`
  (`x0 x1 ... y0 y1 ...`);
- `precision`: `float32` or `float64`;
- byte order: `native`, `little` or `big`;
- header length in bytes, skipped.

The number of particles is inferred from the file size, which must hold a
whole number of them.

:::::{tab-set}

::::{tab-item} Python
```python
import pysif

# 16-byte header, then records of a 64-bit ID and x, y, z, w in float64
field = pysif.io.read_field_binary(
    "path/to/halos.bin",
    format="*8 x y z w",
    layout="rows",
    precision="float64",
    byteorder="little",
    header_bytes=16,
)
```
::::

::::{tab-item} C
```c
#include "sif/io/field_io.h"

field = sif_field_read_binary("path/to/halos.bin", "*8 x y z w",
  SIF_BINARY_ROWS, SIF_BINARY_FLOAT64, SIF_BINARY_LITTLE, 16);
if (!field)
  goto done;
```
::::

::::{tab-item} sif-exodus
```lua
input = {
  path = "path/to/halos.bin",
  format = "binary",
  columns = "*8 x y z w",
  layout = "rows",
  precision = "float64",  -- required
  endian = "little",
  header_bytes = 16,
  box_length = 1000,
}
```
::::

:::::

#### Columns from several files

Binary and ASCII files can be read into an existing field, adding the columns
it does not have yet. The particles must be in the same order in every file.

:::::{tab-set}

::::{tab-item} Python
```python
field = pysif.io.read_field_binary(
    "path/to/positions.bin", format="x y z", layout="blocks"
)
pysif.io.read_field_binary("path/to/masses.bin", format="w", field=field)
```
::::

::::{tab-item} C
```c
field = sif_field_read_binary("path/to/positions.bin", "x y z",
  SIF_BINARY_BLOCKS, SIF_BINARY_FLOAT32, SIF_BINARY_NATIVE, 0);
if (!field ||
    sif_field_read_binary_into(field, "path/to/masses.bin", "w",
      SIF_BINARY_BLOCKS, SIF_BINARY_FLOAT32, SIF_BINARY_NATIVE, 0) != SIF_OK)
  goto done;

// sif_field_read_ascii_into() does the same for ASCII tables
```
::::

:::::

### Subsampling

The GADGET, HDF5 and FITS readers take a subsampling `fraction`, in (0, 1],
and a `seed`. Exactly `round(fraction * N)` particles are kept, in file order.
For FITS tables, the subsample is drawn after the `where` filter.

:::::{tab-set}

::::{tab-item} Python
```python
field, box_length = pysif.io.read_gadget(
    "path/to/snapshot", ptype=1, length="mpc", fraction=0.1, seed=42
)
```
::::

::::{tab-item} C
```c
field = sif_field_read_gadget("path/to/snapshot", SIF_GADGET_FORMAT_AUTO,
  SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP, SIF_GADGET_MASS_SKIP,
  SIF_GADGET_LENGTH_MPC, 0.1, 42, &box_length);
```
::::

::::{tab-item} sif-exodus
```lua
input = {
  path = "path/to/snapshot",
  format = "gadget",
  length = "mpc",
  fraction = 0.1,
  seed = 42,
}
```
::::

:::::

### Inspecting files

`inspect_gadget()` and `inspect_fits()` (`sif_gadget_inspect()` and
`sif_fits_inspect()` in C) print the content of a snapshot or a FITS file.
`sif-exodus --check` identifies any supported file, checks it without loading
it and prints a summary; `--deep` also reads every value and reports the range
of each column. For ASCII tables, `--columns`, `--delimiter` and
`--skip-header` check the file against a column format.

```bash
sif-exodus --check snapdir_010/snap_010
```

```bash
sif-exodus --check --deep --columns "* x y z w" --delimiter , --skip-header 1 halos.csv
```

## Building a field from arrays

In Python, `pysif.field_from_numpy()` copies numpy arrays into a new field. In
C, arrays are copied in with the `sif_field_assign_*()` functions, or written
directly into the field's arrays after reserving them.

:::::{tab-set}

::::{tab-item} Python
```python
import numpy as np
import pysif

# arrays of dtype pysif.real avoid a conversion copy
rng = np.random.default_rng(1)
n = 1_000_000
pos = rng.uniform(0, 500, size=(n, 3)).astype(pysif.real)
mass = rng.lognormal(size=n).astype(pysif.real)

# cartesian positions; vx, vy, vz and weights are optional
field = pysif.field_from_numpy(pos[:, 0], pos[:, 1], pos[:, 2], weights=mass)

# sky coordinates
survey = pysif.field_from_numpy(ra=ra, dec=dec, z=redshift)
```
::::

::::{tab-item} C
```c
#include "sif/structures/field.h"

field = sif_field_alloc(n);
if (!field)
  goto done;

// copy existing arrays
if (sif_field_assign_positions(field, x, y, z) != SIF_OK ||
    sif_field_assign_weights(field, w) != SIF_OK)
  goto done;

// or write directly into the field
if (sif_field_reserve_positions(field) != SIF_OK)
  goto done;
for (uint64_t i = 0; i < n; i++) {
  field->x[i] = xyz[3 * i];
  field->y[i] = xyz[3 * i + 1];
  field->z[i] = xyz[3 * i + 2];
}

// for sky coordinates
field->units = SIF_COORDINATES_SKY;
```
::::

:::::

**sif-exodus** only reads files: write the field to an `.xfield` file first
(see [](#xfield-files)).

## Accessing the data

In Python, `x`, `y`, `z`, `vx`, `vy`, `vz` and `weights` are read-only numpy
views of the field's arrays (`None` if absent). `to_numpy()` returns a copy of
the whole field as an `(N, k)` array.

```python
print(field.n_particles, field.units, field.has_weights, field.has_velocities)
table = field.to_numpy()  # x, y, z, [vx, vy, vz], [w]
```

While a view is alive, operations that move or release the arrays
(`ChainMesh(..., consume_field=True)`, `Field.sort_morton()`) raise
`BufferError`:

```python
x = field.x  # a view
mesh = pysif.ChainMesh(
    n_cells=n_cells_mesh, box_length=box_length, field=field,
    consume_field=True
)
# BufferError: cannot consume the field into a ChainMesh: the field has
# 1 live array view (field.x and the like) ...
```

Keep a copy instead, or delete the view first:

```python
x = np.array(field.x)  # a copy: survives the field being consumed
# or
del x

mesh = pysif.ChainMesh(
    n_cells=n_cells_mesh, box_length=box_length, field=field,
    consume_field=True
)
```

Views that are not stored, such as `field.x.max()`, are released right away
and never get in the way.

In C, the arrays are the members of `sif_field_t` (`NULL` if absent).

## Wrapping

Grids and chain meshes require positions in `[0, box_length)`. Positions
within rounding of the box edges are handled automatically; positions outside
the box are folded back with `wrap`, which returns how many coordinates were
on the edge, outside the box, or not finite.

:::::{tab-set}

::::{tab-item} Python
```python
counts = field.wrap(box_length=box_length)
# {'boundary': ..., 'wrapped': ..., 'nonfinite': ...}
```
::::

::::{tab-item} C
```c
// the counts are optional (NULL)
uint64_t n_boundary, n_wrapped, n_nonfinite;
if (sif_field_wrap_periodic(field, (sif_real)box_length, &n_boundary,
      &n_wrapped, &n_nonfinite) != SIF_OK)
  goto done;
```
::::

::::{tab-item} sif-exodus
In box mode, **sif-exodus** always wraps the field. `sif-exodus --check --deep`
reports the particles outside the box.
::::

:::::

:::{note}
Wrapping assumes a periodic box: do not apply it to surveys. A large number
of wrapped coordinates usually means a wrong box length or length unit.
:::

(sky-coordinates)=
## Sky coordinates

Fields read with `ra`, `dec` and `z` have `units` set to sky, and are refused
by grids, chain meshes, `wrap` and `translate` until they are converted to
comoving Cartesian coordinates. The conversion uses a w0waCDM cosmology, puts
the observer at the origin, and returns positions in Mpc/h. In Python the
cosmology is flat unless `omega_de` is given; in C every parameter is
explicit. All coordinates are validated before the conversion: if any is out
of range, the field is left unchanged.

:::::{tab-set}

::::{tab-item} Python
```python
cosmology = dict(omega_m=0.31, w0=-1.0, wa=0.0)
data.convert_sky_coordinates(**cosmology)
randoms.convert_sky_coordinates(**cosmology)

# mark a field as sky coordinates (no conversion)
field.units = "sky"
```
::::

::::{tab-item} C
```c
#include "sif/model/cosmology.h"

const sif_cosmology_t cosmology = {
  .omega_m = 0.31, .omega_de = 0.69, .omega_r = 0.0, .w0 = -1.0, .wa = 0.0};
if (sif_field_convert_sky_coordinates(data, &cosmology) != SIF_OK ||
    sif_field_convert_sky_coordinates(randoms, &cosmology) != SIF_OK)
  goto done;
```
::::

::::{tab-item} sif-exodus
```lua
input = {
  coordinates = "sky",             -- default; "cartesian" for x, y, z
  cosmology = { omega_m = 0.31 },  -- flat unless omega_de is given
  data = { --[[ ... ]] },
  random = { --[[ ... ]] },
}
```
::::

:::::

Surveys already in comoving Cartesian coordinates (observer at the origin) are
read with `x`, `y` and `z`, and need no conversion.

(preparing-a-survey)=
## Shifting surveys

Surveys have to be placed inside a box with enough padding around them.
`survey_box` computes the box length and the offset from the data, the
randoms and the search radii; the box allows for any grid whose cells are no
larger than the smallest radius. Data and randoms are shifted by the offset,
and the same offset goes to `exodus_survey`, which returns the voids in the
catalogues' own frame.

:::::{tab-set}

::::{tab-item} Python
```python
offset, box_length = pysif.finders.survey_box(
    data=data, randoms=randoms, radii=radii
)
data.translate(offset)
randoms.translate(offset)

# ... grids, chain meshes, then exodus_survey(..., offset, ...)
```
::::

::::{tab-item} C
```c
#include "sif/finder/exodus_finder.h"

sif_real offset[3], box;
if (sif_finder_exodus_survey_box(
      data, randoms, radii, n_radii, SIF_DEFAULT, offset, &box) != SIF_OK)
  goto done;
if (sif_field_translate(data, offset) != SIF_OK ||
    sif_field_translate(randoms, offset) != SIF_OK)
  goto done;

// ... grids, chain meshes, then sif_finder_exodus_survey(..., offset, ...)
```
::::

::::{tab-item} sif-exodus
Done automatically in survey mode.
::::

:::::

:::{note}
Use the same search factor for `survey_box` and the finder, and pass the data
as well as the randoms: the padding is checked around both.
:::

## Onto a grid

The finders and `pysif.measure` work on a density grid, made by depositing the
field with a mass-assignment scheme: each particle is spread over 1, 2, 3 or 4
cells per axis by nearest grid point (NGP), cloud-in-cell (CIC, the default),
triangular-shaped cloud (TSC) or piecewise cubic spline (PCS). A grid can only
hold the modes up to its Nyquist frequency; the field's power above it folds
back onto them -- aliasing -- and a higher order folds back less, at the cost
of a deposit touching more cells. Interlacing deposits the field a second time
shifted by half a cell and averages the two in Fourier space, which cancels
the leading aliases outright, for a second deposit and two FFTs.

The grid records its scheme, and whatever smooths it undoes that scheme's
window first. By default the window is divided out in full, which also boosts
the aliases and shot noise near the grid scale (up to 15 times at the corner
of a CIC grid's Fourier cube); `aliasing` divides by the window and its
aliases together, a least-squares correction that boosts nothing by more than
about 2 for CIC, and accounts for interlacing.

:::::{tab-set}

::::{tab-item} Python
```python
grid = pysif.Grid(n_cells=n_cells, box_length=box_length)
grid.assign(field, "tsc", interlace=True)  # assign(field) is assign_cic(field)
grid.to_density_contrast()
voids = pysif.finders.exodus(grid, mesh, radii, -0.7, deconvolution="aliasing")
```
::::

::::{tab-item} C
```c
#include "sif/structures/grid.h"

if (sif_grid_assign(grid, field, SIF_ASSIGNMENT_TSC, SIF_GRID_INTERLACE) !=
      SIF_OK ||
    sif_grid_to_density_contrast(grid) != SIF_OK)
  goto done;
// ... sif_finder_exodus(..., SIF_FINDER_ALIASING_WINDOW)
```
::::

::::{tab-item} sif-exodus
```lua
grid = { n_cells = 512, assignment = "tsc", interlace = true }
finder = { ..., deconvolution = "aliasing" }
```
::::

:::::

## Memory

A chain mesh can take over the memory of a field instead of copying it
(`consume_field=True` in Python, `sif_chain_mesh_alloc_consume()` in C). The
field is left empty, so the grid has to be assigned first:

```python
grid.assign_cic(field=field)
mesh = pysif.ChainMesh(
    n_cells=n_cells_mesh, box_length=box_length, field=field,
    drop_indices=True, consume_field=True
)
```

In C, a consumed field still has to be released with `sif_field_free()`.
