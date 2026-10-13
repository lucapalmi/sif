# Catalogues

How to read, write and handle void catalogues

In **sif**, a list of voids is represented by a catalogue object, which stores
their centres and radii. Catalogues are produced by the finders and consumed
by the measurements (profiles, size function).

## Contents of a catalogue

A catalogue holds:

- **centres**: three arrays, either Cartesian positions or sky coordinates
  (right ascension and declination in degrees, and redshift; see
  [](#surveys)), specified by `units`.
- **radii**: one array, always a comoving length in Mpc/h.
- **footprint**, optionally: two arrays, `footprint` and `footprint_shell`,
  with the fraction of each void's sphere, and of the shell between one and
  two radii, that lies inside the survey. Only catalogues from the survey
  finder carry them.
- **metadata**: named values (integers, reals or strings) describing the
  catalogue, such as the finder settings or the cosmology.

## Reading and writing

Catalogues can be written in three formats, and every format stores the
footprint and the metadata:

| Format | Python | C |
| --- | --- | --- |
| ASCII | `pysif.io.write_catalogue_ascii` / `read_catalogue_ascii` | `sif_catalogue_write_ascii` / `sif_catalogue_read_ascii` |
| HDF5 | `pysif.io.write_catalogue_hdf5` / `read_catalogue_hdf5` | `sif_catalogue_write_hdf5` / `sif_catalogue_read_hdf5` |
| FITS | `pysif.io.write_catalogue_fits` / `read_catalogue_fits` | `sif_catalogue_write_fits` / `sif_catalogue_read_fits` |

In the C snippets below, `catalogue` is declared as
`sif_catalogue_t* catalogue = NULL;`, `goto done` jumps to the cleanup at the
end of the function, and every catalogue is released with
`sif_catalogue_free()`.

### ASCII

A header behind `#`, then one row per void. The header holds the number of
voids, the metadata (`#key=value`) and the column names: `cx cy cz r`, or
`ra dec z r` on the sky, followed by `footprint footprint_shell` when present.

```text
#n=1024
#threshold=-0.7
#cx cy cz r footprint footprint_shell
101.25 250.5 33.125 12.5 1 0.84
...
```

The rows are plain numbers, readable with `numpy.loadtxt`. Values are written
with enough digits to be read back exactly.

:::::{tab-set}

::::{tab-item} Python
```python
import pysif

pysif.io.write_catalogue_ascii("path/to/voids.txt", catalogue)
catalogue = pysif.io.read_catalogue_ascii("path/to/voids.txt")
```
::::

::::{tab-item} C
```c
#include "sif/io/catalogue_io.h"

if (sif_catalogue_write_ascii("path/to/voids.txt", catalogue) != SIF_OK)
  goto done;

// NULL takes the columns from the header
catalogue = sif_catalogue_read_ascii("path/to/voids.txt", NULL);
if (!catalogue)
  goto done;
```
::::

:::::

#### Catalogues from other finders

Text catalogues written by other codes are read by giving a column format, as
for [fields](fields.md#ascii-tables): `x y z` (or `cx cy cz`) or `ra dec z`
for the centres, `r` for the radius, `*` for a skipped column. Columns can be
separated by blanks or commas, and lines starting with `#` are skipped.

:::::{tab-set}

::::{tab-item} Python
```python
# ID, centre, volume, radius
catalogue = pysif.io.read_catalogue_ascii(
    "path/to/other_voids.txt", format="* x y z * r"
)
```
::::

::::{tab-item} C
```c
catalogue = sif_catalogue_read_ascii("path/to/other_voids.txt", "* x y z * r");
if (!catalogue)
  goto done;
```
::::

:::::

For any other format, read the columns with your own code and build the
catalogue from arrays (see [](#building-a-catalogue-from-arrays)).

### HDF5

The catalogue is stored in the `/catalogue` group: `centres` (N, 3), `radii`,
and `footprint` and `footprint_shell` when present; the metadata are
attributes of the group. The same file can also hold the profiles and the size
function, each in its own group: every writer creates the file if it is
missing and replaces only its own group. The file is readable with `h5py`.

:::::{tab-set}

::::{tab-item} Python
```python
pysif.io.write_catalogue_hdf5("path/to/voids.h5", catalogue)
catalogue = pysif.io.read_catalogue_hdf5("path/to/voids.h5")
```
::::

::::{tab-item} C
```c
#include "sif/io/hdf5_io.h"

if (sif_catalogue_write_hdf5("path/to/voids.h5", catalogue) != SIF_OK)
  goto done;

catalogue = sif_catalogue_read_hdf5("path/to/voids.h5");
if (!catalogue)
  goto done;
```
::::

:::::

:::{note}
In a build of **sif** without HDF5 support, the writers save the data as text
in `<path>.<group>.txt` instead, with a warning; the readers fail.
:::

### FITS

The catalogue is stored in a table named `VOIDS`: columns `CX`, `CY`, `CZ`,
`R` (or `RA`, `DEC`, `Z`, `R` on the sky), then `FOOTPRINT` and
`FOOTPRINT_SHELL` when present; the metadata are keywords of the table. As for
HDF5, the same file can hold the profiles and the size function, and every
writer replaces only its own table.

:::::{tab-set}

::::{tab-item} Python
```python
pysif.io.write_catalogue_fits("path/to/voids.fits", catalogue)
catalogue = pysif.io.read_catalogue_fits("path/to/voids.fits")
```
::::

::::{tab-item} C
```c
#include "sif/io/fits_io.h"

if (sif_catalogue_write_fits("path/to/voids.fits", catalogue) != SIF_OK)
  goto done;

catalogue = sif_catalogue_read_fits("path/to/voids.fits");
if (!catalogue)
  goto done;
```
::::

:::::

:::{note}
Reading and writing FITS files needs a build of **sif** with cfitsio. Neither
HDF5 nor FITS writers touch an existing file that was not written by **sif**.
:::

### sif-exodus

**sif-exodus** picks the format from the extension of `output` (`.h5` or
`.hdf5` for HDF5, `.fits` for FITS, anything else for ASCII), or from an
explicit `format`:

```lua
output = "path/to/voids.h5"
-- or
output = { path = "path/to/voids.dat", format = "ascii" }  -- hdf5 | ascii | fits
```

The settings of the run are stored in the metadata (`threshold`,
`search_factor`, `box_length`, `n_tracers`, the cosmology for a survey, ...).

## Building a catalogue from arrays

In Python, `pysif.catalogue_from_numpy()` copies numpy arrays into a new
catalogue. In C, voids are appended one at a time, and the catalogue grows as
needed.

:::::{tab-set}

::::{tab-item} Python
```python
import pysif

# cartesian centres
catalogue = pysif.catalogue_from_numpy(cx, cy, cz, r)

# sky centres, with footprint
catalogue = pysif.catalogue_from_numpy(
    ra=ra, dec=dec, z=redshift, r=r,
    footprint=footprint, footprint_shell=footprint_shell,
)
```
::::

::::{tab-item} C
```c
#include "sif/structures/catalogue.h"

// initial capacity: grows when full
catalogue = sif_catalogue_alloc(n);
if (!catalogue)
  goto done;

for (uint64_t i = 0; i < n; i++)
  if (sif_catalogue_append(catalogue, cx[i], cy[i], cz[i], r[i]) != SIF_OK)
    goto done;

// optional: footprint columns, set to SIF_CATALOGUE_FOOTPRINT_UNKNOWN
if (sif_catalogue_reserve_footprint(catalogue) != SIF_OK)
  goto done;

// for sky centres
catalogue->units = SIF_COORDINATES_SKY;

// optional: release the unused capacity
if (sif_catalogue_trim(catalogue) != SIF_OK)
  goto done;
```
::::

:::::

## Accessing the data

In Python, `radii`, `footprint` and `footprint_shell` are read-only numpy
views of the catalogue's arrays (`None` for a catalogue without footprint),
while `centres` is an `(N, 3)` copy. `to_numpy()` returns a copy of the whole catalogue as a
structured array, with fields named as the columns of the ASCII header.

```python
print(catalogue.n_voids, catalogue.units)
print(catalogue.centres.shape, catalogue.radii.max())

table = catalogue.to_numpy()  # fields cx, cy, cz, r, [footprint, footprint_shell]
df = pandas.DataFrame(table)
```

Subsets are built from the arrays, e.g. the voids that lie mostly inside the
survey:

```python
keep = catalogue.footprint > 0.9
cx, cy, cz = catalogue.centres[keep].T
selected = pysif.catalogue_from_numpy(
    cx, cy, cz, catalogue.radii[keep],
    footprint=catalogue.footprint[keep],
    footprint_shell=catalogue.footprint_shell[keep],
)
```

:::{note}
A catalogue built this way starts without metadata: copy over the entries you
need with `set_metadata()`.
:::

In C, the arrays are the members of `sif_catalogue_t` (`cx`, `cy`, `cz`,
`radii`, `footprint`, `footprint_shell`). They are reallocated by
`sif_catalogue_append()` and `sif_catalogue_trim()`: do not keep pointers to
them across these calls.

## Metadata

Keys are case-insensitive identifiers (stored in lower case), values are
integers, reals or strings. Setting an existing key replaces it.

:::::{tab-set}

::::{tab-item} Python
```python
catalogue.set_metadata("threshold", -0.7)
catalogue.set_metadata("survey", "BOSS DR12")
catalogue.set_metadata("survey", None)  # removes the key

print(catalogue.metadata)  # a dict, copied
```
::::

::::{tab-item} C
```c
if (sif_catalogue_meta_real_set(catalogue, "threshold", -0.7) != SIF_OK ||
    sif_catalogue_meta_string_set(catalogue, "survey", "BOSS DR12") != SIF_OK)
  goto done;

const double threshold = sif_catalogue_meta_real_get(catalogue, "threshold");
const char* survey = sif_catalogue_meta_string_get(catalogue, "survey");

// every key, in the order they were set
for (uint32_t i = 0; i < sif_catalogue_meta_count(catalogue); i++)
  printf("%s\n", sif_catalogue_meta_name(catalogue, i));
```
::::

:::::

(surveys)=
## Surveys

A catalogue from the survey finder has its centres in the box the survey was
shifted into (see [Shifting surveys](fields.md#shifting-surveys)). `translate` moves them
back by the opposite offset, and `to_sky` converts them to right ascension,
declination and redshift, the inverse of the conversion applied to the fields.
`to_sky` requires the observer at the origin, so it is applied after
`translate`, with the same cosmology used for the fields. Radii stay in Mpc/h.

:::::{tab-set}

::::{tab-item} Python
```python
catalogue.translate(-offset)
catalogue.to_sky(**cosmology)
print(catalogue.units)  # 'sky'
```
::::

::::{tab-item} C
```c
const sif_real back[3] = {-offset[0], -offset[1], -offset[2]};
if (sif_catalogue_translate(catalogue, back) != SIF_OK ||
    sif_catalogue_to_sky(catalogue, &cosmology) != SIF_OK)
  goto done;
```
::::

::::{tab-item} sif-exodus
Done automatically in survey mode: the catalogue is written on the sky with
`coordinates = "sky"`, in the input frame with `coordinates = "cartesian"`.
::::

:::::

:::{note}
Sky catalogues are refused by `translate` and by the measurements. To measure
profiles around survey voids, use the catalogue in the same frame as the chain
mesh, i.e. before translating it back, or translate it again by `offset`.
:::
