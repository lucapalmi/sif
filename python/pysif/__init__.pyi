# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The types of pysif, for editors and type checkers, which cannot read the
# compiled module. The types are written here by hand; the parameters and
# docstrings are the compiled module's, and packaging/stubs.py checks them
# against it (tests/python/test_stubs.py) and rewrites the docstrings
# (packaging/stubs.py fix). Edit the docstrings in the C, not here.

"""sif: cosmic void finding and analysis.

The finders, the grid-based delta statistics and the settings need
system_init() first, and system_finalise() when done; everything else
works without it.

The data structures live here in the package root -- Field, Grid,
Catalogue and the rest -- and the operations on them are grouped
into submodules: io for reading and writing, finders for void
identification, measure for measurements taken from data, and
model for theoretical predictions.

pysif.real is the NumPy scalar type matching the precision the
library was built with, so np.zeros(n, dtype=pysif.real) gives
arrays the bindings take without a conversion copy.
"""

from collections.abc import Sequence
from typing import Any, Literal, TypedDict, TypeVar, final, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import finders as finders
from . import io as io
from . import measure as measure
from . import model as model

# The precision pysif is built with: float32, unless built from source with
# SIF_USE_DOUBLE, which makes it float64.
_Real = np.float32
_RealArray = NDArray[_Real]
_Vec3 = Sequence[float] | NDArray[np.floating[Any]]
_Units = Literal["cartesian", "sky"]
_T = TypeVar("_T")

real: type[_Real]

class _WrapCounts(TypedDict):
    boundary: int
    wrapped: int
    nonfinite: int

def system_init(threads: int = 0, skip_tuning: bool = False,
                log_level: int = 2) -> None:
    """Bring up the system state: FFTW, the settings table, the thread
    ceiling and the log level. Calling it twice warns and does nothing.

    Only some of pysif needs it: the finders (finders.spherical,
    finders.exodus, finders.exodus_survey) and
    measure.delta_moments_grid and measure.delta_distribution_grid, which
    transform a grid, and get_setting and set_setting. Called without it,
    those raise RuntimeError (get_setting returns its fallback instead).
    Everything else works without it, using OpenMP's own thread count and
    logging at info level.

    Args:
        threads: Thread ceiling; 0 leaves OpenMP's own default.
        skip_tuning: Take FFTW's estimated plan instead of tuning. Faster
            to start, slower to transform; worth setting for short runs.
        log_level: 0 trace, 1 debug, 2 info, 3 warning, 4 error, 5 none.
            0 is the verbose mode: everything, timings included.
    """

def system_finalise() -> None:
    """Release everything system_init() acquired.

    Saves the settings table if it changed and tears down the FFTW plan
    cache. The functions that need system_init() may not be called
    afterwards without initialising again.
    """

def init(threads: int = 0, skip_tuning: bool = False,
         log_level: int = 2) -> None:
    """Deprecated name of system_init(); warns and forwards."""

def finalise() -> None:
    """Deprecated name of system_finalise(); warns and forwards."""

def field_from_numpy(x: ArrayLike | None = None, y: ArrayLike | None = None,
                     z: ArrayLike | None = None, *,
                     ra: ArrayLike | None = None, dec: ArrayLike | None = None,
                     vx: ArrayLike | None = None, vy: ArrayLike | None = None,
                     vz: ArrayLike | None = None,
                     weights: ArrayLike | None = None) -> Field:
    """A new Field, with its particles copied out of NumPy arrays.

    The names say what the positions are: x, y and z for Cartesian
    positions, or ra, dec and z for sky coordinates -- right ascension and
    declination in degrees, and redshift -- which give a Field with
    units='sky', to be turned into positions by
    Field.convert_sky_coordinates():

        box = pysif.field_from_numpy(x, y, z, weights=w)
        survey = pysif.field_from_numpy(ra=ra, dec=dec, z=redshift)

    All arrays must have the same length; an array not of dtype
    pysif.real is converted, which costs a copy.

    Args:
        x, y, z: Position components, one entry per particle; z is the
            redshift when ra and dec are given.
        ra, dec: Sky coordinates, in place of x and y.
        vx, vy, vz: Velocity components, all three or none.
        weights: Per-particle weight (a mass, a luminosity, a selection
            weight), or None for an unweighted field, where every particle
            counts as 1. The exodus finder requires them to be finite and
            non-negative.

    Returns:
        Field: The new field.

    Raises:
        ValueError: If the arrays disagree in length, a position is
            missing, x or y is given with ra or dec, or the velocities are
            given in part.
        MemoryError: If the field's buffers could not be allocated.
    """

def catalogue_from_numpy(cx: ArrayLike | None = None,
                         cy: ArrayLike | None = None,
                         cz: ArrayLike | None = None,
                         r: ArrayLike | None = None, *,
                         ra: ArrayLike | None = None,
                         dec: ArrayLike | None = None,
                         z: ArrayLike | None = None,
                         footprint: ArrayLike | None = None,
                         footprint_shell: ArrayLike | None = None
                         ) -> Catalogue:
    """A new Catalogue, with its voids copied out of NumPy arrays: a catalogue
    from another finder, or one cut down by hand.

    The names say what the centres are, as for field_from_numpy(): cx, cy
    and cz for Cartesian centres, or ra, dec and z -- right ascension and
    declination in degrees, and redshift -- for a catalogue on the sky:

        box = pysif.catalogue_from_numpy(cx, cy, cz, r)
        sky = pysif.catalogue_from_numpy(ra=ra, dec=dec, z=z, r=r)

    Catalogue.to_numpy() is the way back.

    Args:
        cx, cy, cz: Cartesian centres, one entry per void.
        r: Radii.
        ra, dec, z: Sky centres, in place of cx, cy and cz.
        footprint, footprint_shell: The footprint columns, both or neither.

    Returns:
        Catalogue: The new catalogue.

    Raises:
        ValueError: If the arrays disagree in length, a column is missing,
            or Cartesian and sky names are mixed.
    """

def set_setting(key: str, value: str | int | float | bool | None,
                persist: bool = True) -> None:
    """Set a runtime setting.

    A setting reads as the first of: a persist=False value, the environment
    variable SIF_<KEY>, the settings file ($SIF_HOME/config, or
    $HOME/.sif/config), the default. The keys are cache_directory,
    grid_cache_enabled, fft_wisdom_dir, fft_tuning_max_gib,
    cic_tile_particles and scratch_directory.

    Values may reference environment variables ($HOME, ${USER}), and
    $SIF_HOME, the directory holding the settings file.

    Requires pysif.system_init().

    Args:
        key: Setting name.
        value: Value to store. With persist=False, None removes the value
            set before.
        persist: Save to the settings file, for this and later runs. False
            sets it for this process only, over the environment and the
            file.

    Raises:
        ValueError: If the key is not a setting, the value is not one the
            key can hold, or contains a newline, which the settings file
            could not hold.
        RuntimeError: If system_init() has not been called.
    """

@overload
def get_setting(key: str, fallback: None = None) -> str | None:
    """Read a runtime setting, as set_setting() describes. Nothing is stored.

    Args:
        key: Setting name.
        fallback: Returned if the key is not a setting, or before
            system_init().

    Returns:
        str: The setting's value, with environment variables expanded, or
            the fallback.
    """
@overload
def get_setting(key: str, fallback: _T) -> str | _T: ...

@final
class Bitmask:
    """A dense array of bits, packed 64 to a word.

    Used where one boolean per grid cell would otherwise cost a byte:
    at 1024**3 cells that is 128 MiB rather than 1 GiB.

    Args:
        n_bits: Number of bits. Must be non-zero.
    """

    def __init__(self, n_bits: int) -> None: ...
    @property
    def n_bits(self) -> int:
        """The total number of bits in the bitmask"""
    @property
    def words(self) -> NDArray[np.uint64]:
        """1D NumPy array of underlying 64-bit words"""
    def clear_all(self) -> None:
        """Clear all bits."""
    def count_set(self) -> int:
        """Count the number of set bits."""
    def set(self, index: int, /) -> None:
        """Set a bit at given index."""
    def unset(self, index: int, /) -> None:
        """Unset a bit at given index."""
    def get(self, index: int, /) -> bool:
        """Get value of bit at given index."""

@final
class Catalogue:
    """A list of voids: centre and radius, one entry each.

    What a finder produces and what pysif.measure consumes. Read one
    back with pysif.io.read_catalogue_ascii(). A catalogue from
    finders.exodus_survey() also carries footprint and
    footprint_shell.

    Args:
        capacity: Voids to make room for up front; it grows as
            needed.
    """

    def __init__(self, capacity: int = 0) -> None: ...
    @property
    def metadata(self) -> dict[str, int | float | str]:
        """dict: the named values describing the catalogue -- the finder and its
        settings, the cosmology, anything set with set_metadata() -- a copy.
        Every writer records them and every reader gives them back: '#key=value'
        lines in text, attributes of /catalogue in HDF5, keywords of VOIDS in
        FITS. Keys are in lower case.
        """
    @property
    def units(self) -> _Units:
        """What the centres are: 'cartesian', as a finder gives them, or 'sky' --
        right ascension, declination (degrees) and redshift -- after to_sky().
        """
    @property
    def centres(self) -> _RealArray:
        """Nx3 NumPy array of void centres (x, y, z, or ra, dec, z on the sky),
        a copy.
        """
    @property
    def radii(self) -> _RealArray:
        """1D NumPy array of void radii, a read-only view of the catalogue's own
        array (no copy).
        """
    @property
    def n_voids(self) -> int:
        """The number of voids in the catalogue"""
    @property
    def footprint(self) -> _RealArray | None:
        """1D NumPy array, a read-only view: the fraction of each void's sphere
        inside the survey footprint, or None for a catalogue with no footprint
        (a periodic box).
        -1 marks a void nobody measured it for.
        """
    @property
    def footprint_shell(self) -> _RealArray | None:
        """1D NumPy array: the same fraction over the shell between one and two
        radii, or None with footprint.
        """
    def set_metadata(self, key: str, value: int | float | str | None) -> None:
        """Set one of the catalogue's named values, which every writer records:
        an int, a float or a str; None removes the key.

            cat.set_metadata("omega_m", 0.31)
            cat.set_metadata("survey", "Euclid DR1")

        Keys are matched without regard to case, as FITS keywords are.

        Raises:
            ValueError: For a key or a value no format can keep.
        """
    def to_numpy(self) -> NDArray[np.void]:
        """The catalogue as one NumPy structured array, a copy: a record per
        void, with fields named as the file headers name them -- cx, cy, cz, r
        (or ra, dec, z, r on the sky), and footprint, footprint_shell when the
        catalogue has them. pandas.DataFrame(cat.to_numpy()) and
        astropy.table.Table(cat.to_numpy()) take it as it is.
        """
    def translate(self, offset: _Vec3) -> None:
        """Shift every void centre by offset, in place.

        Radii and footprints are unchanged. finders.exodus_survey() already
        returns its voids in your own frame, so this is for catalogues moved
        some other way.

        Args:
            offset: Three numbers, added to x, y and z.

        Raises:
            ValueError: For a catalogue on the sky.
        """
    def to_sky(self, omega_m: float, omega_de: float | None = None,
               omega_r: float = 0.0, w0: float = -1.0,
               wa: float = 0.0) -> None:
        """Turn the void centres into sky coordinates, in place.

        The inverse of Field.convert_sky_coordinates(), for the voids a survey
        gave: each centre becomes the right ascension and declination of its
        direction from the origin, in degrees (right ascension in [0, 360)),
        and the redshift at its comoving distance from the origin. units is
        then 'sky', and every writer writes the catalogue that way.

        The observer has to be at the origin, which is where
        finders.exodus_survey() returns its voids, whatever box the survey was
        searched in. Converted with the cosmology the tracers were, a centre
        comes back to the sky within single precision. Radii stay comoving
        lengths, in Mpc/h.

        Args:
            omega_m, omega_de, omega_r, w0, wa: The cosmology, as for
                Field.convert_sky_coordinates().

        Raises:
            ValueError: For a catalogue already on the sky, a centre that is
                not finite, or one past what the cosmology reaches.
        """
    def to_cartesian(self, omega_m: float, omega_de: float | None = None,
                     omega_r: float = 0.0, w0: float = -1.0,
                     wa: float = 0.0) -> None:
        """Turn void centres on the sky back into Cartesian ones, in place.

        The inverse of to_sky(), and the conversion
        Field.convert_sky_coordinates() applies to tracers: comoving
        positions in Mpc/h with the observer at the origin. For voids written
        on the sky that are to be measured against tracers, or passed to
        finders.exodus_survey() as its prior. units is then 'cartesian'.
        Radii and footprints are unchanged.

        Args:
            omega_m, omega_de, omega_r, w0, wa: The cosmology, as for
                Field.convert_sky_coordinates() -- the one the voids were put
                on the sky with, for them to come back where they were.

        Raises:
            ValueError: For a catalogue already Cartesian, a right ascension
                that is not finite, a declination outside [-90, 90] or a
                redshift that is negative or not finite.
        """

@final
class CellLinkedList:
    """A spatial bin built from linked lists, one per cell.

    Where ChainMesh sorts a fixed set of particles, this takes items
    one at a time and never moves an existing entry, which is what a
    finder needs while it is still accepting voids.

    Args:
        n_cells: Cells per side.
        box_length: Physical side length of the box.
        initial_capacity: Items to make room for up front.
        options: Boundary convention, SIF_PBC_PERIODIC (default) or
            SIF_PBC_OPEN.
    """

    def __init__(self, n_cells: int, box_length: float,
                 initial_capacity: int = 1, options: int = 0) -> None: ...
    @property
    def n_cells(self) -> int:
        """Dimension of the coarse grid"""
    @property
    def total_cells(self) -> int:
        """Total number of cells"""
    @property
    def inv_cell_length(self) -> float:
        """Inverse of the cell length"""
    @property
    def capacity(self) -> int:
        """Capacity of the item array"""
    @property
    def periodic(self) -> bool:
        """True if out-of-box coordinates wrap, False if they clamp"""
    @property
    def head(self) -> NDArray[np.int32]:
        """1D NumPy array of head indices"""
    @property
    def next(self) -> NDArray[np.int32]:
        """1D NumPy array of next indices"""
    def ensure_capacity(self, required_capacity: int) -> None:
        """Ensure capacity of the item array."""
    def insert(self, item_idx: int, cx: float, cy: float, cz: float) -> None:
        """Insert an item into the list."""

@final
class ChainMesh:
    """A uniform spatial bin over a periodic box, for neighbour queries.

    Particles are copied into cell order, so the members of a cell sit
    contiguously in memory. Velocities and weights are mirrored from the
    field when it carries them, each costing another pass over the field
    and as much memory again.

    That copy means the field's arrays and the mesh's are both alive
    while the mesh is built -- at 3.4e9 tracers, 81 GB to describe 40 GB
    of particles. consume_field takes the field's storage over instead of
    copying it, needing one spare column (4 bytes per particle) rather
    than a second copy of every payload. The field is left empty and
    must not be relied on afterwards; it is still safe to hold and to
    delete, and n_particles reads back as 0.

    original_indices is always built, and not as a convenience: the
    scatter claims slots atomically, so the order inside a cell is a race
    until each cell's run is sorted by source index. Without that key two
    identical runs give different meshes. drop_indices releases it once
    the sort is done -- the same mesh, 8 bytes per particle lighter (25
    GiB at 3.4e9 tracers) -- at the cost of nearest-neighbour queries,
    which have no way left to name their answer.

    Every coordinate must lie in [0, box_length).

    Args:
        n_cells: Cells per side.
        box_length: Physical side length of the box.
        field: Particle field to bin.
        drop_indices: Release the field index map after construction.
            original_indices then reads back as None.
        consume_field: Build the mesh out of the field's own storage,
            emptying it. The field is consumed whether or not the call
            succeeds, so do not pass a field you still need.
        canonical: Sort each cell by source index, so two identical runs
            give byte-identical meshes. On by default. Turning it off
            leaves cells in scatter order, which changes nothing about
            which particles a cell holds -- only the order they are
            summed in, and so the last bits of anything that walks them.
            The unweighted exodus finder only counts tracers and does not
            care; stacked profiles, anything that sums weights (exodus on a
            weighted mesh included) and nearest-neighbour distance ties do.
    """

    def __init__(self, n_cells: int, box_length: float, field: Field,
                 drop_indices: bool = False, consume_field: bool = False,
                 canonical: bool = True) -> None: ...
    @property
    def n_cells(self) -> int:
        """Number of cells"""
    @property
    def total_cells(self) -> int:
        """Total cells"""
    @property
    def box_length(self) -> float:
        """Box length"""
    @property
    def cell_length(self) -> float:
        """Cell length"""
    @property
    def n_particles(self) -> int:
        """Number of particles"""
    @property
    def x(self) -> _RealArray:
        """X coordinates"""
    @property
    def y(self) -> _RealArray:
        """Y coordinates"""
    @property
    def z(self) -> _RealArray:
        """Z coordinates"""
    @property
    def vx(self) -> _RealArray | None:
        """X velocities"""
    @property
    def vy(self) -> _RealArray | None:
        """Y velocities"""
    @property
    def vz(self) -> _RealArray | None:
        """Z velocities"""
    @property
    def weights(self) -> _RealArray | None:
        """Per-particle weights"""
    @property
    def original_indices(self) -> NDArray[np.uint64] | None:
        """Map from mesh order back to field order, or None when the mesh was built with drop_indices=True."""
    @property
    def cell_offsets(self) -> NDArray[np.uint64]:
        """Cell offsets"""

@final
class Field:
    """A particle field: positions, and optionally velocities and
    weights.

    The container every other structure is built from. Make one with
    pysif.field_from_numpy(), or read one off disk with
    pysif.io.read_field(), read_fits() or read_gadget(). Field() itself
    is empty.

    x, y, z, vx, vy, vz and weights are read-only NumPy views of the
    field's own arrays, in its current particle order -- which
    sort_morton() changes. They see in-place changes (wrap(),
    translate()). While any view is alive, whatever would reallocate or
    take the arrays raises BufferError instead: sort_morton() and
    ChainMesh(..., consume_field=True). Delete the views first, or keep
    copies
    (numpy.array(field.x)).

    units says what x, y and z hold: 'cartesian' positions, or 'sky' --
    right ascension, declination and redshift.

    Args:
        capacity: Particles to make room for up front.
    """

    def __init__(self, capacity: int = 0) -> None: ...
    @property
    def n_particles(self) -> int:
        """int: Number of particles the field holds."""
    @property
    def has_weights(self) -> bool:
        """bool: Whether the field carries per-particle weights. Everything built
        from a weighted field -- grids, meshes, the finders, the profiles --
        uses them.
        """
    @property
    def has_velocities(self) -> bool:
        """bool: Whether the field carries velocities."""
    @property
    def units(self) -> _Units:
        """str: What the positions are: 'cartesian', or 'sky' for right
        ascension and declination (degrees) and redshift held in x, y, z.
        A sky field is refused by everything that reads positions as lengths
        -- grids, meshes, wrap(), translate(), sort_morton() -- until
        convert_sky_coordinates() turns it into positions. Setting it only
        relabels the arrays; it does not convert them.
        """
    @units.setter
    def units(self, value: _Units) -> None: ...
    @property
    def x(self) -> _RealArray:
        """ndarray: x positions, a read-only view of the field's own array (no
        copy). See the class docstring for what a live view prevents.
        """
    @property
    def y(self) -> _RealArray:
        """ndarray: y positions; see x."""
    @property
    def z(self) -> _RealArray:
        """ndarray: z positions; see x."""
    @property
    def vx(self) -> _RealArray | None:
        """ndarray or None: x velocities, a read-only view; see x."""
    @property
    def vy(self) -> _RealArray | None:
        """ndarray or None: y velocities; see vx."""
    @property
    def vz(self) -> _RealArray | None:
        """ndarray or None: z velocities; see vx."""
    @property
    def weights(self) -> _RealArray | None:
        """ndarray or None: per-particle weights, a read-only view; see x."""
    def to_numpy(self) -> _RealArray:
        """The field as one (N, k) NumPy array, a copy: a row per particle, with
        columns x, y, z (ra, dec, z on the sky), then vx, vy, vz if the field
        has velocities, then the weight if it has weights -- 3, 6, 4 or 7
        columns. Rows are in the field's current order, which sort_morton()
        changes.

        Raises:
            ValueError: For a field with particles but no positions.
        """
    def convert_sky_coordinates(self, omega_m: float,
                                omega_de: float | None = None,
                                omega_r: float = 0.0, w0: float = -1.0,
                                wa: float = 0.0) -> None:
        """Turn sky coordinates into comoving Cartesian positions, in place.

        The field has to hold right ascension, declination (both in degrees)
        and redshift in x, y and z, with units='sky'. Each tracer goes to the
        line-of-sight comoving distance of its redshift, along its direction,
        with the observer at the origin -- the convention of pyrecon's
        sky_to_cartesian(). Positions come out in Mpc/h, and units becomes
        'cartesian'. Weights and velocities are untouched.

        Every coordinate is checked first: on a declination outside [-90, 90]
        or a negative or non-finite redshift nothing is changed.

        Args:
            omega_m: Matter density today.
            omega_de: Dark-energy density today; None, the default, makes the
                model flat.
            omega_r: Radiation density today.
            w0, wa: The dark-energy equation of state, w(a) = w0 + wa (1 - a):
                -1 and 0 for a cosmological constant.

        Raises:
            ValueError: For a field that is not in sky coordinates or holds
                one out of range, a bad cosmology, or one with no expansion
                history out to the largest redshift.
        """
    def wrap(self, box_length: float) -> _WrapCounts:
        """Fold every coordinate into [0, box_length) periodically, in place.

        Not needed for rounding: Grid.assign_cic() and ChainMesh() accept
        coordinates within a few ULPs of the box faces -- a coordinate stored
        as exactly box_length, say -- and fold them themselves. This is for a
        periodic snapshot whose coordinates were never wrapped into the box.

        Returns {'boundary': n, 'wrapped': n, 'nonfinite': n}: 'boundary'
        counts coordinates that sat exactly on the box edge, and 'wrapped'
        coordinates that were genuinely outside. A large 'wrapped' means the
        box length is wrong and the folded field is meaningless -- check it
        rather than proceeding. 'nonfinite' counts NaN and infinite
        coordinates, which no wrap repairs: they are left as they are, with a
        RuntimeWarning, and the field will be refused by anything that bins
        it. Only correct for a field that is periodic in this box.
        """
    def translate(self, offset: _Vec3) -> None:
        """Shift every position by offset, in place.

        What moves a survey into the box pysif.finders.survey_box() chose for
        it: data and randoms both, by the same offset. The voids found there
        come back out with Catalogue.translate(-offset).

        Args:
            offset: Three numbers, added to x, y and z.
        """
    def sort_morton(self) -> None:
        """Reorder the particles along a 3D Morton curve, in place.

        Puts particles that are close in space close in memory, which is what
        makes walking them in space order cache-friendly. The
        permutation is kept, so velocities and weights assigned afterwards are
        matched to their own particles automatically.
        """
    def refresh_bounds(self) -> None:
        """Recompute the cached bounding box and centre.

        Only needed after writing into the position arrays directly; every
        operation that goes through this object keeps the bounds current.
        """

@final
class Grid:
    """A cubic grid of n_cells**3 cells over a periodic box.

    Deposit a field onto it with assign_cic(), then convert it with
    to_density_contrast() before handing it to a finder or to
    pysif.measure.

    Args:
        n_cells: Cells per side.
        box_length: Physical side length of the box.
    """

    def __init__(self, n_cells: int, box_length: float) -> None: ...
    @property
    def assignment(self) -> Literal["ngp", "cic", "tsc", "pcs"] | None:
        """The scheme the grid was assigned with -- 'ngp', 'cic', 'tsc' or 'pcs' --
        or None if nothing has been deposited, or the grid was read from a file
        that does not record it. The finders and pysif.measure undo this
        scheme's window (CIC's when it is None).
        """
    @property
    def interlaced(self) -> bool:
        """Whether the grid was assigned with interlace=True."""
    def assign_cic(self, field: Field) -> None:
        """Deposit a particle field onto the grid by Cloud-In-Cell assignment.

        Each particle contributes to the eight cells around it. Weights are
        used when the field carries them, otherwise every particle counts as
        one. Afterwards the cells hold a density, not a density contrast.

        Every coordinate must lie in [0, box_length); use Field.wrap() first
        if a periodic snapshot has drifted onto the boundary, or
        Field.translate() to move a survey into its box.

        Args:
            field: The particle field to deposit.

        Raises:
            ValueError: If a particle lies outside the box, or the field holds
                none. The grid is left as it was.
            MemoryError: If the deposit could not allocate its scratch.
        """
    def assign(self, field: Field,
               scheme: Literal["ngp", "cic", "tsc", "pcs"] = "cic", *,
               interlace: bool = False) -> None:
        """Deposit a particle field onto the grid with a mass-assignment scheme.

        The schemes spread each particle over 1, 2, 3 or 4 cells per axis:
        nearest grid point, cloud-in-cell, triangular-shaped cloud and
        piecewise cubic spline. A higher order folds less power from above the
        grid's Nyquist frequency back onto the resolved modes, at the cost of a
        deposit that touches more cells (27 for TSC, 64 for PCS, against 8).

        interlace=True deposits the field a second time shifted by half a cell
        along each axis and averages the two in Fourier space, which cancels
        the leading aliasing images outright. It costs a second deposit and a
        forward and backward FFT.

        assign(field) is assign_cic(field). The grid records the scheme, and
        the finders and pysif.measure undo that scheme's window.

        Args:
            field: The particle field to deposit.
            scheme: 'ngp', 'cic', 'tsc' or 'pcs'.
            interlace: Average with the half-cell-shifted deposit.

        Raises:
            ValueError: On an unknown scheme, a grid of fewer than 4 cells per
                side, a particle outside the box, or a field that holds none.
                The grid is left as it was.
            MemoryError: If the deposit could not allocate its scratch.
        """
    def assign_cic_tessellation(self, tessellation: Tessellation) -> None:
        """Deposit a Tessellation onto the grid by Cloud-In-Cell assignment.

        The tessellation's density rather than the tracers': every sample
        carries a share of its owner's weight and lands where the owner's cell
        actually reaches, so a region with few tracers is filled by whatever
        large cells cover it instead of being left empty. Depositing sparse
        tracers directly leaves most cells holding nothing and the rest holding
        shot noise, and no smoothing afterwards puts back a field that was
        never sampled.

        The result is a density on the same scale and with the same mean as
        assign_cic() of the tracers, so to_density_contrast() and everything
        downstream treat it identically.

        Costs the sampling rate times what depositing the tracers would.

        Args:
            tessellation: The Tessellation to deposit. Its samples must still
                be present -- not one whose mesh(consume=True) has taken them.
        """
    def to_density_contrast(self) -> None:
        """Convert the cells in place from a density to a density contrast.

        Replaces each cell with delta = rho / rho_mean - 1, so the field
        averages to zero and is bounded below by -1. This is the form the
        finders and pysif.measure expect.

        Calling it twice is refused: the second pass would renormalise by a
        mean that is now zero.
        """

@final
class SizeFunction:
    """A void size function: number density of voids per radius bin.

    Returned by pysif.measure.size_function_catalogue() and by the
    models in pysif.model. One container serves both, so a
    measurement and a prediction can be compared directly -- but
    counts and err are meaningful only for a measurement, and are
    zero for a model.

    Not constructed directly.
    """

    def __init__(self) -> None: ...
    @property
    def n_bins(self) -> int:
        """Number of radial bins"""
    @property
    def options(self) -> int:
        """Bitmask options used for VSF"""
    @property
    def r_min(self) -> float:
        """Min radius"""
    @property
    def r_max(self) -> float:
        """Max radius"""
    @property
    def r_edges(self) -> _RealArray:
        """1D array of bin edges"""
    @property
    def r_centres(self) -> _RealArray:
        """1D array of bin centres"""
    @property
    def counts(self) -> NDArray[np.uint64]:
        """1D array of counts"""
    @property
    def vsf(self) -> _RealArray:
        """1D array of VSF values"""
    @property
    def err(self) -> _RealArray:
        """1D array of Poisson statistical errors"""

@final
class Tessellation:
    """Voronoi cell volumes, measured by sampling rather than constructed.

    Every point of space belongs to the tracer nearest it, so throwing
    uniform random points into the box and asking each one which tracer
    owns it measures the Voronoi partition directly: the fraction of the
    samples a tracer catches is the fraction of the box its cell occupies.
    No cell is ever built, the estimate is unbiased, and the total is
    conserved exactly since every sample lands in one cell.

    This buys two things. volumes is a density per tracer that adapts to
    the local sampling instead of to a fixed grid, which is what a sparse
    field wants. mesh() is the more useful one: each sample carries a
    share of its owner's weight, so binning the samples inside a sphere
    counts the fraction of every cell the sphere covers -- the geometric
    intersection that building the cells would otherwise be needed for.

    For fields too sparse to profile by counting tracers. The sample set
    is samples_per_tracer times the field, in memory and in the cost of
    everything measured from it, and a field dense enough to count has
    nothing to gain here.

    Args:
        mesh: ChainMesh of the tracers. Weights and velocities ride along
            into the samples if it carries them. drop_indices=True is
            fine; nothing here names a tracer in field order.
        samples_per_tracer: Samples thrown per tracer, which sets both the
            accuracy and the size of the result. A cell's volume is
            measured to about 1/sqrt(this).
        seed: Any integer. The samples do not depend on the thread count.
        use_pbc: Whether a sample near a face may be owned by a tracer
            across it.
    """

    def __init__(self, mesh: ChainMesh, samples_per_tracer: int = 32,
                 seed: int = 0, use_pbc: bool = True) -> None: ...
    @property
    def n_tracers(self) -> int:
        """Tracers of the mesh this was built from"""
    @property
    def n_samples(self) -> int:
        """Samples thrown"""
    @property
    def n_empty(self) -> int:
        """Tracers no sample landed in. Non-zero means samples_per_tracer is too low: those cells read back with zero volume and their weight is missing from the samples."""
    @property
    def box_length(self) -> float:
        """Box length"""
    @property
    def sample_volume(self) -> float:
        """Volume one sample stands for"""
    @property
    def volumes(self) -> _RealArray:
        """Cell volume per tracer, in the mesh's order rather than the field's"""
    @property
    def consumed(self) -> bool:
        """True once mesh(consume=True) has taken the samples"""
    def mesh(self, n_cells: int, drop_indices: bool = True,
             consume: bool = False) -> ChainMesh:
        """Bin the samples into a ChainMesh, ready to measure.

        What turns a tessellation into a profile: hand the result to
        measure.profiles() and the densities come out volume-weighted, since
        a sample stands for a piece of space rather than for a tracer.

        Args:
            n_cells: Cells per side. Size it from n_samples, not from the
                tracer count -- it is the samples being binned. See
                measure.profiles_suggest_mesh_cells().
            drop_indices: Release the index map; nothing downstream of here
                needs to name a sample.
            consume: Move the samples into the mesh instead of copying them.
                The sample set is the largest thing in the process by a factor
                of the sampling rate, so copying it means holding two. The
                volumes are unaffected, but the samples cannot be handed out
                again; consumed reads back as True.

        Returns:
            ChainMesh: The samples, binned.
        """
