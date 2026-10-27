# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The types of pysif.measure; see __init__.pyi.

"""Measurements taken from data.

Size functions and profiles from a Catalogue, and the statistics
of the density field itself -- its one-point PDF and its spectral
moments -- from a Grid. The counterparts that predict the same
quantities from theory are in pysif.model, and deliberately
return the same containers so the two can be compared.
"""

from collections.abc import Sequence
from typing import Literal, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import (Catalogue, ChainMesh, Field, Grid, SizeFunction, _RealArray,
               _Vec3)

_Shuffle = Literal["none", "phases", "gaussian"]
_Window = Literal["top_hat", "gaussian"]
_Deconvolution = Literal["full", "aliasing", "none"]

def size_function_catalogue(catalogue: Catalogue, box_length: float,
                            n_bins: int, bins: Literal["ln", "linear"] = "ln",
                            r_min: float = 0.0,
                            r_max: float = 0.0) -> SizeFunction:
    """Bin a catalogue into a void size function.

    Divides the counts by the bin width and the box volume, so the result
    is a number density and catalogues from different volumes are
    comparable. The raw counts are kept, and the Poisson error follows
    from them.

    Args:
        catalogue: Catalogue to bin.
        box_length: Physical side length, which sets the volume.
        n_bins: Radial bins.
        bins: 'ln' for bins uniform in ln R, 'linear' for uniform in R.
        r_min, r_max: Radius bounds; 0 or less takes the catalogue's own
            extremes.

    Returns:
        SizeFunction: The binned size function.
    """

def size_function_combine(
        vsfs: Sequence[SizeFunction], master_bins: int,
        domains: Sequence[tuple[float, float]] | None = None,
        method: Literal["mean", "median", "stitch"] = "mean",
        bin: Literal["ln", "linear"] = "ln") -> SizeFunction:
    """Merge several size functions onto one radius grid.

    How measurements from boxes of different resolution are joined: each
    resolves a different range of radii well, and domains says where each
    should be trusted.

    Args:
        vsfs: Sequence of SizeFunction objects. They must share a binning
            convention -- merging an 'ln' with a 'linear' one combines two
            different quantities.
        master_bins: Radial bins in the result.
        domains: Optional (r_min, r_max) per input; None uses each input's
            full range.
        method: 'mean', 'median' or 'stitch'.
        bin: 'ln' or 'linear', the result's binning convention: the inputs'
            own, as size_function_catalogue() made them.

    Returns:
        SizeFunction: The merged size function.
    """

def survey_volume(randoms: Field, observer: _Vec3 | None = None) -> float:
    """The comoving volume of a survey's footprint, from its randoms.

    The footprint is taken to be separable -- one angular mask, seen out
    to the same range of distances in every direction -- which is how
    survey randoms are made. The volume is then
    Omega * (r_max**3 - r_min**3) / 3, with r_min and r_max the randoms'
    nearest and furthest distances and Omega the mask's solid angle,
    measured from the randoms' directions with the edges and holes
    resolved statistically. Their density on the sky need not be uniform:
    randoms drawn with an angular completeness are measured as they are.
    Unbiased against shapes of known volume to within about its scatter:
    about 0.5% at 10**4 randoms, 0.05-0.1% at 10**5 and 0.015% at 10**6.

    A catalogue whose radial range changes across the sky -- regions with
    different redshift cuts -- breaks the assumption; that is reported in
    a warning, and such regions are measured one call each. Weights are
    ignored: this is the geometric volume.

    Args:
        randoms: Field of the randoms, in comoving Cartesian coordinates.
        observer: Where the observer is, in the same frame. Default: the
            origin, where Field's sky conversion puts it. Pass it if the
            randoms have been moved since.

    Returns:
        float: The volume, in the cube of the positions' unit.

    Raises:
        ValueError: For sky coordinates, a position that is not finite,
            or too few randoms to resolve the mask.
    """

def profiles(catalogue: Catalogue, mesh: ChainMesh, n_bins: int,
             ext: float = 5.0, compute_velocity: bool = False,
             use_pbc: bool = True, differential: bool = False,
             weighted_velocity: bool = False) -> Profiles:
    """Stack radial density and velocity profiles around voids.

    Radii are scaled by each void's own radius, so profiles of
    different-sized voids stack directly. Densities are normalised to the
    box mean, so a profile approaches 1 far from the centre. On a mesh
    whose field carried weights the densities are weighted ones.

    Args:
        catalogue: Voids to profile.
        mesh: ChainMesh of the tracers, which also supplies the box length
            and the mean density -- so it has to hold the whole sample, not
            a subset. Size it with profiles_suggest_mesh_cells(), or reuse
            the mesh a finder was given: any resolution gives the same
            profiles. Must carry velocities if compute_velocity.
        n_bins: Radial bins per profile.
        ext: Outer edge of the profile, in units of each void's radius.
            Anything not positive selects the default of 5.
        compute_velocity: Also stack the radial velocity profile.
        use_pbc: Treat the box as periodic.
        differential: Make each density bin the contrast of its own shell
            rather than of everything enclosed within its outer edge. The
            one worth measuring from a Tessellation, since a shell is all
            boundary. Velocity bins are shell means either way.
        weighted_velocity: Average each shell's radial velocity over the
            tracer weights, sum(w v) / sum(w), rather than over the tracers.
            Only differs on a weighted mesh. Off by default because of what
            it means on a Tessellation mesh: the plain mean over samples is
            the volume-weighted velocity, the weighted one is not.

    Returns:
        Profiles: The stacked profiles.
    """

def profiles_suggest_mesh_cells(n_particles: int) -> int:
    """Suggest a ChainMesh resolution for profiles().

    Resolution changes only speed and memory -- the profiles are identical
    at any of them -- so a mesh built for something else is always a valid
    input, and one sized for finders.exodus() is as good as one sized
    here. It is still worth tuning: the minimum is broad, but sitting a
    long way off it costs a factor of a few.

    Args:
        n_particles: Tracers the mesh will hold. The void size does not
            enter: the balance is between cell overhead and tracers read
            outside the sphere, and it lands in the same place whatever
            the voids measure.

    Returns:
        int: n_cells to pass to ChainMesh.
    """

def delta_distribution_grid(grid: Grid, radii: ArrayLike, n_bins: int,
                            delta_min: float, delta_max: float,
                            shuffle: _Shuffle = "none",
                            window: _Window = "top_hat", seed: int = 0,
                            keep_cic_window: bool = False, *,
                            deconvolution: _Deconvolution = "full"
                            ) -> DeltaDistribution:
    """Measure the one-point PDF of the smoothed density contrast.

    Pass shuffle='phases' or 'gaussian' to measure the same field with its
    Fourier phases randomised: that preserves the power spectrum and
    destroys everything above it, so the difference between the two PDFs
    is the non-Gaussian information the field carries.

    The same seed and grid size give the identical surrogate here and in
    delta_moments_grid(), which is what lets a PDF and a set of moments
    describe one field.

    Requires pysif.system_init().

    Args:
        grid: Density contrast field.
        radii: Smoothing radii, each at least two cells and at most half
            the box. Smaller radii resolve grid artefacts, not field, and
            are rejected.
        n_bins: Histogram bins per radius.
        delta_min, delta_max: Histogram range. Samples outside it are
            counted but not binned, so a row integrates to the fraction
            that fell in range.
        shuffle: 'none', 'phases' or 'gaussian'.
        window: 'top_hat' or 'gaussian'.
        seed: Seed for the phase shuffle.
        keep_cic_window: Skip deconvolving the assignment window: the
            older spelling of deconvolution='none'.
        deconvolution: How the window of the scheme the grid was assigned
            with is undone: 'full', 'aliasing' (allowing for the aliases,
            and for interlacing) or 'none'; see pysif.finders.exodus().

    Returns:
        DeltaDistribution: One PDF per radius.

    Raises:
        RuntimeError: If pysif.system_init() has not been called, or the
            computation failed (the sif log says why).
    """

def delta_moments_grid(grid: Grid, radii: ArrayLike, order: int,
                       shuffle: _Shuffle = "none",
                       window: _Window = "top_hat", n_tracers: int = 0,
                       seed: int = 0, keep_cic_window: bool = False, *,
                       deconvolution: _Deconvolution = "full"
                       ) -> DeltaMoments:
    """Measure the spectral moments of a gridded density field.

    Pass window='gaussian' when order >= 2 has to be meaningful: the
    top-hat sum is cut off by the grid, not by the field.

    Requires pysif.system_init().

    Args:
        grid: Density contrast field.
        radii: Smoothing radii; see delta_distribution_grid().
        order: Highest moment order. Each extra order is nearly free, but
            pass 0 if only sigma_0 is wanted and one array is allocated.
        shuffle: 'none', 'phases' or 'gaussian'.
        window: 'top_hat' or 'gaussian'.
        n_tracers: Tracer count behind the grid, used to subtract the shot
            noise. 0 leaves it in.
        seed: Seed for the phase shuffle; shared with
            delta_distribution_grid().
        keep_cic_window: As in delta_distribution_grid().
        deconvolution: As in delta_distribution_grid().

    Returns:
        DeltaMoments: sigma_0..sigma_order at every radius.

    Raises:
        RuntimeError: If pysif.system_init() has not been called, or the
            computation failed (the sif log says why).
    """

@final
class Profiles:
    """Stacked radial profiles of voids: density and radial velocity.

    Returned by pysif.measure.profiles(). One row per void, binned in
    radius scaled by that void's own radius, so profiles of
    different-sized voids stack directly.

    Not constructed directly.
    """

    def __init__(self) -> None: ...
    @property
    def has_density(self) -> bool:
        """True if density profile exists"""
    @property
    def has_velocity(self) -> bool:
        """True if velocity profile exists"""
    @property
    def n_voids(self) -> int:
        """Number of voids"""
    @property
    def n_bins(self) -> int:
        """Number of radial bins"""
    @property
    def ext(self) -> float:
        """Maximum scaling radius boundary"""
    @property
    def differential(self) -> bool | None:
        """True when a density bin holds its own shell, False when it holds everything enclosed. None if there are no densities."""
    @property
    def r_edges(self) -> _RealArray:
        """1D array of bin edges"""
    @property
    def r_bins(self) -> _RealArray | None:
        """Radius each density bin's value belongs at, in units of the void radius: the outer edge for a cumulative profile, the midpoint for a differential one. Plot against this, not against bin centres -- for a cumulative profile those differ by half a bin, enough to move a feature at r = R_v off that mark. None if there are no densities."""
    @property
    def density(self) -> _RealArray | None:
        """2D array of density profiles"""
    @property
    def velocity(self) -> _RealArray | None:
        """2D array of radial velocity profiles"""


@final
class DeltaDistribution:
    """The one-point PDF of the smoothed density contrast, one row per
    smoothing radius.

    Returned by pysif.measure.delta_distribution_grid(). Every row
    shares one set of bin edges, so rows are comparable across radii.

    Not constructed directly.
    """

    def __init__(self) -> None: ...
    @property
    def n_radii(self) -> int:
        """Number of smoothing radii"""
    @property
    def n_bins(self) -> int:
        """Number of delta bins"""
    @property
    def n_samples(self) -> int:
        """Grid cells sampled per radius"""
    @property
    def radii(self) -> _RealArray:
        """1D array of smoothing radii"""
    @property
    def delta_edges(self) -> _RealArray:
        """1D array of delta bin edges"""
    @property
    def delta_centres(self) -> _RealArray:
        """1D array of delta bin centres"""
    @property
    def distributions(self) -> _RealArray:
        """2D array of delta PDFs, one row per radius"""


@final
class DeltaMoments:
    """Spectral moments sigma_0..sigma_order at several smoothing
    radii.

    Returned by pysif.measure.delta_moments_grid() (measured from a
    field) or pysif.model.delta_moments_pk() (integrated from a model
    spectrum). The orders are held together because they are only
    meaningful together: sigma_0 and sigma_2 describe the same field
    only if they came from the same radii and the same window.

    Ask the producer for order 0 if only sigma_0 is wanted.

    Not constructed directly.
    """

    def __init__(self) -> None: ...
    @property
    def n_radii(self) -> int:
        """Number of smoothing radii"""
    @property
    def order(self) -> int:
        """Highest moment order computed"""
    @property
    def n_moments(self) -> int:
        """order + 1"""
    @property
    def radii(self) -> _RealArray:
        """1D array of smoothing radii"""
    @property
    def sigma(self) -> _RealArray:
        """(n_moments, n_radii) array; sigma[j] is sigma_j across radii"""
    @property
    def high_k_fraction(self) -> _RealArray:
        """(n_moments, n_radii) fraction of each sum from the top half of the available k range; a large value means that moment is resolution- or truncation-limited"""
    @property
    def offsets(self) -> NDArray[np.uint32]:
        """n_moments + 1 offsets into the flat C arrays, one block per order"""
    @property
    def gamma(self) -> _RealArray:
        """BBKS spectral parameter sigma_1^2 / (sigma_0 sigma_2); needs order >= 2"""
    @property
    def r_star(self) -> _RealArray:
        """BBKS coherence scale sqrt(3) sigma_1 / sigma_2; needs order >= 2"""
    def moment(self, order: int, /) -> _RealArray:
        """sigma_order across radii, as a read-only view into the moment set.
        Equivalent to sigma[order], but rejects an order that was not computed rather than indexing past the end.
        """
