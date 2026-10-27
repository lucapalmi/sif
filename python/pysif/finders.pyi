# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The types of pysif.finders; see __init__.pyi.

"""Void finders.

spherical() and exodus() take a density-contrast grid of a
periodic box and return a Catalogue. They differ in where the radii
come from: spherical() can only report radii from the ladder you
pass in, while exodus() grows each void to the radius its tracers
actually support, and so also needs the particles as a
ChainMesh.

exodus_survey() is exodus() for a survey, whose geometry comes
from a random catalogue; survey_box() sizes the box to run it
in.
"""

from os import PathLike
from typing import Any, Literal

from numpy.typing import ArrayLike

from . import Catalogue, ChainMesh, Field, Grid, _RealArray, _Vec3

_Path = str | bytes | PathLike[str] | PathLike[bytes]
_Variant = Literal["excursion-set", "excursion-peak"]
_Deconvolution = Literal["full", "aliasing", "none"]
_Seeds = Literal["cells", "minima"]

def exodus(grid: Grid, mesh: ChainMesh, radii: ArrayLike, threshold: float,
           variant: _Variant = "excursion-set", *,
           checkpoint: _Path | None = None, prior: Catalogue | None = None,
           keep_checkpoint: bool = False,
           save_memory: bool = False) -> Catalogue:
    """Find voids, growing each to the radius its tracers support.

    Uses the radius ladder to locate candidates, then walks the enclosed
    density outward from each accepted centre until it crosses the
    threshold. The radii therefore come from the data rather than from the
    ladder, which is why this needs the particles as well as the grid.

    What a void is is the variant: one of two, which fix every setting of
    the finder. exodus_raw() takes those settings one by one, for any
    other combination.

    - 'excursion-set': every cell below the threshold at a rung is a
      candidate, on its grid node, and voids may not overlap -- the finder
      of 0.4, and the voids of the excursion-set models. The void size
      function depends on the grid below about 10-15 cells in radius. The
      default, for now.
    - 'excursion-peak': the local minima of each rung's field, found after
      smoothing it by a Gaussian of 0.15 of the rung and placed between
      the nodes; a void is refused only when its centre falls inside a
      larger one. The void size function is within about 1.5% of a finer
      grid's from about 6 cells in radius; start the ladder about 1.5
      times below the smallest radius you report.

    The mesh is borrowed and never freed, so it can be reused across runs
    at different thresholds. Velocities and original indices are never
    read. Weights are: on a mesh built from a weighted Field, a void's
    radius is where the enclosed weight reaches the threshold against the
    mean weight density -- the same criterion the grid applies when it
    was built from that Field. The weights must be finite and
    non-negative. A Tessellation mesh carries its density only in its
    weights, so it is always run this way.

    Requires pysif.system_init().

    Args:
        grid: Density contrast field. Smoothed in place and restored
            afterwards.
        mesh: ChainMesh over the same box.
        radii: Smoothing radii to try.
        threshold: Density contrast a cell must reach to seed a void, in
            (-1, 0).
        variant: Which finder, from the two above.
        checkpoint: Where the run saves its progress after every rung: a
            path, or None (the default) for exodus-<hex>.ckpt in the
            working directory, logged as the run starts. A path that exists
            is resumed -- the run picks up after the last rung saved and
            returns what it would have uninterrupted, bit for bit -- once
            its header is found to describe this run (ladder, threshold,
            the variant's settings, grid and its assignment, tracer counts
            and weights); otherwise ValueError, naming what differs.
            Removed when the run completes; kept, with its name logged,
            when it fails after a rung. A variant always checkpoints;
            exodus_raw() takes False for none.
        prior: A Catalogue of voids to start from, or None. They are
            returned first, unchanged, and what the run adds respects the
            overlap rule against them. Nothing ties them to this run: to
            extend a catalogue to smaller radii, pass it with the rungs
            below the ones it was found with, which gives one run's
            catalogue while consecutive rungs are no further apart than
            the search factor.
        keep_checkpoint: Leave the checkpoint in place when the run
            completes, for the caller to remove once the catalogue is
            safely written.
        save_memory: Hold about half the memory, at the cost of time. The
            grid goes to a scratch file in the scratch_directory setting
            (see set_setting(); SIF_SCRATCH_DIRECTORY for one job) and
            is read back and transformed again at every smoothing, instead
            of its transform being kept in memory. The catalogue and the
            grid handed back are bit for bit the same. The file is as
            large as the grid and goes when the run does.

    Returns:
        Catalogue: The voids found, the prior's first.

    Raises:
        RuntimeError: If pysif.system_init() has not been called.
        ValueError: For an argument out of range, a mesh and grid over
            different boxes, or a search sphere wider than the mesh.
        MemoryError: If the run runs out of memory. The grid is restored
            either way.
        OSError: If save_memory's scratch file cannot be made, written or
            read: a directory without the room, say.
    """

def exodus_raw(grid: Grid, mesh: ChainMesh, radii: ArrayLike,
               threshold: float, overlap_fraction: float = 0.0, *,
               seeds: _Seeds = "cells", search_factor: float = 1.5,
               deconvolution: _Deconvolution = "full",
               consume_grid: bool = False,
               checkpoint: _Path | Literal[False] | None = None,
               prior: Catalogue | None = None, keep_checkpoint: bool = False,
               save_memory: bool = False) -> Catalogue:
    """exodus(), with every setting a variant fixes taken one by one.

    The variants are two points of this: excursion-set is
    overlap_fraction=0 with seeds='cells', excursion-peak is
    overlap_fraction=1 with seeds='minima'. Everything else at its
    default.

    With seeds='minima' the finder measures a crossing past a rung's reach
    by extending the reach, and lets a void displace the smaller ones it
    overlaps, since a void that grew past its rung can be larger than voids
    accepted before it. seeds='cells' is exactly the finder of 0.4.

    Args:
        grid, mesh, radii, threshold: As in exodus().
        overlap_fraction: How much two voids may overlap, as a fraction of
            the smaller one's radius, in [0, 1]. 0 forbids overlap
            entirely; 1 refuses a void only when its centre is inside
            another.
        seeds: Where candidates come from. 'cells': every cell below the
            threshold at a rung, on its node, deepest first. 'minima': the
            local minima of the rung's field smoothed by a further Gaussian
            of 0.15 of the rung, placed between the nodes; periodic boxes
            only, for now.
        search_factor: How far past a rung the rescaling looks for the
            crossing, as a multiple of the rung. Default 1.5. This is the
            finder's dominant cost and it goes as factor**3 - 1, so 1.5
            does about a third the work of 2.0. Snapped to the nearest of
            1.25, 1.5, 1.75, 2.0; pass 2.0 to reproduce catalogues made
            before this argument existed.
        deconvolution: How the grid's assignment window -- the scheme
            Grid.assign() recorded, CIC for assign_cic() -- is undone
            before smoothing. 'full' divides it out; it amplifies the
            modes near the grid's Nyquist frequency, up to 15x in the
            corner for CIC, folded-in aliases and shot noise with them.
            'aliasing' divides by the window and the aliases together (the
            least-squares correction), which gains 1.8x at most for CIC
            and allows for interlacing. 'none' keeps the window.
        consume_grid: Skip restoring the grid, saving one inverse FFT.
        checkpoint: As in exodus(), or False for none.
        prior, keep_checkpoint, save_memory: As in exodus().

    Returns:
        Catalogue: The voids found, the prior's first.

    Raises:
        As exodus().
    """

def exodus_survey(data_grid: Grid, random_grid: Grid, data_mesh: ChainMesh,
                  random_mesh: ChainMesh, offset: _Vec3, radii: ArrayLike,
                  threshold: float,
                  variant: Literal["excursion-set"] = "excursion-set", *,
                  checkpoint: _Path | None = None,
                  prior: Catalogue | None = None,
                  keep_checkpoint: bool = False,
                  save_memory: bool = False) -> Catalogue:
    """Find voids in a survey: any geometry, described by a random catalogue.

    The same finder as exodus(), with the box mean replaced by the randoms.
    Only the excursion-set variant runs on a survey, for now.
    A sphere's expected content is the random weight inside it times
    alpha = W_data / W_random, so its contrast is D(<r) / (alpha R(<r)) - 1:
    the footprint, its holes and the radial selection all come in through
    the randoms. They may follow the survey's n(z), and both sets may carry
    weights.

    Where the randoms are is the footprint: the grid cells holding at least
    one random, with the holes a sparse catalogue leaves by chance closed
    over 3^3 cells (up to 9^3 for very sparse randoms), which also fills
    real gaps a couple of cells wide. Voids are only centred in
    observed cells, and every void found is kept, with the fraction of its
    sphere and of the shell out to twice its radius that was observed --
    Catalogue.footprint and Catalogue.footprint_shell.

    Coordinates are comoving Cartesian, converted beforehand (astropy does
    it). Nothing wraps around, so the survey needs empty padding inside the
    box, about the largest search sphere on every side: survey_box() works
    out the box and the offset, and Field.translate() moves data and
    randoms in. Pass the same offset here, and the voids come back out in
    your own frame.

    Requires pysif.system_init().

    Args:
        data_grid: Grid of the data after assign_cic(). A density, NOT a
            density contrast. Smoothed in place and restored afterwards.
        random_grid: Grid of the randoms, same cells and box. Restored the
            same way.
        data_mesh: ChainMesh of the data over the same box.
        random_mesh: ChainMesh of the randoms over the same box; its cells
            need not match the data mesh's.
        offset: What survey_box() moved the survey into the box by, as 3
            numbers; (0, 0, 0) for a survey that was not moved. The voids
            are moved back by it.
        radii: Smoothing radii to try.
        threshold: Density contrast a void is grown to, against the
            randoms, in (-1, 0).
        variant: As in exodus(); 'excursion-set' only, for now.
        checkpoint: As in exodus(); the offset is part of what a resume
            checks.
        prior: As in exodus(), in your own frame (Cartesian: to_cartesian()
            takes one off the sky). The box survey_box() sizes follows the
            ladder's ends, so extending to smaller radii matches one run
            only if both are searched in the box sized for the whole
            ladder.
        keep_checkpoint, save_memory: As in exodus().

    Returns:
        Catalogue: The voids found, with their footprint, in your own
        frame.

    Raises:
        RuntimeError: If pysif.system_init() has not been called.
        ValueError: If the finder refused its inputs -- most often a
            survey too close to the box faces; the message says how much
            padding it needs, and survey_box() works it out.
        MemoryError: As in exodus().
    """

def exodus_survey_raw(data_grid: Grid, random_grid: Grid,
                      data_mesh: ChainMesh, random_mesh: ChainMesh,
                      offset: _Vec3, radii: ArrayLike, threshold: float,
                      overlap_fraction: float = 0.0, *,
                      seeds: _Seeds = "cells", search_factor: float = 1.5,
                      deconvolution: _Deconvolution = "full",
                      consume_grid: bool = False,
                      checkpoint: _Path | Literal[False] | None = None,
                      prior: Catalogue | None = None,
                      keep_checkpoint: bool = False,
                      save_memory: bool = False) -> Catalogue:
    """exodus_survey(), with every setting taken one by one, as exodus_raw()
    takes them for a box. Only seeds='cells' runs on a survey, for now.

    Args:
        data_grid, random_grid, data_mesh, random_mesh: As in
            exodus_survey().
        offset, radii, threshold: As in exodus_survey().
        overlap_fraction, seeds, deconvolution: As in exodus_raw().
        search_factor: As in exodus_raw(). Pass the same one to
            survey_box().
        consume_grid: Skip restoring both grids.
        checkpoint: As in exodus_survey(), or False for none.
        prior, keep_checkpoint, save_memory: As in exodus_survey().

    Returns:
        Catalogue: As exodus_survey().

    Raises:
        As exodus_survey().
    """

def read_checkpoint(path: _Path) -> tuple[Catalogue, dict[str, Any]]:
    """Read an exodus checkpoint: the voids so far, and the run they belong
    to.

    For a look at a run in progress or one that stopped; resuming it is
    exodus() given the same checkpoint=.

        voids, info = pysif.finders.read_checkpoint("run.ckpt")
        print(info["rungs_done"], "of", len(info["radii"]))

    Args:
        path: The checkpoint, str or os.PathLike.

    Returns:
        tuple: (Catalogue, dict). The voids are in your own frame, a
        survey's moved back out of its box. The dict holds mode ('box'
        or 'survey'), weighted, radii (the ladder, largest first),
        rungs_done, n_voids, n_prior, threshold, overlap_fraction,
        search_factor, keep_cic_window, deconvolution, seeds ('cells'
        or 'minima'), grid_n_cells, box_length, assignment, interlaced,
        n_particles, total_weight, n_randoms, random_total_weight and
        offset.

    Raises:
        OSError: If the file cannot be read, or is not a checkpoint, is
            damaged, or was written in another byte order or precision.
    """

def survey_box(data: Field, randoms: Field, radii: ArrayLike,
               search_factor: float = 1.5) -> tuple[_RealArray, float]:
    """The box a survey has to be searched in, and the offset that moves it
    there.

    exodus_survey() needs empty padding around the survey, and how much
    depends on the radii and the search factor. This works it out from the
    tracers: a cubic box, just large enough, with the survey centred in it.
    Both catalogues are needed: the finder checks the padding around both,
    and the outermost galaxies can sit a little past the outermost
    randoms.

    The box comes before the grid, so it is sized for any grid whose cells
    are no larger than the smallest radius -- every grid that resolves it
    at all. That costs some extra padding: the smallest radius a side for
    a ladder spanning a factor of 6 or more, up to four times it for a
    much shorter one. exodus_survey() refuses a coarser grid and says
    why.

        offset, box = pysif.finders.survey_box(data, randoms, radii)
        data.translate(offset); randoms.translate(offset)
        # Grid(n, box), ChainMesh(..., box, ...), then
        # exodus_survey(..., offset, ...): the voids come back in your frame

    Args:
        data: Field of the data, in your own Cartesian frame.
        randoms: Field of the randoms, in the same frame.
        radii: The radii exodus_survey() will be run with.
        search_factor: The one exodus_survey() will be run with.

    Returns:
        tuple: (offset, box_length). offset is a length-3 array to add to
        every position, data and randoms alike.

    Raises:
        ValueError: For a catalogue with no positions or in sky
            coordinates, a coordinate that is not finite, or a radius that
            is not positive.
    """

def spherical(grid: Grid, radii: ArrayLike, threshold: float,
              overlap_fraction: float = 0.0, consume_grid: bool = False, *,
              deconvolution: _Deconvolution = "full") -> Catalogue:
    """Find voids as fixed-radius spheres on a density grid.

    Walks the radii from largest to smallest, accepting a sphere wherever
    one does not overlap a void already accepted. Voids therefore come out
    with radii drawn from the ladder you passed in; use exodus() to get
    radii from the data.

    Requires pysif.system_init().

    Args:
        grid: Density contrast field. Smoothed in place and restored
            afterwards unless consume_grid is set.
        radii: Smoothing radii to try. Order does not matter.
        threshold: Density contrast a cell must reach to seed a void, in
            (-1, 0).
        overlap_fraction: How much two voids may overlap, as a fraction of
            the smaller one's radius, in [0, 1].
        consume_grid: Skip restoring the grid.
        deconvolution: As in exodus().

    Returns:
        Catalogue: The voids found.

    Raises:
        RuntimeError: If pysif.system_init() has not been called.
        ValueError: For an argument out of range.
        MemoryError: If the run runs out of memory.
    """

def grid_cells_from_size(n_particles: int, box_length: float,
                         cell_size: float,
                         units: Literal["mps", "physical"] = "mps",
                         volume: float | None = None) -> int:
    """Grid cells per side that make each cell a given size.

    The finder's smallest radius should span two grid cells at least, so
    one cell per mean separation suits a ladder that starts at two
    separations. Rounds up, so the cells are never larger than asked for;
    a size that divides the box to within rounding gives that exact count.
    The result is not snapped to an FFT-friendly size.

        n = pysif.finders.grid_cells_from_size(n_particles, box, 1.0)  # 1 mps
        grid = pysif.Grid(n, box)

    Args:
        n_particles: Tracers in the box; only read for units="mps".
        box_length: Side of the box the grid spans.
        cell_size: Wanted cell side, positive.
        units: "mps" for mean particle separations,
            (volume / n_particles)**(1/3), or "physical" for the box's
            own unit.
        volume: The volume the tracers occupy, which sets the mean
            separation; None for the whole box. In a survey box the
            tracers fill only the footprint: pass
            pysif.measure.survey_volume(randoms), or the separation comes
            out inflated by the padding. Only read for units="mps".

    Returns:
        int: n_cells to pass to Grid.

    Raises:
        ValueError: For a size, box or volume that is not positive and
            finite, unknown units, no tracers with units="mps", or a count
            past 32 bits.
    """

def suggest_mesh_cells(n_particles: int, box_length: float,
                       max_radius: float = 0.0,
                       volume: float | None = None) -> int:
    """Suggest a ChainMesh resolution for exodus() or exodus_survey().

    Resolution changes only speed and memory -- the catalogue is identical
    at any resolution -- so it is safe to tune and worth tuning. The rule
    sizes the cells for the density the tracers are at, which in a survey
    box is the density inside the footprint: pass its volume, and call once
    for each mesh, the data's and the randoms', with that set's own count.

    Args:
        n_particles: Tracers the mesh will hold.
        box_length: Physical side length, matching the grid's.
        max_radius: Largest smoothing radius the run will use. The search
            sphere it implies must fit inside the mesh, which puts a floor
            under the resolution. Pass 0 to skip that constraint.
        volume: The volume the tracers occupy; None for the whole box. For
            a survey, pysif.measure.survey_volume(randoms).

    Returns:
        int: n_cells to pass to ChainMesh.

    Raises:
        ValueError: For no tracers, a box or volume that is not positive
            and finite, or a search sphere wider than the box.
    """
