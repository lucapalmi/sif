# Sif void library

[![tests](https://github.com/lucapalmi/sif/actions/workflows/tests.yml/badge.svg)](https://github.com/lucapalmi/sif/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](COPYING)

**sif** is a high performance library to study cosmic voids, both in 
simulations and surveys.

The library is written in ANSI-C99, with Python bindings available via the 
**pysif** package. The complete documentation is available 
[here](https://lucapalmi.github.io/sif/).

## Main features

These are the main features of **sif**:

- **exodus:** a spherical void finder for N-body simulations and surveys; fast
and reliable. 
- **measuring:** routines to measure the void size function, density and 
velocity profiles for any void catalogue.
- **modelling:** routines to compute the 
[SvdW](https://arxiv.org/abs/astro-ph/0311260), 
[Vdn](https://arxiv.org/abs/1304.6087) and 
[excursion-peak](https://arxiv.org/abs/2401.14451) models for void abundances.

## Project status

**sif** is under active development. There is no regular schedule for the 
releases.

## Installation

You can install the Python package with pip
```
pip install pysif
```
To use the C library, clone and compile the repository. A complete guide to 
the installation is available 
[here](https://lucapalmi.github.io/sif/installation).

## License

Copyright (C) 2026 Luca Palmieri.

sif is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

sif is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

The full license text is in [COPYING](COPYING). Every source file carries an
`SPDX-License-Identifier: GPL-3.0-or-later` tag.

Note that this applies to the Python bindings as well: `pysif` links the same
GPL-licensed code, so anything distributed against it is bound by the same
terms.

## Third-party components

**Robust geometric predicates** — sif uses Jonathan
Richard Shewchuk's implementation of adaptive-precision floating-point
arithmetic and incircle/insphere tests (Carnegie Mellon University,
<https://www.cs.cmu.edu/~quake/robust.html>). The accompanying
`predicates.h` is not part of the original distribution.

**FFTW3** — sif links FFTW 3 (<https://www.fftw.org/>) for its transforms. FFTW
is not bundled; it is located at configure time and must be installed
separately. FFTW is licensed under the GNU General Public License, version 2 or
later.

**OpenMP** — sif is parallelized with OpenMP and links the runtime provided by
the compiler (LLVM's `libomp` or GCC's `libgomp`) under its
own license.

**HDF5** — Optionally, sif uses the HDF5 library (<https://www.hdfgroup.org/solutions/hdf5/>)
for binary I/O. HDF5 is not bundled; it is located at configure time and must be installed
separately. HDF5 is licensed under the 3-clause BSD License.

## AI policy

All the functions and algorithms in **sif** have been designed by humans. 
Implementations, refactoring and bug-hunting are mostly done with coding agents. 
The code comments and the Doxygen API documentation are mostly written by 
agents; the documentation, except for the API reference, is completely 
handwritten.
