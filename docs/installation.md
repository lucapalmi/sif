# Installation

:::::{tab-set}

::::{tab-item} Stable release
You can install the latest stable release with pip

```shell
pip install pysif
```
In this case, the library is configured with HDF5 and FITS support.

:::: 

::::{tab-item} Development
First, clone the repository and move into it

```shell
git clone https://github.com/lucapalmi/sif.git
cd sif
```

:::{note}
The library requires OpenMP. If you are on macOS, make sure to use a compiler that supports it (e.g. LLVM-Clang or GCC) or install the libomp runtime (`brew install libomp`)
:::

# Dependencies

The library has few, non-vendored dependencies that should be available on your system

1. `cmake`
2. `fftw3`
3. `hdf5` (optional)
4. `cfitsio` (optional)

You can install them with any package manager,
```shell
# macOS
brew install cmake fftw hdf5

# Ubuntu / Debian
sudo apt install build-essential cmake libfftw3-dev libhdf5-dev
```

:::{admonition} HPC cluster usage
:class: tip
If you are on an HPC cluster, make sure to load the module files for the dependencies and the compiler.
:::

# C library

The library is compiled through CMake. These are the available options

- **SIF_FFTW_THREADING:** set the FFTW multi-threading backend to use, either the OpenMP runtime or the pthreads runtime. By default, the backend is set to OpenMP. If you use LLVM-clang (`brew install llvm`) on macOS, the FFTW OpenMP runtime may cause crashes and/or wrong results (unless properly configured); in this scenario, it is suggested to use the pthreads runtime with `-DSIF_FFTW_THREADING=threads`. Options **(omp|threads)**, default **omp**.

- **SIF_HDF5_SUPPORT:** activates the [HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html) integration. By default, HDF5 is used only if it's available as a CMake package and is a serial build (no MPI). Options **(AUTO|ON|OFF)**, default **AUTO**.

- **SIF_FITS_SUPPORT:** activates the FITS format integration through [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/). It is found through pkg-config when available; if it is installed somewhere pkg-config does not know about (common on clusters), pass its install prefix with `-DCFITSIO_ROOT=/path/to/cfitsio` or set the `CFITSIO_ROOT` environment variable. Options **(AUTO|ON|OFF)**, default **AUTO**.

- **SIF_NATIVE_ARCH:** activates the `-march=native` compile flag in Release builds. If the compiled binary is going to be used on machines with different architectures, this option should be turned off. Options **(ON|OFF)**, default **ON**.

- **SIF_BUILD_PROGRAMS:** builds the command-line programs (`sif-exodus`) into `build/bin/`; `cmake --install` puts them in the `bin/` directory of the install prefix. Options **(ON|OFF)**, default **ON**.

- **SIF_SANITIZE:** builds everything with the given sanitizers, passed to `-fsanitize=` (e.g. `thread`, or `address,undefined`); for development, not for production runs. ThreadSanitizer needs Clang with LLVM's OpenMP runtime, and, when the tests run, `OMP_TOOL_LIBRARIES` set to LLVM's `libarcher` and `TSAN_OPTIONS=ignore_noninstrumented_modules=1`. Default empty (off).

:::{admonition} HPC cluster usage
:class: tip
If you are on an HPC cluster, check that the login nodes have the same architecture of the compute nodes. If not, compile with `-DSIF_NATIVE_ARCH=OFF`.
:::

:::{note} 
If you are using AppleClang with libomp, make sure to use the option `-DOpenMP_ROOT=$(brew --prefix libomp)` when configuring the project.
:::

To configure the CMake, run
```shell
cmake -B build -DSIF_HDF5_SUPPORT=ON
```
Then, compile and install with
```shell
cmake --build build
cmake --install build
```

# Python package

The python package can be installed with pip. 
```
pip install .
```
The project's compile options are set to their default values. If you need to modify them, the standard syntax is
```
pip install . -C cmake.define.OpenMP_ROOT=$(brew --prefix libomp)
```

:::{admonition} HPC cluster usage
:class: tip
A build from source records where it found FFTW, HDF5 and the OpenMP runtime, so `import pysif` and the command-line programs work without having to load the modules. They only have to be loaded while `pip install .` runs. If the modules are later upgraded or removed, rebuild.
:::

The package carries type stubs, so editors (VS Code, PyCharm, or any editor with a Python language server) complete pysif's functions, their parameters and docstrings, and type checkers such as mypy and pyright check calls into it. Nothing needs installing or configuring for this.

::::

:::::
