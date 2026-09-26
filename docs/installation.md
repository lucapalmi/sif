# Installation

:::::{tab-set}

::::{tab-item} Stable release
<br>
You can install the latest stable release with pip

```shell
pip install pysif
```
In this case, the library is configured with HDF5 support.

:::: 

::::{tab-item} Development
<br>
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

- **SIF_HDF5_SUPPORT:** activates the HDF5 integration. By default, HDF5 is used only if it's available as a CMake package and is a serial build (no MPI). Options **(AUTO|ON|OFF)**, default **AUTO**.

- **SIF_NATIVE_ARCH:** activates the `-march=native` compile flag in Release builds. If the compiled binary is going to be used on machines with different architectures, this option should be turned off. Options **(ON|OFF)**, default **ON**.

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

:::{note}
The project's compile options are set to their default values. If you need to modify them, the standard syntax is
```
pip install . -C cmake.define.OpenMP_ROOT=$(brew --prefix libomp)
```

:::

:::{admonition} HPC cluster usage
:class: tip
A build from source records where it found FFTW, HDF5 and the OpenMP runtime, so `import pysif` works without the modules loaded and without setting `LD_LIBRARY_PATH`. The modules only have to be loaded while `pip install .` runs. If the modules are later upgraded or removed, rebuild. This is the **SIF_PYTHON_RPATH** option, **ON** by default; turn it off (`-C cmake.define.SIF_PYTHON_RPATH=OFF`) only when building a wheel to distribute.
:::

::::

:::::

