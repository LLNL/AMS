# Setup and Build

AMSLib depends on the following packages:
* UMPIRE (Mandatory)
* MPI (Mandatory)
* CALIPER (Optional)
* FAISS (Optional)
* MFEM (Optional)
* PY-TORCH (Optional)
* MFEM (Optional)
* REDIS (Optional)
* HDF5 (Optional)
* CUDA (Optional)

## Spack Installation

AMS depends on multiple complex external libraries, our preferred and suggested mechanism to install AMS is through [spack](https://github.com/spack/spack) as follows:

```bash
spack install ams
```

If you are a developer and would like to extend AMS you can do so by using the `spack dev-build' command.
For more instructions look [here](https://spack-tutorial.readthedocs.io/en/lanl19/tutorial_developer_workflows.html)


## Manual cmake installation

Below you can find a `cmake` command to configure to configure AMS, build and install it.

```bash
$ mkdir build; cd build
$ cmake \
  -DWITH_DB=On -DWITH_RMQ=On \
  -Damqpcpp_DIR=$AMS_AMQPCPP_PATH \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_PREFIX_PATH=$INSTALL_DIR \
  -DWITH_CALIPER=On \
  -DWITH_HDF5=On \
  -DWITH_EXAMPLES=On \
  -DHDF5_Dir=$AMS_HDF5_PATH \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=On \
  -DUMPIRE_DIR=$AMS_UMPIRE_PATH \
  -DMFEM_DIR=$AMS_MFEM_PATH \
  -DWITH_FAISS=On \
  -DWITH_MPI=On \
  -DWITH_TORCH=On \
  -DWITH_TESTS=Off \
  -DTorch_DIR=$AMS_TORCH_PATH \
  -DFAISS_DIR=$AMS_FAISS_PATH \
  -DAMS_CUDA_ARCH=${AMS_CUDA_ARCH} \
  -DWITH_AMS_DEBUG=On \
  ../

$ make -j6
$ make install
```

Most of the compile time options are optional.

## Building AMS with PerfFlowAspect

To built AMS with [PFA](https://github.com/flux-framework/PerfFlowAspect) support you first need to install a PFA clang/llvm version and add it to `$PATH`. Next to configure, built and install perform the following:

```
$ cd $CODE_ROOT/setup
$ source setup_env_with_pfa.sh
$ mkdir build; cd build
$  cmake \
   -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-10.0.1-gcc-8.3.1/bin/clang++ \
   -DCMAKE_C_COMPILER=/usr/tce/packages/clang/clang-10.0.1-gcc-8.3.1/bin/clang \
   -DWITH_HDF5=On -DHDF5_Dir=$AMS_HDF5_PATH \
   -DCMAKE_PREFIX_PATH=$INSTALL_DIR \
   -DWITH_DB=On \
   -DCMAKE_INSTALL_PREFIX=./install \
   -DCMAKE_BUILD_TYPE=Debug \
   -DWITH_EXAMPLES=On \
   -DMFEM_DIR=$AMS_MFEM_PATH \
   -DUMPIRE_DIR=$AMS_UMPIRE_PATH \
   -DWITH_MPI=On \
   -DWITH_CUDA=On \
   -DWITH_CALIPER=On \
   -DWITH_TORCH=On -DTorch_DIR=$AMS_TORCH_PATH \
   -DWITH_FAISS=On -DFAISS_DIR=$AMS_FAISS_PATH \
   -DAMS_CUDA_ARCH=${AMS_CUDA_ARCH} \
   -DWITH_PERFFLOWASPECT=On \
   -Dperfflowaspect_DIR=$AMS_PFA_PATH/share \
  ../
$ make -j6
```

