#!/bin/bash

source scripts/gitlab/setup-env.sh

export CTEST_OUTPUT_ON_FAILURE=1
# WITH_CUDA is defined in the per machine job yml.

cleanup() {
  if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
  fi
  rm -rf ci-venv
  rm -rf build
}

build_and_test() {
  WITH_TORCH=${1}
  WITH_FAISS=${2}
  WITH_HDF5=${3}
  WITH_MPI=${4}

  echo "*******************************************************************************************"
  echo "Build configuration" \
    "WITH_TORCH ${WITH_TORCH}" \
    "WITH_FAISS ${WITH_FAISS}" \
    "WITH_HDF5 ${WITH_HDF5}" \
    "WITH_MPI ${WITH_MPI}" \
    "WITH_CUDA ${WITH_CUDA}"
  echo "*******************************************************************************************"

  mkdir -p /tmp/ams
  pushd /tmp/ams

  cleanup

  python -m venv ci-venv
  source ci-venv/bin/activate
  mkdir build
  pushd build

  cmake \
    -DBUILD_SHARED_LIBS=On \
    -DCMAKE_PREFIX_PATH=$INSTALL_DIR \
    -DWITH_CALIPER=On \
    -DWITH_HDF5=${WITH_HDF5} \
    -DWITH_EXAMPLES=On \
    -DAMS_HDF5_DIR=$AMS_HDF5_PATH \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH=$AMS_CUDA_ARCH \
    -DWITH_CUDA=${WITH_CUDA} \
    -DUMPIRE_DIR=$AMS_UMPIRE_PATH \
    -DMFEM_DIR=$AMS_MFEM_PATH \
    -DWITH_FAISS=${WITH_FAISS} \
    -DWITH_MPI=${WITH_MPI} \
    -DWITH_TORCH=${WITH_TORCH} \
    -DWITH_TESTS=On \
    -DTorch_DIR=$AMS_TORCH_PATH \
    -DFAISS_DIR=$AMS_FAISS_PATH \
    -DWITH_AMS_DEBUG=On \
    -DWITH_WORKFLOW=On \
    -DWITH_ADIAK=On \
    ${CI_PROJECT_DIR} || { echo "CMake failed"; exit 1; }

  make -j || { echo "Building failed"; exit 1; }
  make test || { echo "Tests failed"; exit 1; }
  popd

  cleanup

  popd
  rm -rf /tmp/ams
}

# build_and_test WITH_TORCH WITH_FAISS WITH_HDF5 WITH_MPI
build_and_test "On" "On" "On" "On"
build_and_test "On" "On" "On" "Off"
build_and_test "Off" "On" "On" "On"
build_and_test "Off" "Off" "On" "On"
build_and_test "Off" "Off" "Off" "On"

