#!/bin/bash
echo ${CI_PROJECT_DIR}

source scripts/gitlab/setup-env.sh

export CTEST_OUTPUT_ON_FAILURE=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

cleanup() {
  if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
  fi
  rm -rf venv-*
  rm -rf build
}

build_and_test() {

  echo "*******************************************************************************************"
  echo "Build configuration" \
    "WITH_MPI=${WITH_MPI}" \
    "WITH_WORKFLOW=${WITH_WORKFLOW}" \
    "WITH_CUDA=${WITH_CUDA}" \
    "WITH_HIP=${WITH_HIP}" \
    "WITH_TORCH ${WITH_TORCH}"
  echo "*******************************************************************************************"

  build_dir="/tmp/ams/$(uuidgen)"
  mkdir -p ${build_dir}
  pushd ${build_dir}

  cleanup

  # We need custom Virtual env on LC machines because we use Spack python external
  export host=$(hostname)
  export host=${host//[0-9]/}

  venv_args=(
    --env /usr/workspace/AMS/ams-spack-environments/1.1/${host}/
    --with-system-flux-python
    --output venv-${host}
  )

  python3 ${CI_PROJECT_DIR}/scripts/make-spack-venv.py "${venv_args[@]}"
  source venv-${host}/bin/activate

  if [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
    C_COMPILER=amdclang
    CXX_COMPILER=amdclang++
  elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib" ]]; then
    C_COMPILER=gcc
    CXX_COMPILER=g++
  fi

  if [[ "${WITH_WORKFLOW}" == "On" ]]; then
    AMS_WORKFLOW_PIP_ARGS="--no-build-isolation"
  else
    AMS_WORKFLOW_PIP_ARGS=""
  fi

  mkdir build
  pushd build

  cmake \
    -DBUILD_SHARED_LIBS=On \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CALIPER=On \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DENABLE_RMQ=Off \
    -DENABLE_WORKFLOW=${WITH_WORKFLOW} \
    -DAMS_PIP_INSTALL_ARGS="${AMS_WORKFLOW_PIP_ARGS}" \
    -DENABLE_TESTS=On \
    -DCUDA_ARCH=${AMS_CUDA_ARCH} \
    -DENABLE_CUDA=${WITH_CUDA} \
    -DENABLE_HIP=${WITH_HIP} \
    -DENABLE_MPI=${WITH_MPI} \
    -DAMS_ENABLE_DEBUG=On \
    -DWITH_TORCH=${WITH_TORCH} \
    -DTorch_DIR="$AMS_TORCH_PATH" \
    -DZLIB_DIR="$AMS_ZLIB_PATH" \
    -Dcaliper_DIR="$AMS_CALIPER_PATH" \
    -DAMS_FMT_DIR="$AMS_FMT_DIR" \
    -DHDF5_DIR="$AMS_HDF5_PATH" \
    -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
    -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR" \
    -DAMS_CATCH2_DIR="$AMS_CATCH2_DIR" \
    -Damqpcpp_DIR="$AMS_AMQPCPP_PATH" \
    -DCMAKE_C_COMPILER="$C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
    ${CI_PROJECT_DIR} || { echo "CMake failed"; exit 1; }

  make -j || { echo "Building failed"; exit 1; }
  make test || { echo "Tests failed"; exit 1; }
  popd

  cleanup

  popd
  rm -rf ${build_dir}
}

build_and_test
