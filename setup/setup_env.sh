#!/usr/bin/env bash

## load relevant modules on lassen
module load gcc/8.3.1
module load cmake/3.23.1
host=$(hostname)
host=${host//[0-9]/}

SPACK_ENV_PATH="0.19"

if [[ "$SYS_TYPE" == "blueos_3_ppc64le_ib_p9" ]]; then
  echo "Loading mpi and cuda"
  if [[ "${SPACK_ENV_PATH}" == "0.19" ]]; then
    module load cuda/11.6.1
  elif [[ "${SPACK_ENV_PATH}" == "0.18" ]]; then
    module load cuda/11.4.1
  fi
  module load spectrum-mpi
  CUDA_ARCH=70
elif [[ "$SYS_TYPE" == "toss_3_x86_64_ib" ]]; then
  module load mvapich2/2.3
  if [[ "${SPACK_ENV_PATH}" == "0.19" ]]; then
    module load cuda/11.6.1
  elif [[ "${SPACK_ENV_PATH}" == "0.18" ]]; then
    module load cuda/11.4.1
  fi
  CUDA_ARCH=60
fi

## activate spack
source /usr/workspace/AMS/ams-spack-environments/${SPACK_ENV_PATH}/spack/share/spack/setup-env.sh

## activate the spack environment
spack env activate /usr/workspace/AMS/ams-spack-environments/${SPACK_ENV_PATH}/$host

## export the paths (currently cmake needs these)
export AMS_MFEM_PATH=`spack location -i mfem`
export AMS_TORCH_PATH=`spack location -i py-torch`
export AMS_FAISS_PATH=`spack location -i faiss`
export AMS_UMPIRE_PATH=`spack location -i umpire`
export AMS_HIREDIS_PATH=`spack location -i hiredis`
export AMS_REDIS_PLUS_PLUS_PATH=`spack location -i redis-plus-plus`
export AMS_HDF5_PATH=`spack location -i hdf5`
export AMS_CALIPER_PATH=`spack location -i caliper`
export AMS_AMQPCPP_PATH=`spack location -i amqp-cpp`
export AMS_OPENSSL_PATH=`spack location -i openssl`
export AMS_CUDA_ARCH=${CUDA_ARCH}

echo "AMS_MFEM_PATH              = $AMS_MFEM_PATH"
echo "AMS_TORCH_PATH             = $AMS_TORCH_PATH"
echo "AMS_FAISS_PATH             = $AMS_FAISS_PATH"
echo "AMS_UMPIRE_PATH            = $AMS_UMPIRE_PATH"
echo "AMS_CUDA_ARCH              = $AMS_CUDA_ARCH"
echo "AMS_HIREDIS_PATH           = $AMS_HIREDIS_PATH"
echo "AMS_REDIS_PLUS_PLUS_PATH   = $AMS_REDIS_PLUS_PLUS_PATH"
echo "AMS_HDF5_PATH              = $AMS_HDF5_PATH"
echo "AMS_CALIPER_PATH           = $AMS_CALIPER_PATH"
echo "AMS_AMQPCPP_PATH           = $AMS_AMQPCPP_PATH"
echo "AMS_OPENSSL_PATH           = $AMS_OPENSSL_PATH"

export AMS_TORCH_PATH=$(echo $AMS_TORCH_PATH/lib/python3.*/site-packages/torch/share/cmake/Torch)
export AMS_AMQPCPP_PATH=$(echo $AMS_AMQPCPP_PATH/cmake)
export AMS_CALIPER_PATH=$(echo $AMS_CALIPER_PATH/share/cmake/caliper)

echo "(for cmake) AMS_TORCH_PATH   = $AMS_TORCH_PATH"
echo "(for cmake) AMS_AMQPCPP_PATH = $AMS_AMQPCPP_PATH"
echo "(for cmake) AMS_CALIPER_PATH = $AMS_CALIPER_PATH"

