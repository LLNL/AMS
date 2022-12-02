#!/usr/bin/env bash

## load relevant modules on lassen
module load gcc/8.3.1
module load cmake/3.23.1
if [[ "$SYS_TYPE" == "blueos_3_ppc64le_ib_p9" ]]; then
  module load spectrum-mpi/2019.06.24
  module load cuda/11.1.0
fi

## activate spack
source /usr/workspace/AMS/ams-spack-environments/0.19/spack/share/spack/setup-env.sh

## activate the spack environment
spack env activate /usr/workspace/AMS/ams-spack-environments/0.19/$LCSCHEDCLUSTER

## export the paths (currently cmake needs these)
export AMS_MFEM_PATH=`spack location -i mfem`
export AMS_TORCH_PATH=`spack location -i py-torch`
export AMS_FAISS_PATH=`spack location -i faiss`
export AMS_UMPIRE_PATH=`spack location -i umpire`

echo "AMS_MFEM_PATH   = $AMS_MFEM_PATH"
echo "AMS_TORCH_PATH  = $AMS_TORCH_PATH"
echo "AMS_FAISS_PATH  = $AMS_FAISS_PATH"
echo "AMS_UMPIRE_PATH = $AMS_UMPIRE_PATH"

export AMS_TORCH_PATH=$(echo $AMS_TORCH_PATH/lib/python3.*/site-packages/torch/share/cmake/Torch)

echo "(for cmake) AMS_TORCH_PATH = $AMS_TORCH_PATH"
