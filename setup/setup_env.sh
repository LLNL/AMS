#!/usr/bin/env bash


## need to know the code root
#miniapp_code_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#echo "miniapp_code_root = $miniapp_code_root"


## load relevant modules on lassen
module load gcc/8.3.1
module load cuda/11.1.0
module load cmake/3.20.2
module load spectrum-mpi/2019.06.24


## activate spack
source /usr/workspace/AMS/spack/share/spack/setup-env.sh


## activate spack env (comes with the repo)
#spack env activate $miniapp_code_root/spack_env
spack load umpire caliper py-torch
spack load mfem/dffe5gw
spack load faiss/qems7pu

# export the paths (currently cmake needs these)
export AMS_MFEM_PATH=`spack location -i mfem/dffe5gw`
export AMS_TORCH_PATH=`spack location -i py-torch`
export AMS_FAISS_PATH=`spack location -i faiss/qems7pu`
export AMS_UMPIRE_PATH=`spack location -i umpire`

echo "AMS_MFEM_PATH   = $AMS_MFEM_PATH"
echo "AMS_TORCH_PATH  = $AMS_TORCH_PATH"
echo "AMS_FAISS_PATH  = $AMS_FAISS_PATH"
echo "AMS_UMPIRE_PATH = $AMS_UMPIRE_PATH"

export AMS_TORCH_PATH=$AMS_TORCH_PATH/lib/python3.8/site-packages/torch/share/cmake/Torch

