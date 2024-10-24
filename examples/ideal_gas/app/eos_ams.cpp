/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "eos_ams.hpp"

#include <vector>

template <typename FPType>
void callBack(void *cls,
              long elements,
              const void *const *inputs,
              void *const *outputs)
{
  static_cast<EOS<FPType> *>(cls)->Eval(elements,
                                        static_cast<const FPType *>(inputs[0]),
                                        static_cast<const FPType *>(inputs[1]),
                                        static_cast<FPType *>(outputs[0]),
                                        static_cast<FPType *>(outputs[1]),
                                        static_cast<FPType *>(outputs[2]),
                                        static_cast<FPType *>(outputs[3]));
}


template <typename FPType>
AMSEOS<FPType>::AMSEOS(EOS<FPType> *model,
                       const AMSDBType db_type,
                       const AMSDType dtype,
                       const AMSExecPolicy exec_policy,
                       const AMSResourceType res_type,
                       const AMSUQPolicy uq_policy,
                       const int k_nearest,
                       const int mpi_task,
                       const int mpi_nproc,
                       const double threshold,
                       const char *surrogate_path,
                       const char *uq_path)
    : model_(model)
{
  AMSCAbstrModel model_descr = AMSRegisterAbstractModel("ideal_gas",
                                                        uq_policy,
                                                        threshold,
                                                        surrogate_path,
                                                        uq_path,
                                                        "ideal_gas",
                                                        k_nearest);
#ifdef __ENABLE_MPI__
  wf_ = AMSCreateDistributedExecutor(model_descr,
                                     dtype,
                                     res_type,
                                     (AMSPhysicFn)callBack<FPType>,
                                     MPI_COMM_WORLD,
                                     mpi_task,
                                     mpi_nproc);
#else
  wf_ = AMSCreateExecutor(model_descr,
                          dtype,
                          res_type,
                          (AMSPhysicFn)callBack<FPType>,
                          mpi_task,
                          mpi_nproc);
#endif
}

template <typename FPType>
#ifdef __ENABLE_PERFFLOWASPECT__
__attribute__((annotate("@critical_path(pointcut='around')")))
#endif
void AMSEOS<FPType>::Eval(const int length,
                          const FPType *density,
                          const FPType *energy,
                          FPType *pressure,
                          FPType *soundspeed2,
                          FPType *bulkmod,
                          FPType *temperature) const
{
  std::vector<const FPType *> inputs = {density, energy};
  std::vector<FPType *> outputs = {pressure, soundspeed2, bulkmod, temperature};

  AMSExecute(wf_,
             (void *)model_,
             length,
             reinterpret_cast<const void **>(inputs.data()),
             reinterpret_cast<void **>(outputs.data()),
             inputs.size(),
             outputs.size());
}

template class AMSEOS<double>;
template class AMSEOS<float>;
