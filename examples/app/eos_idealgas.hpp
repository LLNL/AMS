/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef _IDEALGAS_EOS_HPP_
#define _IDEALGAS_EOS_HPP_

#include "eos.hpp"
#include "mfem/general/forall.hpp"

//! Ideal Gas EOS
//! Code given by Thomas Stitt
template<typename FPType>
class IdealGas : public EOS<FPType>
{
  const FPType gamma_;
  const FPType specific_heat_;

public:
  IdealGas(FPType gamma, FPType specific_heat)
      : gamma_(gamma), specific_heat_(specific_heat)
  {
  }

#ifdef __ENABLE_PERFFLOWASPECT__
   __attribute__((annotate("@critical_path(pointcut='around')")))
#endif
  void Eval(const int length,
            const FPType *density,
            const FPType *energy,
            FPType *pressure,
            FPType *soundspeed2,
            FPType *bulkmod,
            FPType *temperature) const override
  {
    const FPType gamma = gamma_;
    const FPType specific_heat = specific_heat_;

    using mfem::ForallWrap;
    MFEM_FORALL(i, length, {
      pressure[i] = (gamma - 1) * density[i] * energy[i];
      soundspeed2[i] = gamma * (gamma - 1) * energy[i];
      bulkmod[i] = gamma * pressure[i];
      temperature[i] = energy[i] / specific_heat;
    });
  }

  void Eval_with_filter(const int length,
                        const FPType *density,
                        const FPType *energy,
                        const bool *filter,
                        FPType *pressure,
                        FPType *soundspeed2,
                        FPType *bulkmod,
                        FPType *temperature) const override
  {
    const FPType gamma = gamma_;
    const FPType specific_heat = specific_heat_;

    using mfem::ForallWrap;
    MFEM_FORALL(i, length, {
      if (filter[i]) {
        pressure[i] = (gamma - 1) * density[i] * energy[i];
        soundspeed2[i] = gamma * (gamma - 1) * energy[i];
        bulkmod[i] = gamma * pressure[i];
        temperature[i] = energy[i] / specific_heat;
      }
    });
  }

#ifdef __ENABLE_PERFFLOWASPECT__
   __attribute__((annotate("@critical_path(pointcut='around')")))
#endif
  void Eval(const int length,
            const FPType **inputs,
            FPType **outputs) const override
  {
    Eval(length,
         inputs[0],
         inputs[1],
         outputs[0],
         outputs[1],
         outputs[2],
         outputs[3]);
  }
};

#endif
