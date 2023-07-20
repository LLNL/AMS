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
class IdealGas : public EOS
{
  const double gamma_;
  const double specific_heat_;

public:
  IdealGas(double gamma, double specific_heat)
      : gamma_(gamma), specific_heat_(specific_heat)
  {
  }

#ifdef __ENABLE_PERFFLOWASPECT__
   __attribute__((annotate("@critical_path(pointcut='around')")))
#endif
  void Eval(const int length,
            const double *density,
            const double *energy,
            double *pressure,
            double *soundspeed2,
            double *bulkmod,
            double *temperature) const override
  {
    const double gamma = gamma_;
    const double specific_heat = specific_heat_;

    using mfem::ForallWrap;
    MFEM_FORALL(i, length, {
      pressure[i] = (gamma - 1) * density[i] * energy[i];
      soundspeed2[i] = gamma * (gamma - 1) * energy[i];
      bulkmod[i] = gamma * pressure[i];
      temperature[i] = energy[i] / specific_heat;
    });
  }

  void Eval_with_filter(const int length,
                        const double *density,
                        const double *energy,
                        const bool *filter,
                        double *pressure,
                        double *soundspeed2,
                        double *bulkmod,
                        double *temperature) const override
  {
    const double gamma = gamma_;
    const double specific_heat = specific_heat_;

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
            const double **inputs,
            double **outputs) const override
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
