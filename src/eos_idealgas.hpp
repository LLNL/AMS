#ifndef _IDEALGAS_EOS_HPP_
#define _IDEALGAS_EOS_HPP_

#include "mfem/general/forall.hpp"
#include "eos.hpp"

//! Ideal Gas EOS
//! Code given by Thomas Stitt
class IdealGas : public EOS
{
   const double gamma_;
   const double specific_heat_;

public:
   IdealGas(double gamma, double specific_heat) : gamma_(gamma), specific_heat_(specific_heat) {}

   void Eval(const int length,
             const double *density,
             const double *energy,
             double *pressure,
             double *soundspeed2,
             double *bulkmod,
             double *temperature) override
   {
      const double gamma         = gamma_;
      const double specific_heat = specific_heat_;

      using mfem::ForallWrap;
      MFEM_FORALL(i, length, {
         pressure[i]    = (gamma - 1) * density[i] * energy[i];
         soundspeed2[i] = gamma * (gamma - 1) * energy[i];
         bulkmod[i]     = gamma * pressure[i];
         temperature[i] = energy[i] / specific_heat;
      });
   }
};

#endif
