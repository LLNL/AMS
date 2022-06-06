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
   IdealGas(double gamma, double specific_heat) :
       gamma_(gamma), specific_heat_(specific_heat) {}

   void
   Eval(const int length,
        const double *density,
        const double *energy,
        double *pressure,
        double *soundspeed2,
        double *bulkmod,
        double *temperature) const override
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

   void
   Eval_with_filter(const int length,
                    const double *density,
                    const double *energy,
                    const bool *filter,
                    double *pressure,
                    double *soundspeed2,
                    double *bulkmod,
                    double *temperature) const override
   {
      const double gamma         = gamma_;
      const double specific_heat = specific_heat_;

      using mfem::ForallWrap;
      MFEM_FORALL(i, length, {
         if (filter[i]) {
             pressure[i]    = (gamma - 1) * density[i] * energy[i];
             soundspeed2[i] = gamma * (gamma - 1) * energy[i];
             bulkmod[i]     = gamma * pressure[i];
             temperature[i] = energy[i] / specific_heat;
        }
      });
   }

   void computeRMSE(const int length,
        const double *density,
        const double *energy,
        double *pressure,
        double *soundspeed2,
        double *bulkmod,
        double *temperature) const override {

      double *tmp_press = new double[length];
      double *tmp_sound = new double[length];
      double *tmp_bulkmod = new double[length];
      double *tmp_temperature = new double[length];
      Eval(length, density, energy, tmp_press, tmp_sound, tmp_bulkmod,
                     tmp_temperature);
      double error = 0;
      for (long i = 0; i < length; i++) {
         error += pow((tmp_press[i] - pressure[i]), 2);
         error += pow((tmp_sound[i] - soundspeed2[i]), 2);
         error += pow((tmp_bulkmod[i] - bulkmod[i]), 2);
         error += pow((tmp_temperature[i] - temperature[i]), 2);
      }
      std::cout<< "RMSE:" << error/(double)(4*length) << "\n";
      delete [] tmp_press;
      delete [] tmp_sound;
      delete [] tmp_bulkmod;
      delete [] tmp_temperature;
   }
};

#endif