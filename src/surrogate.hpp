#ifndef _SURROGATE_EOS_HPP_
#define _SURROGATE_EOS_HPP_

#include "mfem/general/forall.hpp"
#include "eos.hpp"
#include "eos_idealgas.hpp"

//! An implementation for a surrogate model
class SurrogateModel : public EOS {

    const IdealGas *base_eos;

public:
    SurrogateModel(const IdealGas *_base_eos) : base_eos(_base_eos) {}

    void Eval_with_uq(const int length,
                      const double *density,
                      const double *energy,
                      double *pressure,
                      double *soundspeed2,
                      double *bulkmod,
                      double *temperature,
                      bool *is_acceptable)  const {

        // fill in random values for uq
        const double uq_factor = 0.5;
        for(int i = 0; i < length; i++) {
            is_acceptable[i] = ((double)rand() / RAND_MAX) <= uq_factor;
        }

        base_eos->Eval(length, density, energy, pressure, soundspeed2, bulkmod, temperature);
    }

    void Eval(const int length,
              const double *density,
              const double *energy,
              double *pressure,
              double *soundspeed2,
              double *bulkmod,
              double *temperature)  const override {

        // this is where we will call the surrogate model
        // through tensorflow or torch
        // currently, we will just use the ideal gas eos
        base_eos->Eval(length, density, energy, pressure, soundspeed2, bulkmod, temperature);
    }
};

#endif
