/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef _EOS_HPP_
#define _EOS_HPP_

//! Abstract EOS class
//! Code given by Thomas Stitt
class EOS
{

public:
  virtual void Eval(const int length,
                    const double **inputs,
                    double **outputs) const = 0;

  virtual void Eval(const int length,
                    const double *density,
                    const double *energy,
                    double *pressure,
                    double *soundspeed2,
                    double *bulkmod,
                    double *temperature) const = 0;

  virtual void Eval_with_filter(const int length,
                                const double *density,
                                const double *energy,
                                const bool *filter,
                                double *pressure,
                                double *soundspeed2,
                                double *bulkmod,
                                double *temperature) const = 0;
};

void callBack(void *cls,
              long elements,
              const void *const *inputs,
              void *const *outputs)
{
  static_cast<EOS *>(cls)->Eval(elements,
                                static_cast<const double *>(inputs[0]),
                                static_cast<const double *>(inputs[1]),
                                static_cast<double *>(outputs[0]),
                                static_cast<double *>(outputs[1]),
                                static_cast<double *>(outputs[2]),
                                static_cast<double *>(outputs[3]));
}

#endif
