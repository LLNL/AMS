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
template <typename FPType>
class EOS
{
public:
#ifdef __ENABLE_PERFFLOWASPECT__
  __attribute__((annotate("@critical_path(pointcut='around')")))
#endif
  void
  Eval(const int length, const FPType **inputs, FPType **outputs) const
  {
    Eval(length,
         inputs[0],
         inputs[1],
         outputs[0],
         outputs[1],
         outputs[2],
         outputs[3]);
  }

  virtual ~EOS() = default;

  virtual void Eval(const int length,
                    const FPType *density,
                    const FPType *energy,
                    FPType *pressure,
                    FPType *soundspeed2,
                    FPType *bulkmod,
                    FPType *temperature) const = 0;

  virtual void Eval_with_filter(const int length,
                                const FPType *density,
                                const FPType *energy,
                                const bool *filter,
                                FPType *pressure,
                                FPType *soundspeed2,
                                FPType *bulkmod,
                                FPType *temperature) const = 0;
};
#endif
