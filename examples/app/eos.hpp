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
  virtual void Eval(const int length,
                    const FPType **inputs,
                    FPType **outputs) const = 0;

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

template<typename FPType>
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

#endif
