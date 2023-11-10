/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef _AMS_EOS_HPP_
#define _AMS_EOS_HPP_

#include "eos.hpp"

template <typename FPType>
class AMSEOS : public EOS
{
  AMSExecutor wf_;
  EOS * model_ = nullptr;

public:
  AMSEOS(EOS * model);
  ~AMSEOS() { delete model_; }

  void Eval(const int length,
            const FPType *density,
            const FPType *energy,
            FPType *pressure,
            FPType *soundspeed2,
            FPType *bulkmod,
            FPType *temperature) const override;

  void Eval_with_filter(const int length,
                        const FPType *density,
                        const FPType *energy,
                        const bool *filter,
                        FPType *pressure,
                        FPType *soundspeed2,
                        FPType *bulkmod,
                        FPType *temperature) const override
  {
    throw std::runtime_error("AMSEOS: Eval_with_filter is not implemented");
  }
};

#endif // _AMS_EOS_HPP_
