/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_RANDOM_UQ_HPP__
#define __AMS_RANDOM_UQ_HPP__

#include "AMS.h"
#include "wf/debug.h"
#include "wf/utils.hpp"

class RandomUQ
{
public:
  PERFFASPECT()
  inline void evaluate(const size_t ndata, bool *is_acceptable) const
  {
    if (resourceLocation == AMSResourceType::DEVICE) {
#ifdef __ENABLE_CUDA__
      random_uq_device<<<1, 1>>>(is_acceptable, ndata, threshold);
#else
      THROW(std::runtime_error,
            "Random-uq is not configured to use device allocations");
#endif
    } else {
      random_uq_host(is_acceptable, ndata, threshold);
    }
  }
  RandomUQ(AMSResourceType resourceLocation, float threshold)
      : resourceLocation(resourceLocation), threshold(threshold)
  {
  }

private:
  AMSResourceType resourceLocation;
  float threshold;
};

#endif
