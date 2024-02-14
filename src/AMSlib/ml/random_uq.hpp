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

static size_t round_up(size_t num, size_t denom)
{
  return (num + denom - 1) / denom;
}

class RandomUQ
{
public:
  PERFFASPECT()
  inline void evaluate(const size_t ndata, bool *is_acceptable)
  {
    if (resourceLocation == AMSResourceType::DEVICE) {
#ifdef __ENABLE_CUDA__
      //TODO: Move all of this code on device.cpp and provide better logic regarding
      // number of threads
      size_t threads = 256;
      size_t blocks = round_up(ndata, threads);
      random_uq_device<<<blocks, threads>>>(seed,
                                            is_acceptable,
                                            ndata,
                                            threshold);
      seed = seed + 1;
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
  size_t seed;
  AMSResourceType resourceLocation;
  float threshold;
};

#endif
