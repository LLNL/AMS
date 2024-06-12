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
#include "wf/device.hpp"
#include "wf/utils.hpp"

class RandomUQ
{
public:
  PERFFASPECT()
  inline void evaluate(const size_t ndata, bool *is_acceptable)
  {
    if (resourceLocation == AMSResourceType::AMS_DEVICE) {
      ams::device_random_uq(seed, is_acceptable, ndata, threshold);
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
