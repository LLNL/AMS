/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_UQ_HPP__
#define __AMS_UQ_HPP__

#include <stdexcept>
#include <vector>

#include "AMS.h"
#include "ml/hdcache.hpp"
#include "ml/surrogate.hpp"
#include "wf/resource_manager.hpp"

template <typename FPTypeValue>
class UQ
{
public:
  PERFFASPECT()
  static void evaluate(
      AMSUQPolicy uqPolicy,
      const int totalElements,
      std::vector<const FPTypeValue *> &inputs,
      std::vector<FPTypeValue *> &outputs,
      const std::shared_ptr<HDCache<FPTypeValue>> &hdcache,
      const std::shared_ptr<SurrogateModel<FPTypeValue>> &surrogate,
      bool *p_ml_acceptable)
  {
    if ((uqPolicy == AMSUQPolicy::DeltaUQ_Mean) ||
        (uqPolicy == AMSUQPolicy::DeltaUQ_Max)) {
      CALIPER(CALI_MARK_BEGIN("DELTAUQ");)
      const size_t ndims = outputs.size();
      std::vector<FPTypeValue *> outputs_stdev(ndims);
      // TODO: Enable device-side allocation and predicate calculation.
      for (int dim = 0; dim < ndims; ++dim)
        outputs_stdev[dim] =
            ams::ResourceManager::allocate<FPTypeValue>(totalElements,
                                                        AMSResourceType::HOST);

      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      DBG(Workflow,
          "Model exists, I am calling DeltaUQ surrogate (for all data)");
      surrogate->evaluate(totalElements, inputs, outputs, outputs_stdev);
      CALIPER(CALI_MARK_END("SURROGATE");)

      if (uqPolicy == DeltaUQ_Mean) {
        for (size_t i = 0; i < totalElements; ++i) {
          // Use double for increased precision, range in the calculation
          double mean = 0.0;
          for (size_t dim = 0; dim < ndims; ++dim)
            mean += outputs_stdev[dim][i];
          mean /= ndims;
          p_ml_acceptable[i] = (mean < _threshold);
        }
      } else if (uqPolicy == DeltaUQ_Max) {
        for (size_t i = 0; i < totalElements; ++i) {
          bool is_acceptable = true;
          for (size_t dim = 0; dim < ndims; ++dim)
            if (outputs_stdev[dim][i] >= _threshold) {
              is_acceptable = false;
              break;
            }

          p_ml_acceptable[i] = is_acceptable;
        }
      } else {
        THROW(std::runtime_error, "Invalid UQ policy");
      }

      for (int dim = 0; dim < ndims; ++dim)
        ams::ResourceManager::deallocate(outputs_stdev[dim],
                                         AMSResourceType::HOST);
      CALIPER(CALI_MARK_END("DELTAUQ");)
    } else {
      CALIPER(CALI_MARK_BEGIN("HDCACHE");)
      if (hdcache) hdcache->evaluate(totalElements, inputs, p_ml_acceptable);
      CALIPER(CALI_MARK_END("HDCACHE");)

      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      DBG(Workflow, "Model exists, I am calling surrogate (for all data)");
      surrogate->evaluate(totalElements, inputs, outputs);
      CALIPER(CALI_MARK_END("SURROGATE");)
    }
  }

  PERFFASPECT()
  static void setThreshold(FPTypeValue threshold) { _threshold = threshold; }

private:
  static FPTypeValue _threshold;
};

template <typename FPTypeValue>
FPTypeValue UQ<FPTypeValue>::_threshold = 0.5;

#endif
