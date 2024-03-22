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
#include "ml/random_uq.hpp"
#include "ml/surrogate.hpp"
#include "wf/resource_manager.hpp"

static inline bool isNullOrEmpty(const char *p) { return (!p || p[0] == '\0'); }

template <typename FPTypeValue>
class UQ
{
public:
  UQ(AMSResourceType resourceLocation,
     const AMSUQPolicy uqPolicy,
     const char *uqPath,
     const int nClusters,
     const char *surrogatePath,
     FPTypeValue threshold)
      : uqPolicy(uqPolicy), threshold(threshold)
  {
    if (isNullOrEmpty(surrogatePath)) {
      surrogate = nullptr;
      hdcache = nullptr;
      randomUQ = nullptr;
      return;
    }

    if (!(AMSUQPolicy::AMSUQPolicy_BEGIN <= uqPolicy &&
          uqPolicy <= AMSUQPolicy::AMSUQPolicy_END))
      THROW(std::runtime_error, "Invalid UQ policy, value is out-of-bounds");


    bool is_DeltaUQ = ((uqPolicy == AMSUQPolicy::DeltaUQ_Max ||
                        uqPolicy == AMSUQPolicy::DeltaUQ_Mean)
                           ? true
                           : false);

    surrogate = SurrogateModel<FPTypeValue>::getInstance(surrogatePath,
                                                         resourceLocation,
                                                         is_DeltaUQ);

    if (uqPolicy == AMSUQPolicy::FAISS_Max ||
        uqPolicy == AMSUQPolicy::FAISS_Mean) {
      if (isNullOrEmpty(uqPath))
        THROW(std::runtime_error, "Missing file path to a FAISS UQ model");

      hdcache = HDCache<FPTypeValue>::getInstance(
          uqPath, resourceLocation, uqPolicy, nClusters, threshold);
    }

    if (uqPolicy == AMSUQPolicy::RandomUQ)
      randomUQ = std::make_unique<RandomUQ>(resourceLocation, threshold);

    DBG(UQ, "UQ Model is of type %d", uqPolicy)
  }

  PERFFASPECT()
  void evaluate(const int totalElements,
                std::vector<const FPTypeValue *> &inputs,
                std::vector<FPTypeValue *> &outputs,
                bool *p_ml_acceptable)
  {
    if ((uqPolicy == AMSUQPolicy::DeltaUQ_Mean) ||
        (uqPolicy == AMSUQPolicy::DeltaUQ_Max)) {

      auto &rm = ams::ResourceManager::getInstance();

      CALIPER(CALI_MARK_BEGIN("DELTAUQ");)
      const size_t ndims = outputs.size();
      std::vector<FPTypeValue *> outputs_stdev(ndims);
      // TODO: Enable device-side allocation and predicate calculation.
      for (int dim = 0; dim < ndims; ++dim)
        outputs_stdev[dim] =
            rm.allocate<FPTypeValue>(totalElements, AMSResourceType::HOST);

      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      DBG(UQ,
          "Model exists, I am calling DeltaUQ surrogate [%ld %ld] -> (mu:[%ld "
          "%ld], std:[%ld %ld])",
          totalElements,
          inputs.size(),
          totalElements,
          outputs.size(),
          totalElements,
          inputs.size());
      surrogate->evaluate(totalElements, inputs, outputs, outputs_stdev);
      CALIPER(CALI_MARK_END("SURROGATE");)

      // FIXME: We do something sub-optimal. We copy all the data from the GPU
      // to the CPU and then we compute the predicate. Then we copy back the computed
      // predicate to the device. We should avoid this unecessary back and forth.
      bool *predicate = p_ml_acceptable;
      if (surrogate->getModelResource() == AMSResourceType::DEVICE) {
        predicate = rm.allocate<bool>(totalElements, AMSResourceType::HOST);
        rm.copy(p_ml_acceptable, predicate);
      }


      if (uqPolicy == AMSUQPolicy::DeltaUQ_Mean) {
        for (size_t i = 0; i < totalElements; ++i) {
          // Use double for increased precision, range in the calculation
          double mean = 0.0;
          for (size_t dim = 0; dim < ndims; ++dim)
            mean += outputs_stdev[dim][i];
          mean /= ndims;
          predicate[i] = (mean < threshold);
        }
      } else if (uqPolicy == AMSUQPolicy::DeltaUQ_Max) {
        for (size_t i = 0; i < totalElements; ++i) {
          bool is_acceptable = true;
          for (size_t dim = 0; dim < ndims; ++dim)
            if (outputs_stdev[dim][i] >= threshold) {
              is_acceptable = false;
              break;
            }

          predicate[i] = is_acceptable;
        }
      } else {
        THROW(std::runtime_error, "Invalid UQ policy");
      }

      if (surrogate->getModelResource() == AMSResourceType::DEVICE) {
        rm.copy(predicate, p_ml_acceptable);
        rm.deallocate(predicate, AMSResourceType::HOST);
      }

      for (int dim = 0; dim < ndims; ++dim)
        rm.deallocate(outputs_stdev[dim], AMSResourceType::HOST);
      CALIPER(CALI_MARK_END("DELTAUQ");)
    } else if (uqPolicy == AMSUQPolicy::FAISS_Mean ||
               uqPolicy == AMSUQPolicy::FAISS_Max) {
      CALIPER(CALI_MARK_BEGIN("HDCACHE");)
      hdcache->evaluate(totalElements, inputs, p_ml_acceptable);
      CALIPER(CALI_MARK_END("HDCACHE");)

      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      DBG(Workflow, "Model exists, I am calling surrogate (for all data)");
      surrogate->evaluate(totalElements, inputs, outputs);
      CALIPER(CALI_MARK_END("SURROGATE");)
    } else if (uqPolicy == AMSUQPolicy::RandomUQ) {
      CALIPER(CALI_MARK_BEGIN("RANDOM_UQ");)
      DBG(Workflow, "Evaluating Random UQ");
      randomUQ->evaluate(totalElements, p_ml_acceptable);
      CALIPER(CALI_MARK_END("RANDOM_UQ");)

      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      DBG(Workflow, "Model exists, I am calling surrogate (for all data)");
      surrogate->evaluate(totalElements, inputs, outputs);
      CALIPER(CALI_MARK_END("SURROGATE");)
    } else {
      THROW(std::runtime_error, "Invalid UQ policy");
    }
  }

  void updateModel(const std::string &model_path,
                   const std::string &uq_path = "")
  {
    if (uqPolicy != AMSUQPolicy::RandomUQ &&
        uqPolicy != AMSUQPolicy::DeltaUQ_Max &&
        uqPolicy != AMSUQPolicy::DeltaUQ_Mean) {
      THROW(std::runtime_error, "UQ model does not support update.");
    }

    if (uqPolicy == AMSUQPolicy::RandomUQ && uq_path != "") {
      WARNING(Workflow,
              "RandomUQ cannot update hdcache path, ignoring argument")
    }

    surrogate->update(model_path);
    return;
  }

  bool hasSurrogate() { return (surrogate ? true : false); }

private:
  AMSUQPolicy uqPolicy;
  FPTypeValue threshold;
  std::unique_ptr<RandomUQ> randomUQ;
  std::shared_ptr<HDCache<FPTypeValue>> hdcache;
  std::shared_ptr<SurrogateModel<FPTypeValue>> surrogate;
};

#endif
