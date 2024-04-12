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

class BaseUQ
{
public:
  static inline bool isDeltaUQ(AMSUQPolicy policy)
  {
    if (policy >= AMSUQPolicy::DeltaUQ_Mean &&
        policy <= AMSUQPolicy::DeltaUQ_Max) {
      return true;
    }
    return false;
  }

  static inline bool isFaissUQ(AMSUQPolicy policy)
  {
    if (policy >= AMSUQPolicy::FAISS_Mean && policy <= AMSUQPolicy::FAISS_Max) {
      return true;
    }
    return false;
  }

  static inline bool isRandomUQ(AMSUQPolicy policy)
  {
    return policy == AMSUQPolicy::Random;
  }


  static inline bool isUQPolicy(AMSUQPolicy policy)
  {
    if (AMSUQPolicy::AMSUQPolicy_BEGIN < policy &&
        policy < AMSUQPolicy::AMSUQPolicy_END)
      return true;
    return false;
  }

  static std::string UQPolicyToStr(AMSUQPolicy policy)
  {
    if (policy == AMSUQPolicy::Random)
      return "random";
    else if (policy == AMSUQPolicy::FAISS_Max)
      return "faiss (max)";
    else if (policy == AMSUQPolicy::FAISS_Mean)
      return "faiss (mean)";
    else if (policy == AMSUQPolicy::DeltaUQ_Mean)
      return "deltaUQ (mean)";
    else if (policy == AMSUQPolicy::DeltaUQ_Max)
      return "deltaUQ (max)";
    return "Unknown";
  }
};

template <typename FPTypeValue>
class UQ : public BaseUQ
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

    if (!isUQPolicy(uqPolicy))
      THROW(std::runtime_error, "Invalid UQ policy, value is out-of-bounds");


    bool is_DeltaUQ = isDeltaUQ(uqPolicy);

    surrogate = SurrogateModel<FPTypeValue>::getInstance(surrogatePath,
                                                         resourceLocation,
                                                         is_DeltaUQ);

    if (isFaissUQ(uqPolicy)) {
      if (isNullOrEmpty(uqPath))
        THROW(std::runtime_error, "Missing file path to a FAISS UQ model");

      hdcache = HDCache<FPTypeValue>::getInstance(
          uqPath, resourceLocation, uqPolicy, nClusters, threshold);
    }

    if (isRandomUQ(uqPolicy))
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

      CALIPER(CALI_MARK_BEGIN("DELTAUQ SURROGATE");)
      DBG(UQ,
          "Model exists, I am calling DeltaUQ surrogate [%ld %ld] -> (mu:[%ld "
          "%ld])",
          totalElements,
          inputs.size(),
          totalElements,
          outputs.size());
      surrogate->evaluate(
          totalElements, inputs, outputs, uqPolicy, p_ml_acceptable, threshold);
      CALIPER(CALI_MARK_END("DELTAUQ SURROGATE");)
    } else if (uqPolicy == AMSUQPolicy::FAISS_Mean ||
               uqPolicy == AMSUQPolicy::FAISS_Max) {
      CALIPER(CALI_MARK_BEGIN("HDCACHE");)
      hdcache->evaluate(totalElements, inputs, p_ml_acceptable);
      CALIPER(CALI_MARK_END("HDCACHE");)

      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      DBG(Workflow, "Model exists, I am calling surrogate (for all data)");
      surrogate->evaluate(totalElements, inputs, outputs);
      CALIPER(CALI_MARK_END("SURROGATE");)
    } else if (uqPolicy == AMSUQPolicy::Random) {
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
    if (uqPolicy != AMSUQPolicy::Random &&
        uqPolicy != AMSUQPolicy::DeltaUQ_Max &&
        uqPolicy != AMSUQPolicy::DeltaUQ_Mean) {
      THROW(std::runtime_error, "UQ model does not support update.");
    }

    if (uqPolicy == AMSUQPolicy::Random && uq_path != "") {
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
