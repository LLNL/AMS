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
    if (policy >= AMSUQPolicy::AMS_DELTAUQ_MEAN &&
        policy <= AMSUQPolicy::AMS_DELTAUQ_MAX) {
      return true;
    }
    return false;
  }

  static inline bool isFaissUQ(AMSUQPolicy policy)
  {
    if (policy >= AMSUQPolicy::AMS_FAISS_MEAN &&
        policy <= AMSUQPolicy::AMS_FAISS_MAX) {
      return true;
    }
    return false;
  }

  static inline bool isRandomUQ(AMSUQPolicy policy)
  {
    return policy == AMSUQPolicy::AMS_RANDOM;
  }


  static inline bool isUQPolicy(AMSUQPolicy policy)
  {
    if (AMSUQPolicy::AMS_UQ_BEGIN < policy && policy < AMSUQPolicy::AMS_UQ_END)
      return true;
    return false;
  }

  static std::string UQPolicyToStr(AMSUQPolicy policy)
  {
    if (policy == AMSUQPolicy::AMS_RANDOM)
      return "random";
    else if (policy == AMSUQPolicy::AMS_FAISS_MAX)
      return "faiss (max)";
    else if (policy == AMSUQPolicy::AMS_FAISS_MEAN)
      return "faiss (mean)";
    else if (policy == AMSUQPolicy::AMS_DELTAUQ_MEAN)
      return "deltaUQ (mean)";
    else if (policy == AMSUQPolicy::AMS_DELTAUQ_MAX)
      return "deltaUQ (max)";
    return "Unknown";
  }

  static AMSUQPolicy UQPolicyFromStr(std::string &policy)
  {
    if (policy.compare("random") == 0)
      return AMSUQPolicy::AMS_RANDOM;
    else if (policy.compare("faiss (max)") == 0)
      return AMSUQPolicy::AMS_FAISS_MAX;
    else if (policy.compare("faiss (mean)") == 0)
      return AMSUQPolicy::AMS_FAISS_MEAN;
    else if (policy.compare("deltaUQ (mean)") == 0)
      return AMSUQPolicy::AMS_DELTAUQ_MEAN;
    else if (policy.compare("deltaUQ (max)") == 0)
      return AMSUQPolicy::AMS_DELTAUQ_MAX;
    return AMSUQPolicy::AMS_UQ_END;
  }
};

template <typename FPTypeValue>
class UQ : public BaseUQ
{
public:
  UQ(AMSResourceType resourceLocation,
     const AMSUQPolicy uqPolicy,
     std::string &uqPath,
     const int nClusters,
     std::string &surrogatePath,
     FPTypeValue threshold)
      : uqPolicy(uqPolicy), threshold(threshold)
  {
    if (surrogatePath.empty()) {
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
      if (uqPath.empty())
        THROW(std::runtime_error, "Missing file path to a FAISS UQ model");

      hdcache = HDCache<FPTypeValue>::getInstance(
          uqPath, resourceLocation, uqPolicy, nClusters, threshold);
    }

    if (isRandomUQ(uqPolicy))
      randomUQ = std::make_unique<RandomUQ>(resourceLocation, threshold);

    DBG(UQ,
        "UQ Model is of type %s with threshold %f",
        BaseUQ::UQPolicyToStr(uqPolicy).c_str(),
        threshold)
  }

  PERFFASPECT()
  void evaluate(const int totalElements,
                std::vector<const FPTypeValue *> &inputs,
                std::vector<FPTypeValue *> &outputs,
                bool *p_ml_acceptable)
  {

    DBG(UQ,
        "Calling %s surrogate [in:%ld %ld] -> (out:[%ld "
        "%ld])",
        BaseUQ::UQPolicyToStr(uqPolicy).c_str(),
        totalElements,
        inputs.size(),
        totalElements,
        outputs.size());

    if ((uqPolicy == AMSUQPolicy::AMS_DELTAUQ_MEAN) ||
        (uqPolicy == AMSUQPolicy::AMS_DELTAUQ_MAX)) {

      auto &rm = ams::ResourceManager::getInstance();

      CALIPER(CALI_MARK_BEGIN("DELTAUQ SURROGATE");)
      surrogate->evaluate(
          totalElements, inputs, outputs, uqPolicy, p_ml_acceptable, threshold);
      CALIPER(CALI_MARK_END("DELTAUQ SURROGATE");)
    } else if (uqPolicy == AMSUQPolicy::AMS_FAISS_MEAN ||
               uqPolicy == AMSUQPolicy::AMS_FAISS_MAX) {
      CALIPER(CALI_MARK_BEGIN("HDCACHE");)
      hdcache->evaluate(totalElements, inputs, p_ml_acceptable);
      CALIPER(CALI_MARK_END("HDCACHE");)

      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      surrogate->evaluate(totalElements, inputs, outputs);
      CALIPER(CALI_MARK_END("SURROGATE");)
    } else if (uqPolicy == AMSUQPolicy::AMS_RANDOM) {
      CALIPER(CALI_MARK_BEGIN("RANDOM_UQ");)
      DBG(Workflow, "Evaluating Random UQ");
      randomUQ->evaluate(totalElements, p_ml_acceptable);
      CALIPER(CALI_MARK_END("RANDOM_UQ");)

      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      surrogate->evaluate(totalElements, inputs, outputs);
      CALIPER(CALI_MARK_END("SURROGATE");)
    } else {
      THROW(std::runtime_error, "Invalid UQ policy");
    }
  }

  void updateModel(const std::string &model_path,
                   const std::string &uq_path = "")
  {
    if (uqPolicy == AMSUQPolicy::AMS_FAISS_MAX ||
        uqPolicy == AMSUQPolicy::AMS_FAISS_MEAN) {
      THROW(std::runtime_error, "UQ model does not support update.");
    }

    if (uqPolicy == AMSUQPolicy::AMS_RANDOM && uq_path != "") {
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
