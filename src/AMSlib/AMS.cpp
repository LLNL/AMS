/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "AMS.h"
#include "include/AMS.h"
#include "ml/uq.hpp"
#include "wf/basedb.hpp"
#include "wf/debug.h"
#include "wf/resource_manager.hpp"
#include "wf/workflow.hpp"

struct AMSWrap {
  using json = nlohmann::json;

  enum UQAggrType {
    Unknown = -1,
    Mean = 0,
    Max = 1,
  };

public:
  std::vector<std::pair<AMSDType, void *>> executors;
  std::unordered_map<std::string, AMSEnvObject> ams_candidate_models;
  AMSDBType dbType = AMSDBType::None;

private:
  void dumpEnv()
  {
    for (auto &KV : ams_candidate_models) {
      DBG(AMS,
          "=================================================================")
      DBG(AMS, "Model: %s", KV.first.c_str());
      if (KV.second.SPath)
        DBG(AMS, "Surrogate Model Path: %s", KV.second.SPath);
      if (KV.second.UQPath) DBG(AMS, "UQ-Model: %s", KV.second.UQPath);
      DBG(AMS,
          "db-Label: %s threshold %f UQ-Policy: %u nClusters: %d",
          KV.second.dbLabel,
          KV.second.threshold,
          KV.second.uqPolicy,
          KV.second.nClusters);
    }
    DBG(AMS,
        "=================================================================")
  }


  static AMSDBType getDBType(std::string type)
  {
    if (type.compare("hdf5") == 0) {
      return AMSDBType::HDF5;
    } else if (type.compare("csv") == 0) {
      return AMSDBType::CSV;
    } else if (type.compare("redis") == 0) {
      return AMSDBType::REDIS;
    } else if (type.compare("rmq") == 0) {
      return AMSDBType::RMQ;
    }
    return AMSDBType::None;
  }

  static UQAggrType getUQAggregate(std::string policy)
  {
    if (policy.compare("mean"))
      return UQAggrType::Mean;
    else if (policy.compare("max"))
      return UQAggrType::Max;
    return UQAggrType::Unknown;
  }

  static AMSUQPolicy getUQType(std::string type)
  {
    if (type.compare("deltaUQ") == 0) {
      return AMSUQPolicy::DeltaUQ_Mean;
    } else if (type.compare("faiss") == 0) {
      return AMSUQPolicy::FAISS_Mean;
    } else if (type.compare("random") == 0) {
      return AMSUQPolicy::Random;
    } else {
      THROW(std::runtime_error, "Unknown uq type " + type);
    }
    return AMSUQPolicy::AMSUQPolicy_END;
  }

  static char *getStringValue(std::string str)
  {
    char *cstr = new char[str.size() + 1];
    str.copy(cstr, str.size());
    cstr[str.size()] = '\0';
    return cstr;
  }


  void parseDomainModels(
      json &jRoot,
      std::unordered_map<std::string, std::string> &domain_map)
  {
    if (!jRoot.contains("domain_models")) return;

    auto domain_models = jRoot["domain_models"];
    for (auto &field : domain_models.items()) {
      auto &name = field.key();
      auto val = field.value().get<std::string>();
      domain_map.emplace(name, val);
    }
    return;
  }

  int parseClusters(json &value)
  {
    if (!value.contains("neighbours"))
      THROW(std::runtime_error, "UQ Policy must contain neighbours");

    return value["neighbours"].get<int>();
  }

  char *parseDBLabel(json &value)
  {
    if (!value.contains("db_label")) {
      THROW(std::runtime_error, "ml model must contain <db_label> entry");
    }

    return getStringValue(value["db_label"].get<std::string>());
  }

  AMSUQPolicy parseUQPolicy(json &value)
  {
    AMSUQPolicy policy = AMSUQPolicy::AMSUQPolicy_END;
    if (value.contains("uq_type")) {
      policy = getUQType(value["uq_type"].get<std::string>());
    } else {
      THROW(std::runtime_error, "Model must specify the UQ type");
    }

    std::cout << "UQ Policy is " << BaseUQ::UQPolicyToStr(policy);
    DBG(AMS, "UQ Policy is %s", BaseUQ::UQPolicyToStr(policy).c_str())

    if (!BaseUQ::isUQPolicy(policy)) {
      THROW(std::runtime_error, "UQ Policy is not supported");
    }

    UQAggrType uqAggregate = UQAggrType::Unknown;
    if (value.contains("uq_aggregate")) {
      uqAggregate = getUQAggregate(value["uq_aggregate"].get<std::string>());
    } else {
      THROW(std::runtime_error, "Model must specify UQ Policy");
    }


    if ((BaseUQ::isDeltaUQ(policy) || BaseUQ::isFaissUQ(policy)) &&
        uqAggregate == UQAggrType::Unknown) {
      THROW(std::runtime_error,
            "UQ Type should be defined or set to undefined value");
    }

    if (uqAggregate == Max) {
      if (BaseUQ::isDeltaUQ(policy)) {
        policy = AMSUQPolicy::DeltaUQ_Max;
      } else if (BaseUQ::isFaissUQ(policy)) {
        policy = AMSUQPolicy::FAISS_Max;
      }
    } else if (uqAggregate == Mean) {
      if (BaseUQ::isDeltaUQ(policy)) {
        policy = AMSUQPolicy::DeltaUQ_Mean;
      } else if (BaseUQ::isFaissUQ(policy)) {
        policy = AMSUQPolicy::FAISS_Mean;
      }
    }
    return policy;
  }

  void parseUQPaths(AMSUQPolicy policy, AMSEnvObject &object, json &jRoot)
  {

    if (!jRoot.contains("model_path")) {
      THROW(std::runtime_error, "Model should contain path");
    }

    object.SPath = getStringValue(jRoot["model_path"].get<std::string>());

    DBG(AMS, "Model Is Random or DeltaUQ %s %u", object.SPath, policy);
    if (BaseUQ::isRandomUQ(policy) || BaseUQ::isDeltaUQ(policy)) {
      object.UQPath = nullptr;
      return;
    }

    if (!jRoot.contains("faiss_path")) {
      THROW(std::runtime_error,
            "Model is of UQ type 'faiss' and thus expecting a path to FAISS");
    }

    object.UQPath = getStringValue(jRoot["faiss_path"].get<std::string>());
  }


  void parseCandidateAMSModels(
      json &jRoot,
      std::unordered_map<std::string, AMSEnvObject> &registered_models)
  {
    if (jRoot.contains("ml_models")) {
      auto models = jRoot["ml_models"];
      for (auto &field : models.items()) {
        AMSEnvObject object = {
            nullptr, nullptr, nullptr, -1, AMSUQPolicy::AMSUQPolicy_END, -1};
        auto entry = registered_models.find(field.key());
        if (entry != registered_models.end()) {
          FATAL(AMS,
                "There are multiple models (%s) with the same name. "
                "Overriding "
                "previous entry",
                field.key().c_str());
        }

        auto value = field.value();
        object.uqPolicy = parseUQPolicy(value);

        if (BaseUQ::isFaissUQ(object.uqPolicy)) {
          object.nClusters = parseClusters(value);
        }

        if (!value.contains("threshold")) {
          THROW(std::runtime_error,
                "Model must define threshold value (threshold < 0 always "
                "performs original code, threshold=1e30 always use the "
                "model)");
        }

        object.threshold = value["threshold"].get<float>();
        parseUQPaths(object.uqPolicy, object, value);
        object.dbLabel = parseDBLabel(value);
        DBG(AMS, "Adding Env Entry");
        registered_models.emplace(field.key(), object);
      }
    }
  }

  void parseDatabase(json &jRoot)
  {
    DBG(AMS, "Parsing Data Base Fields")
    if (jRoot.contains("db")) {
      auto entry = jRoot["db"];
    }
  }

  void mergeCandidatesWithDomain(
      std::unordered_map<std::string, AMSEnvObject> &mlModels,
      const std::unordered_map<std::string, std::string> &domainModels)
  {
    /* We need to match the domain key with the ml-model value (AMSEnvObject) 
     * Conceptually, the domainModels-keys are what application domain scientists are using
     * as existing solvers. And the domainModels-Value is the ml-model that the application scientist
     * decides to assign on the specific solver. We need now to point the domainModels-Key (application-scientist)
     * with the ML scientist model implementation (mlModels->Value) to bridge those independent stake-holders
     */
    for (auto &domainModel : domainModels) {
      auto ml_model = domainModel.second;
      auto registered_model = mlModels.find(ml_model);
      if (registered_model == mlModels.end()) {
        THROW(std::runtime_error,
              "Requesting model: " + ml_model +
                  " which is not registered undel <ml_models> entries");
        return;
      }
      ams_candidate_models.emplace(ml_model, registered_model->second);
    }

    /* Here ams_candidate_models contains all the models that the applicatin can use now.
     * We are pessimistic here and we will find which models are not accessible by the application
     * and delete this entries.
     */

    for (auto it = mlModels.begin(); it != mlModels.end();) {
      auto ml_key = it->first;
      auto ams_it = ams_candidate_models.find(ml_key);
      if (ams_it == mlModels.end()) {
        releaseEnvObject(ams_it->second);
        it = mlModels.erase(it);
      } else {
        it++;
      }
    }
  }

  void releaseEnvObject(AMSEnvObject &object)
  {
    if (object.SPath) delete[] object.SPath;
    if (object.UQPath) delete[] object.UQPath;
    if (object.dbLabel) delete[] object.dbLabel;
  }

public:
  AMSWrap()
  {
    if (const char *object_descr = std::getenv("AMS_OBJECTS")) {
      DBG(AMS, "Opening env file %s", object_descr);
      std::unordered_map<std::string, AMSEnvObject> models;
      std::unordered_map<std::string, std::string> domain_mapping;
      std::ifstream json_file(object_descr);
      json data = json::parse(json_file);
      parseCandidateAMSModels(data, models);
      parseDatabase(data);
      parseDomainModels(data, domain_mapping);
      mergeCandidatesWithDomain(models, domain_mapping);
    }
    dumpEnv();
  }

  ~AMSWrap()
  {
    for (auto E : executors) {
      if (E.second != nullptr) {
        if (E.first == AMSDType::Double) {
          delete reinterpret_cast<ams::AMSWorkflow<double> *>(E.second);
        } else {
          delete reinterpret_cast<ams::AMSWorkflow<float> *>(E.second);
        }
      }
    }

    for (auto it = ams_candidate_models.begin();
         it != ams_candidate_models.end();) {
      releaseEnvObject(it->second);
      it = ams_candidate_models.erase(it);
    }
  }
};

static AMSWrap _amsWrap;

void _AMSExecute(AMSExecutor executor,
                 void *probDescr,
                 const int numElements,
                 const void **input_data,
                 void **output_data,
                 int inputDim,
                 int outputDim,
                 MPI_Comm Comm = 0)
{
  uint64_t index = reinterpret_cast<uint64_t>(executor);
  if (index >= _amsWrap.executors.size())
    throw std::runtime_error("AMS Executor identifier does not exist\n");
  auto currExec = _amsWrap.executors[index];

  if (currExec.first == AMSDType::Double) {
    ams::AMSWorkflow<double> *dWF =
        reinterpret_cast<ams::AMSWorkflow<double> *>(currExec.second);
    dWF->evaluate(probDescr,
                  numElements,
                  reinterpret_cast<const double **>(input_data),
                  reinterpret_cast<double **>(output_data),
                  inputDim,
                  outputDim,
                  Comm);
  } else if (currExec.first == AMSDType::Single) {
    ams::AMSWorkflow<float> *sWF =
        reinterpret_cast<ams::AMSWorkflow<float> *>(currExec.second);
    sWF->evaluate(probDescr,
                  numElements,
                  reinterpret_cast<const float **>(input_data),
                  reinterpret_cast<float **>(output_data),
                  inputDim,
                  outputDim,
                  Comm);
  } else {
    throw std::invalid_argument("Data type is not supported by AMSLib!");
    return;
  }
}

#ifdef __cplusplus
extern "C" {
#endif

AMSExecutor AMSCreateExecutor(const AMSConfig config)
{
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    auto &rm = ams::ResourceManager::getInstance();
    rm.init();
  });

  if (config.dType == Double) {
    ams::AMSWorkflow<double> *dWF =
        new ams::AMSWorkflow<double>(config.cBack,
                                     config.UQPath,
                                     config.SPath,
                                     config.DBPath,
                                     config.dbType,
                                     config.device,
                                     config.threshold,
                                     config.uqPolicy,
                                     config.nClusters,
                                     config.pId,
                                     config.wSize,
                                     config.ePolicy);
    _amsWrap.executors.push_back(
        std::make_pair(config.dType, static_cast<void *>(dWF)));
    return reinterpret_cast<AMSExecutor>(_amsWrap.executors.size() - 1L);
  } else if (config.dType == AMSDType::Single) {
    ams::AMSWorkflow<float> *sWF =
        new ams::AMSWorkflow<float>(config.cBack,
                                    config.UQPath,
                                    config.SPath,
                                    config.DBPath,
                                    config.dbType,
                                    config.device,
                                    static_cast<float>(config.threshold),
                                    config.uqPolicy,
                                    config.nClusters,
                                    config.pId,
                                    config.wSize,
                                    config.ePolicy);
    _amsWrap.executors.push_back(
        std::make_pair(config.dType, static_cast<void *>(sWF)));
    return reinterpret_cast<AMSExecutor>(_amsWrap.executors.size() - 1L);
  } else {
    throw std::invalid_argument("Data type is not supported by AMSLib!");
    return reinterpret_cast<AMSExecutor>(-1L);
  }
}

void AMSExecute(AMSExecutor executor,
                void *probDescr,
                const int numElements,
                const void **input_data,
                void **output_data,
                int inputDim,
                int outputDim)
{
  _AMSExecute(executor,
              probDescr,
              numElements,
              input_data,
              output_data,
              inputDim,
              outputDim);
}

void AMSDestroyExecutor(AMSExecutor executor)
{
  uint64_t index = reinterpret_cast<uint64_t>(executor);
  if (index >= _amsWrap.executors.size())
    throw std::runtime_error("AMS Executor identifier does not exist\n");
  auto currExec = _amsWrap.executors[index];

  if (currExec.first == AMSDType::Double) {
    delete reinterpret_cast<ams::AMSWorkflow<double> *>(currExec.second);
  } else if (currExec.first == AMSDType::Single) {
    delete reinterpret_cast<ams::AMSWorkflow<float> *>(currExec.second);
  } else {
    throw std::invalid_argument("Data type is not supported by AMSLib!");
    return;
  }
}

#ifdef __ENABLE_MPI__
void AMSDistributedExecute(AMSExecutor executor,
                           MPI_Comm Comm,
                           void *probDescr,
                           const int numElements,
                           const void **input_data,
                           void **output_data,
                           int inputDim,
                           int outputDim)
{
  _AMSExecute(executor,
              probDescr,
              numElements,
              input_data,
              output_data,
              inputDim,
              outputDim,
              Comm);
}
#endif

const char *AMSGetAllocatorName(AMSResourceType device)
{
  auto &rm = ams::ResourceManager::getInstance();
  return std::move(rm.getAllocatorName(device)).c_str();
}

void AMSSetAllocator(AMSResourceType resource, const char *alloc_name)
{
  auto &rm = ams::ResourceManager::getInstance();
  rm.setAllocator(std::string(alloc_name), resource);
}

#ifdef __cplusplus
}
#endif
