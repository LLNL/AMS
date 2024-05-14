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


struct AMSAbstractModel {
  enum UQAggrType {
    Unknown = -1,
    Mean = 0,
    Max = 1,
  };

public:
  std::string SPath;
  std::string UQPath;
  std::string DBLabel;
  double threshold;
  AMSUQPolicy uqPolicy;
  int nClusters;

  static AMSUQPolicy getUQType(std::string type)
  {
    if (type.compare("deltaUQ") == 0) {
      return AMSUQPolicy::AMS_DELTAUQ_MEAN;
    } else if (type.compare("faiss") == 0) {
      return AMSUQPolicy::AMS_FAISS_MEAN;
    } else if (type.compare("random") == 0) {
      return AMSUQPolicy::AMS_RANDOM;
    } else {
      THROW(std::runtime_error, "Unknown uq type " + type);
    }
    return AMSUQPolicy::AMS_UQ_END;
  }

  static UQAggrType getUQAggregate(std::string policy)
  {
    if (policy.compare("mean"))
      return UQAggrType::Mean;
    else if (policy.compare("max"))
      return UQAggrType::Max;
    return UQAggrType::Unknown;
  }

  std::string parseDBLabel(nlohmann::json &value)
  {
    if (!value.contains("db_label")) {
      THROW(std::runtime_error, "ml model must contain <db_label> entry");
    }

    return value["db_label"].get<std::string>();
  }


  void parseUQPaths(AMSUQPolicy policy, nlohmann::json &jRoot)
  {

    if (!jRoot.contains("model_path")) {
      THROW(std::runtime_error, "Model should contain path");
    }

    SPath = jRoot["model_path"].get<std::string>();
    std::cout << SPath << "is \n";

    DBG(AMS, "Model Is Random or DeltaUQ %s %u", SPath.c_str(), policy);
    if (BaseUQ::isRandomUQ(policy) || BaseUQ::isDeltaUQ(policy)) {
      UQPath = "";
      return;
    }

    if (!jRoot.contains("faiss_path")) {
      THROW(std::runtime_error,
            "Model is of UQ type 'faiss' and thus expecting a path to FAISS");
    }

    UQPath = jRoot["faiss_path"].get<std::string>();
  }


  int parseClusters(nlohmann::json &value)
  {
    if (!value.contains("neighbours"))
      THROW(std::runtime_error, "UQ Policy must contain neighbours");

    return value["neighbours"].get<int>();
  }


  AMSUQPolicy parseUQPolicy(nlohmann::json &value)
  {
    AMSUQPolicy policy = AMSUQPolicy::AMS_UQ_END;
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
        policy = AMSUQPolicy::AMS_DELTAUQ_MAX;
      } else if (BaseUQ::isFaissUQ(policy)) {
        policy = AMSUQPolicy::AMS_FAISS_MAX;
      }
    } else if (uqAggregate == Mean) {
      if (BaseUQ::isDeltaUQ(policy)) {
        policy = AMSUQPolicy::AMS_DELTAUQ_MEAN;
      } else if (BaseUQ::isFaissUQ(policy)) {
        policy = AMSUQPolicy::AMS_FAISS_MEAN;
      }
    }
    return policy;
  }


public:
  AMSAbstractModel(nlohmann::json &value)
  {

    uqPolicy = parseUQPolicy(value);

    if (BaseUQ::isFaissUQ(uqPolicy)) {
      nClusters = parseClusters(value);
    }

    if (!value.contains("threshold")) {
      THROW(std::runtime_error,
            "Model must define threshold value (threshold < 0 always "
            "performs original code, threshold=1e30 always use the "
            "model)");
    }
    threshold = value["threshold"].get<float>();
    parseUQPaths(uqPolicy, value);
    DBLabel = parseDBLabel(value);
  }


  AMSAbstractModel(AMSUQPolicy uq_policy,
                   const char *surrogate_path,
                   const char *uq_path,
                   const char *db_label,
                   double threshold,
                   int num_clusters)
  {
    if (db_label == nullptr)
      FATAL(AMS, "registering model without a database identifier\n");

    DBLabel = std::string(db_label);

    if (!BaseUQ::isUQPolicy(uq_policy)) {
      FATAL(AMS, "Invalid UQ policy %d", uq_policy)
    }

    uqPolicy = uq_policy;

    if (surrogate_path != nullptr) SPath = std::string(surrogate_path);

    if (uq_path != nullptr) UQPath = std::string(uq_path);

    this->threshold = threshold;
    num_clusters = num_clusters;
    DBG(AMS,
        "Registered Model %s %g",
        BaseUQ::UQPolicyToStr(uqPolicy).c_str(),
        threshold);
  }


  void dump()
  {
    if (!SPath.empty()) DBG(AMS, "Surrogate Model Path: %s", SPath.c_str());
    if (!UQPath.empty()) DBG(AMS, "UQ-Model: %s", UQPath.c_str());
    DBG(AMS,
        "db-Label: %s threshold %f UQ-Policy: %u nClusters: %d",
        DBLabel.c_str(),
        threshold,
        uqPolicy,
        nClusters);
  }
};


class AMSWrap
{
  using json = nlohmann::json;

public:
  std::vector<std::pair<AMSDType, void *>> executors;
  std::vector<AMSAbstractModel> registered_models;
  std::unordered_map<std::string, int> ams_candidate_models;
  AMSDBType dbType = AMSDBType::AMS_NONE;

private:
  void dumpEnv()
  {
    for (auto &KV : ams_candidate_models) {
      DBG(AMS,
          "\t\t\t Model: %s With AMSAbstractID: %d",
          KV.first.c_str(),
          KV.second);
      if (KV.second >= ams_candidate_models.size()) {
        FATAL(AMS,
              "Candidate model mapped to AMSAbstractID that does not exist "
              "(%d)",
              KV.second);
      }
      auto &abstract_model = registered_models[KV.second];
      abstract_model.dump();
    }
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
      domain_map.emplace(val, name);
    }
    return;
  }

  void parseCandidateAMSModels(
      json &jRoot,
      std::unordered_map<std::string, std::string> ml_domain_mapping)
  {
    if (jRoot.contains("ml_models")) {
      auto models = jRoot["ml_models"];
      for (auto &field : models.items()) {
        // We skip models not registered to respective domains. We will not use
        // those.
        auto &key = field.key();
        if (ml_domain_mapping.find(key) == ml_domain_mapping.end()) continue;

        if (ams_candidate_models.find(ml_domain_mapping[key]) !=
            ams_candidate_models.end()) {
          FATAL(AMS,
                "Domain Model %s has multiple ml model mappings",
                ml_domain_mapping[key].c_str())
        }

        registered_models.push_back(AMSAbstractModel(field.value()));
        // We add the value of the domain mappings, as the application can
        // only query based on these.
        ams_candidate_models.emplace(ml_domain_mapping[key],
                                     registered_models.size() - 1);
      }
    }
  }

  void parseDatabase(json &jRoot)
  {
    DBG(AMS, "Parsing Data Base Fields")
    if (jRoot.contains("db")) {
      auto entry = jRoot["db"];
      if (!entry.contains("dbType"))
        THROW(std::runtime_error,
              "JSON file instantiates db-fields without defining a "
              "\"dbType\" "
              "entry");
      auto dbStrType = entry["dbType"].get<std::string>();
      DBG(AMS, "DB Type is: %s", dbStrType.c_str())
      AMSDBType dbType = ams::db::getDBType(dbStrType);
      if (dbType == AMSDBType::AMS_NONE) return;

      if (dbType == AMSDBType::AMS_CSV || dbType == AMSDBType::AMS_HDF5) {
        if (!entry.contains("fs_path"))
          THROW(std::runtime_error,
                "JSON db-fiels does not provide file system path");

        std::string db_path = entry["fs_path"].get<std::string>();
        auto &DB = ams::db::DBManager::getInstance();
        DB.instantiate_fs_db(dbType, db_path);
        DBG(AMS,
            "Configured AMS File system database to point to %s using file "
            "type %s",
            db_path.c_str(),
            dbStrType.c_str());
      }
    }
  }

public:
  AMSWrap()
  {
    if (const char *object_descr = std::getenv("AMS_OBJECTS")) {
      DBG(AMS, "Opening env file %s", object_descr);
      std::ifstream json_file(object_descr);
      json data = json::parse(json_file);
      /* We first parse domain models. Domain models can be potentially 
       * queried and returned to the main application using the "key" value
       * as query parameter. This redirection only applies for ml-models 
       * registered by the application itself.
       */
      std::unordered_map<std::string, std::string> domain_mapping;
      parseDomainModels(data, domain_mapping);
      parseCandidateAMSModels(data, domain_mapping);
      parseDatabase(data);
    }

    dumpEnv();
  }

  int register_model(const char *domain_name,
                     AMSUQPolicy uq_policy,
                     double threshold,
                     const char *surrogate_path,
                     const char *uq_path,
                     const char *db_label,
                     int num_clusters)
  {
    auto model = ams_candidate_models.find(domain_name);
    if (model != ams_candidate_models.end()) {
      FATAL(AMS,
            "Trying to register model on domain: %s but model already exists "
            "%s",
            domain_name,
            registered_models[model->second].SPath.c_str());
    }
    registered_models.push_back(AMSAbstractModel(
        uq_policy, surrogate_path, uq_path, db_label, threshold, num_clusters));
    ams_candidate_models.emplace(std::string(domain_name),
                                 registered_models.size() - 1);
    return registered_models.size() - 1;
  }

  int get_model_index(const char *domain_name)
  {
    auto model = ams_candidate_models.find(domain_name);
    if (model == ams_candidate_models.end()) return -1;

    return model->second;
  }

  AMSAbstractModel &get_model(int index)
  {
    if (index >= registered_models.size()) {
      FATAL(AMS, "Model id: %d does not exist", index);
    }

    return registered_models[index];
  }

  ~AMSWrap()
  {
    for (auto E : executors) {
      if (E.second != nullptr) {
        if (E.first == AMSDType::AMS_DOUBLE) {
          delete reinterpret_cast<ams::AMSWorkflow<double> *>(E.second);
        } else {
          delete reinterpret_cast<ams::AMSWorkflow<float> *>(E.second);
        }
      }
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
  long index = static_cast<long>(executor);
  if (index >= _amsWrap.executors.size())
    throw std::runtime_error("AMS Executor identifier does not exist\n");
  auto currExec = _amsWrap.executors[index];

  if (currExec.first == AMSDType::AMS_DOUBLE) {
    ams::AMSWorkflow<double> *dWF =
        reinterpret_cast<ams::AMSWorkflow<double> *>(currExec.second);
    dWF->evaluate(probDescr,
                  numElements,
                  reinterpret_cast<const double **>(input_data),
                  reinterpret_cast<double **>(output_data),
                  inputDim,
                  outputDim,
                  Comm);
  } else if (currExec.first == AMSDType::AMS_SINGLE) {
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

AMSExecutor AMSCreateExecutor(AMSCAbstrModel model,
                              AMSExecPolicy exec_policy,
                              AMSDType data_type,
                              AMSResourceType resource_type,
                              AMSPhysicFn call_back,
                              int process_id,
                              int world_size)
{
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    auto &rm = ams::ResourceManager::getInstance();
    rm.init();
  });

  AMSAbstractModel &model_descr = _amsWrap.get_model(model);

  if (data_type == AMSDType::AMS_DOUBLE) {
    ams::AMSWorkflow<double> *dWF =
        new ams::AMSWorkflow<double>(call_back,
                                     model_descr.UQPath,
                                     model_descr.SPath,
                                     model_descr.DBLabel,
                                     resource_type,
                                     model_descr.threshold,
                                     model_descr.uqPolicy,
                                     model_descr.nClusters,
                                     process_id,
                                     world_size,
                                     exec_policy);
    _amsWrap.executors.push_back(
        std::make_pair(data_type, static_cast<void *>(dWF)));
    return static_cast<AMSExecutor>(_amsWrap.executors.size()) - 1L;
  } else if (data_type == AMSDType::AMS_SINGLE) {
    ams::AMSWorkflow<float> *sWF =
        new ams::AMSWorkflow<float>(call_back,
                                    model_descr.UQPath,
                                    model_descr.SPath,
                                    model_descr.DBLabel,
                                    resource_type,
                                    model_descr.threshold,
                                    model_descr.uqPolicy,
                                    model_descr.nClusters,
                                    process_id,
                                    world_size,
                                    exec_policy);
    _amsWrap.executors.push_back(
        std::make_pair(data_type, static_cast<void *>(sWF)));
    return static_cast<AMSExecutor>(_amsWrap.executors.size()) - 1L;
  } else {
    throw std::invalid_argument("Data type is not supported by AMSLib!");
    return static_cast<AMSExecutor>(-1);
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
  long index = static_cast<long>(executor);
  if (index >= _amsWrap.executors.size())
    throw std::runtime_error("AMS Executor identifier does not exist\n");
  auto currExec = _amsWrap.executors[index];

  if (currExec.first == AMSDType::AMS_DOUBLE) {
    delete reinterpret_cast<ams::AMSWorkflow<double> *>(currExec.second);
  } else if (currExec.first == AMSDType::AMS_SINGLE) {
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
  return rm.getAllocatorName(device).c_str();
}

void AMSSetAllocator(AMSResourceType resource, const char *alloc_name)
{
  auto &rm = ams::ResourceManager::getInstance();
  rm.setAllocator(std::string(alloc_name), resource);
}

AMSCAbstrModel AMSRegisterAbstractModel(const char *domain_name,
                                        AMSUQPolicy uq_policy,
                                        double threshold,
                                        const char *surrogate_path,
                                        const char *uq_path,
                                        const char *db_label,
                                        int num_clusters)
{
  int id = _amsWrap.register_model(domain_name,
                                   uq_policy,
                                   threshold,
                                   surrogate_path,
                                   uq_path,
                                   db_label,
                                   num_clusters);

  return id;
}


AMSCAbstrModel AMSQueryModel(const char *domain_model)
{
  return _amsWrap.get_model_index(domain_model);
}

void configure_ams_fs_database(AMSDBType db_type, const char *db_path)
{
  auto &db_instance = ams::db::DBManager::getInstance();
  db_instance.instantiate_fs_db(db_type, std::string(db_path));
}

#ifdef __cplusplus
}
#endif
