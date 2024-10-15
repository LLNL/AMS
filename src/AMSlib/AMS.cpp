/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "AMS.h"

#include <limits.h>
#ifdef __ENABLE_MPI__
#include <mpi.h>
#endif
#include <unistd.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/AMS.h"
#include "ml/uq.hpp"
#include "wf/basedb.hpp"
#include "wf/debug.h"
#include "wf/logger.hpp"
#include "wf/resource_manager.hpp"
#include "wf/workflow.hpp"

static int get_rank_id()
{
  if (const char *rid = std::getenv("SLURM_PROCID")) {
    return std::stoi(rid);
  } else if (const char *jsm = std::getenv("JSM_NAMESPACE_RANK")) {
    return std::stoi(jsm);
  } else if (const char *pmi = std::getenv("PMIX_RANK")) {
    return std::stoi(pmi);
  }
  return 0;
}

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
  bool DebugDB;
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
    if (policy.compare("mean") == 0)
      return UQAggrType::Mean;
    else if (policy.compare("max") == 0)
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

  bool parseDebugDB(nlohmann::json &value)
  {
    if (!value.contains("debug_db")) {
      return false;
    }

    return value["debug_db"].get<bool>();
  }


  void parseUQPaths(AMSUQPolicy policy, nlohmann::json &jRoot)
  {

    /* 
     * Empty models can exist in cases were the user annotates
     * the code without having data to train a model. In such a case,
     * the user deploys without specifying the model and lib AMS will
     * collect everything
     */
    if (!jRoot.contains("model_path")) {
      SPath = "";
    } else {
      SPath = jRoot["model_path"].get<std::string>();
    }

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
    DBG(AMS, "UQ Policy is %s", BaseUQ::UQPolicyToStr(policy).c_str())
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
    DebugDB = parseDebugDB(value);

    CFATAL(AMS,
           DebugDB && (SPath.empty()),
           "To store predicates in dabase, a surrogate model field is "
           "mandatory");
  }


  AMSAbstractModel(AMSUQPolicy uq_policy,
                   const char *surrogate_path,
                   const char *uq_path,
                   const char *db_label,
                   double threshold,
                   int num_clusters)
  {
    DebugDB = false;
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
    nClusters = num_clusters;
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


/* The class is reponsible to instantiate and 
 * initialize objects from environment variables
 * and acts as the C to CPP wrapper
 */
class AMSWrap
{
  using json = nlohmann::json;

public:
  std::vector<std::pair<AMSDType, void *>> executors;
  std::vector<std::pair<std::string, AMSAbstractModel>> registered_models;
  std::unordered_map<std::string, int> ams_candidate_models;
  AMSDBType dbType = AMSDBType::AMS_NONE;
  ams::ResourceManager &memManager;
  int rId;

private:
  void dumpEnv()
  {
    for (auto &KV : ams_candidate_models) {
      DBG(AMS, "\n")
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
      auto &abstract_model = registered_models[KV.second].second;
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
    if (!jRoot.contains("ml_models")) return;
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

      registered_models.push_back(
          std::make_pair(ml_domain_mapping[key],
                         AMSAbstractModel(field.value())));
      // We add the value of the domain mappings, as the application can
      // only query based on these.
      ams_candidate_models.emplace(ml_domain_mapping[key],
                                   registered_models.size() - 1);
    }
  }

  void setupFSDB(json &entry, std::string &dbStrType)
  {
    if (!entry.contains("fs_path"))
      THROW(std::runtime_error,
            "JSON db-fields does not provide file system path");

    std::string db_path = entry["fs_path"].get<std::string>();
    auto &DB = ams::db::DBManager::getInstance();
    DB.instantiate_fs_db(dbType, db_path);
    DBG(AMS,
        "Configured AMS File system database to point to %s using file "
        "type %s",
        db_path.c_str(),
        dbStrType.c_str());
  }

  template <typename T>
  T getEntry(json &entry, std::string field)
  {
    if (!entry.contains(field)) {
      THROW(std::runtime_error,
            ("I was expecting entry '" + field + "' to exist in json").c_str())
    }
    return entry[field].get<T>();
  }

  void setupRMQ(json &entry, std::string &dbStrType)
  {
    if (!entry.contains("rmq_config")) {
      THROW(std::runtime_error,
            "JSON db-fields do not contain rmq_config entires")
    }
    auto rmq_entry = entry["rmq_config"];
    int port = getEntry<int>(rmq_entry, "service-port");
    std::string host = getEntry<std::string>(rmq_entry, "service-host");
    std::string rmq_name = getEntry<std::string>(rmq_entry, "rabbitmq-name");
    std::string rmq_pass =
        getEntry<std::string>(rmq_entry, "rabbitmq-password");
    std::string rmq_user = getEntry<std::string>(rmq_entry, "rabbitmq-user");
    std::string rmq_vhost = getEntry<std::string>(rmq_entry, "rabbitmq-vhost");
    std::string rmq_out_queue =
        getEntry<std::string>(rmq_entry, "rabbitmq-outbound-queue");
    std::string exchange =
        getEntry<std::string>(rmq_entry, "rabbitmq-exchange");
    std::string routing_key =
        getEntry<std::string>(rmq_entry, "rabbitmq-routing-key");
    bool update_surrogate = getEntry<bool>(entry, "update_surrogate");

    // We allow connection to RabbitMQ without TLS certificate
    std::string rmq_cert = "";
    if (rmq_entry.contains("rabbitmq-cert"))
      rmq_cert = getEntry<std::string>(rmq_entry, "rabbitmq-cert");

    auto &DB = ams::db::DBManager::getInstance();
    DB.instantiate_rmq_db(port,
                          host,
                          rmq_name,
                          rmq_pass,
                          rmq_user,
                          rmq_vhost,
                          rmq_cert,
                          rmq_out_queue,
                          exchange,
                          routing_key,
                          update_surrogate);
  }

  void parseDatabase(json &jRoot)
  {
    DBG(AMS, "Parsing Data Base Fields")
    if (!jRoot.contains("db")) return;
    auto entry = jRoot["db"];
    if (!entry.contains("dbType"))
      THROW(std::runtime_error,
            "JSON file instantiates db-fields without defining a "
            "\"dbType\" "
            "entry");
    auto dbStrType = entry["dbType"].get<std::string>();
    dbType = ams::db::getDBType(dbStrType);
    switch (dbType) {
      case AMSDBType::AMS_NONE:
        return;
      case AMSDBType::AMS_CSV:
      case AMSDBType::AMS_HDF5:
        setupFSDB(entry, dbStrType);
        break;
      case AMSDBType::AMS_RMQ:
        setupRMQ(entry, dbStrType);
        break;
      case AMSDBType::AMS_REDIS:
        FATAL(AMS, "Cannot connect to REDIS database, missing implementation");
    }
    return;
  }

  std::pair<bool, std::string> setup_loggers()
  {
    const char *ams_logger_level = std::getenv("AMS_LOG_LEVEL");
    const char *ams_logger_dir = std::getenv("AMS_LOG_DIR");
    const char *ams_logger_prefix = std::getenv("AMS_LOG_PREFIX");
    std::string log_fn("");
    std::string log_path("./");

    auto logger = ams::util::Logger::getActiveLogger();
    bool enable_log = false;

    if (ams_logger_level) {
      auto log_lvl = ams::util::getVerbosityLevel(ams_logger_level);
      logger->setLoggingMsgLevel(log_lvl);
      enable_log = true;
    }

    // In the case we specify a directory and we do not specify a file
    // by default we write to a file.
    if (ams_logger_dir && !ams_logger_prefix) {
      ams_logger_prefix = "ams";
    }

    if (ams_logger_prefix) {
      // We are going to redirect stdout to some file
      // By default we store to the current directory
      std::string pattern("");
      std::string log_prefix(ams_logger_prefix);

      if (ams_logger_dir) {
        log_path = std::string(ams_logger_dir);
      }

      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0) {
        FATAL(AMS, "Get hostname returns error");
      }

      int id = 0;
      if (log_prefix.find("<RID>") != std::string::npos) {
        pattern = std::string("<RID>");
        id = get_rank_id();
      } else if (log_prefix.find("<PID>") != std::string::npos) {
        pattern = std::string("<PID>");
        id = getpid();
      }

      // Combine hostname and pid
      std::ostringstream combined;
      combined << "." << hostname << "." << id;

      if (!pattern.empty()) {
        log_path = fs::absolute(log_path).string();
        log_fn =
            std::regex_replace(log_prefix, std::regex(pattern), combined.str());
      } else {
        log_path = fs::absolute(log_path).string();
        log_fn = log_prefix + combined.str();
      }
    }
    logger->initialize_std_io_err(enable_log, log_path, log_fn);

    return std::make_pair(enable_log, log_path);
  }

public:
  AMSWrap() : memManager(ams::ResourceManager::getInstance())
  {
    auto log_stats = setup_loggers();
    DBG(AMS,
        "Enable Log %d stored under %s",
        log_stats.first,
        log_stats.second.c_str())
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
            registered_models[model->second].second.SPath.c_str());
    }
    registered_models.push_back(std::make_pair(std::string(domain_name),
                                               AMSAbstractModel(uq_policy,
                                                                surrogate_path,
                                                                uq_path,
                                                                db_label,
                                                                threshold,
                                                                num_clusters)));
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

  std::pair<std::string, AMSAbstractModel> &get_model(int index)
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
    ams::util::close();
  }
};

static AMSWrap _amsWrap;

void _AMSExecute(AMSExecutor executor,
                 void *probDescr,
                 const int numElements,
                 const void **input_data,
                 void **output_data,
                 int inputDim,
                 int outputDim)
{
  int64_t index = static_cast<int64_t>(executor);
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
                  outputDim);
  } else if (currExec.first == AMSDType::AMS_SINGLE) {
    ams::AMSWorkflow<float> *sWF =
        reinterpret_cast<ams::AMSWorkflow<float> *>(currExec.second);
    sWF->evaluate(probDescr,
                  numElements,
                  reinterpret_cast<const float **>(input_data),
                  reinterpret_cast<float **>(output_data),
                  inputDim,
                  outputDim);
  } else {
    throw std::invalid_argument("Data type is not supported by AMSLib!");
    return;
  }
}

template <typename FPTypeValue>
ams::AMSWorkflow<FPTypeValue> *_AMSCreateExecutor(AMSCAbstrModel model,
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

  auto &model_descr = _amsWrap.get_model(model);

  ams::AMSWorkflow<FPTypeValue> *WF =
      new ams::AMSWorkflow<FPTypeValue>(call_back,
                                        model_descr.second.UQPath,
                                        model_descr.second.SPath,
                                        model_descr.first,
                                        model_descr.second.DBLabel,
                                        model_descr.second.DebugDB,
                                        resource_type,
                                        model_descr.second.threshold,
                                        model_descr.second.uqPolicy,
                                        model_descr.second.nClusters,
                                        process_id,
                                        world_size);
  return WF;
}

template <typename FPTypeValue>
AMSExecutor _AMSRegisterExecutor(AMSDType data_type,
                                 ams::AMSWorkflow<FPTypeValue> *workflow)
{
  _amsWrap.executors.push_back(
      std::make_pair(data_type, static_cast<void *>(workflow)));
  return static_cast<AMSExecutor>(_amsWrap.executors.size()) - 1L;
}


#ifdef __cplusplus
extern "C" {
#endif

AMSExecutor AMSCreateExecutor(AMSCAbstrModel model,
                              AMSDType data_type,
                              AMSResourceType resource_type,
                              AMSPhysicFn call_back,
                              int process_id,
                              int world_size)
{
  if (data_type == AMSDType::AMS_DOUBLE) {
    auto *dWF = _AMSCreateExecutor<double>(
        model, data_type, resource_type, call_back, process_id, world_size);
    return _AMSRegisterExecutor(data_type, dWF);

  } else if (data_type == AMSDType::AMS_SINGLE) {
    auto *sWF = _AMSCreateExecutor<float>(
        model, data_type, resource_type, call_back, process_id, world_size);
    return _AMSRegisterExecutor(data_type, sWF);
  } else {
    throw std::invalid_argument("Data type is not supported by AMSLib!");
    return static_cast<AMSExecutor>(-1);
  }
}

#ifdef __ENABLE_MPI__
AMSExecutor AMSCreateDistributedExecutor(AMSCAbstrModel model,
                                         AMSDType data_type,
                                         AMSResourceType resource_type,
                                         AMSPhysicFn call_back,
                                         MPI_Comm Comm,
                                         int process_id,
                                         int world_size)
{
  if (data_type == AMSDType::AMS_DOUBLE) {
    auto *dWF = _AMSCreateExecutor<double>(
        model, data_type, resource_type, call_back, process_id, world_size);
    dWF->set_communicator(Comm);
    return _AMSRegisterExecutor(data_type, dWF);

  } else if (data_type == AMSDType::AMS_SINGLE) {
    auto *sWF = _AMSCreateExecutor<float>(
        model, data_type, resource_type, call_back, process_id, world_size);
    sWF->set_communicator(Comm);
    return _AMSRegisterExecutor(data_type, sWF);
  } else {
    throw std::invalid_argument("Data type is not supported by AMSLib!");
    return static_cast<AMSExecutor>(-1);
  }
}
#endif

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
  int64_t index = static_cast<int64_t>(executor);
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


const char *AMSGetAllocatorName(AMSResourceType device)
{
  auto &rm = ams::ResourceManager::getInstance();
  return rm.getAllocatorName(device).c_str();
}

void AMSSetAllocator(AMSResourceType resource, const char *alloc_name)
{
  auto &rm = ams::ResourceManager::getInstance();
  std::string alloc(alloc_name);
  rm.setAllocator(alloc, resource);
}

AMSCAbstrModel AMSRegisterAbstractModel(const char *domain_name,
                                        AMSUQPolicy uq_policy,
                                        double threshold,
                                        const char *surrogate_path,
                                        const char *uq_path,
                                        const char *db_label,
                                        int num_clusters)
{
  auto id = _amsWrap.get_model_index(domain_name);
  if (id == -1) {
    id = _amsWrap.register_model(domain_name,
                                 uq_policy,
                                 threshold,
                                 surrogate_path,
                                 uq_path,
                                 db_label,
                                 num_clusters);
  }

  return id;
}


AMSCAbstrModel AMSQueryModel(const char *domain_model)
{
  return _amsWrap.get_model_index(domain_model);
}

void AMSConfigureFSDatabase(AMSDBType db_type, const char *db_path)
{
  auto &db_instance = ams::db::DBManager::getInstance();
  db_instance.instantiate_fs_db(db_type, std::string(db_path));
}

#ifdef __cplusplus
}
#endif
