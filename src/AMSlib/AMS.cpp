/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "AMS.h"

#include <vector>

#include "wf/resource_manager.hpp"
#include "wf/basedb.hpp"
#include "wf/workflow.hpp"
#include "wf/debug.h"

struct AMSWrap {
  std::vector<std::pair<AMSDType, void *>> executors;
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
    auto& rm = ams::ResourceManager::getInstance();
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

void AMSDestroyExecutor(AMSExecutor executor) {
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
  auto& rm = ams::ResourceManager::getInstance();
  return std::move(rm.getAllocatorName(device)).c_str();
}

void AMSSetAllocator(AMSResourceType resource, const char *alloc_name)
{
  auto& rm = ams::ResourceManager::getInstance();
  rm.setAllocator(std::string(alloc_name), resource);
}

#ifdef __cplusplus
}
#endif
