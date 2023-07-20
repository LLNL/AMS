/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <vector>

#include "AMS.h"
#include "wf/workflow.hpp"

struct AMSWrap{
  std::vector<std::pair<AMSDType, void *>> executors;
  ~AMSWrap() {
    for ( auto E : executors ){
      if ( E.second != nullptr ){
        if ( E.first == AMSDType::Double ){
          delete reinterpret_cast<ams::AMSWorkflow<double> *> (E.second);
        } else{
          delete reinterpret_cast<ams::AMSWorkflow<float> *> (E.second);
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
  if (config.dType == Double) {
    ams::AMSWorkflow<double> *dWF =
        new ams::AMSWorkflow<double>(config.cBack,
                                     config.UQPath,
                                     config.SPath,
                                     config.DBPath,
                                     config.dbType,
                                     config.device == AMSResourceType::HOST,
                                     config.threshold,
                                     config.uqPolicy,
                                     config.nClusters,
                                     config.pId,
                                     config.wSize,
                                     config.ePolicy);

    _amsWrap.executors.push_back(std::make_pair(config.dType, static_cast<void *>(dWF)));
    return reinterpret_cast<AMSExecutor>(_amsWrap.executors.size() - 1L);
  } else if (config.dType == AMSDType::Single) {
    ams::AMSWorkflow<float> *sWF =
        new ams::AMSWorkflow<float>(config.cBack,
                                    config.UQPath,
                                    config.SPath,
                                    config.DBPath,
                                    config.dbType,
                                    config.device == AMSResourceType::HOST,
                                    static_cast<float>(config.threshold),
                                    config.uqPolicy,
                                    config.nClusters,
                                    config.pId,
                                    config.wSize,
                                    config.ePolicy);
    _amsWrap.executors.push_back(std::make_pair(config.dType, static_cast<void *>(sWF)));

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
  if (device == AMSResourceType::HOST) {
    return ams::ResourceManager::getHostAllocatorName();
  } else if (device == AMSResourceType::DEVICE) {
    return ams::ResourceManager::getDeviceAllocatorName();
  }

  throw std::runtime_error("requested Device Allocator does not exist");

  return nullptr;
}

void AMSSetupAllocator(const AMSResourceType Resource)
{
  ams::ResourceManager::setup(Resource);
}

void AMSResourceInfo() { ams::ResourceManager::list_allocators(); }

int AMSGetLocationId(void *ptr)
{
  return ams::ResourceManager::getDataAllocationId(ptr);
}

void AMSSetDefaultAllocator(const AMSResourceType device)
{
  ams::ResourceManager::setDefaultDataAllocator(device);
}

#ifdef __cplusplus
}
#endif
