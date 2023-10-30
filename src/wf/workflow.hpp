/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_WORKFLOW_HPP__
#define __AMS_WORKFLOW_HPP__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "AMS.h"
#include "ml/uq.hpp"
#include "ml/hdcache.hpp"
#include "ml/surrogate.hpp"
#include "resource_manager.hpp"
#include "wf/basedb.hpp"

#ifdef __ENABLE_MPI__
#include "wf/redist_load.hpp"
#endif

#include "wf/debug.h"

//! ----------------------------------------------------------------------------
//! AMS Workflow class
//! the purpose of this class is to expose an "evaluate" function
//!     which has the same interface as the physics evaluation
//! the intention is that we can easily switch in and out AMS wf in app code
//! ----------------------------------------------------------------------------
namespace ams
{
template <typename FPTypeValue>
class AMSWorkflow
{

  static_assert(std::is_floating_point<FPTypeValue>::value,
                "HDCache supports floating-point values (floats, doubles, and "
                "long doubles) only!");

  using data_handler = ams::DataHandler<FPTypeValue>;

  /** @brief The application call back to perform the original SPMD physics
   * execution */
  AMSPhysicFn AppCall;

  /** @brief The module that performs uncertainty quantification (UQ) */
  std::shared_ptr<HDCache<FPTypeValue>> hdcache;

  /** The metric/type of UQ we will use to select between physics and ml computations **/
  const AMSUQPolicy uqPolicy = AMSUQPolicy::FAISS_Mean;

  /** The Number of clusters we will use to compute FAISS UQ  **/
  const int nClusters = 10;

  /** @brief The torch surrogate model to replace the original physics function
   */
  std::shared_ptr<SurrogateModel<FPTypeValue>> surrogate;

  /** @brief The database to store data for which we cannot apply the current
   * model */
  std::shared_ptr<BaseDB<FPTypeValue>> DB;

  /** @brief The type of the database we will use (HDF5, CSV, etc) */
  AMSDBType dbType = AMSDBType::None;

  /** @brief The process id. For MPI runs this is the rank */
  const int rId;

  /** @brief The total number of processes participating in the simulation
   * (world_size for MPI) */
  int wSize;

  /** @brief  Location of the original application data  (CPU or GPU) */
  AMSResourceType appDataLoc;

  /** @brief execution policy of the distributed system. Load balance or not. */
  const AMSExecPolicy ePolicy;

  /** \brief Store the data in the database and copies
   * data from the GPU to the CPU and then to the database.
   * To store GPU resident data we use a 1MB of "pinned"
   * memory as a buffer
   * @param[in] num_elements Number of elements of each 1-D vector
   * @param[in] inputs vector to 1-D vectors storing num_elements
   * items to be stored in the database
   * @param[in] outputs vector to 1-D vectors storing num_elements
   * items to be stored in the database
   */
  void Store(size_t num_elements,
             std::vector<FPTypeValue *> &inputs,
             std::vector<FPTypeValue *> &outputs)
  {
    // 1 MB of buffer size;
    // TODO: Fix magic number
    // TODO: This is likely not efficient for RabbitMQ backend at scale
    //       We could just linearize the whole input+output and do one send (or two) per cycle
    static const long bSize = 1 * 1024 * 1024;
    const int numIn = inputs.size();
    const int numOut = outputs.size();

    // No database, so just de-allocate and return
    if (!DB) return;

    std::vector<FPTypeValue *> hInputs, hOutputs;

    if (appDataLoc == AMSResourceType::HOST ) return DB->store(num_elements, inputs, outputs);

    // Compute number of elements that fit inside the buffer
    size_t bElements = bSize / sizeof(FPTypeValue);
    FPTypeValue *pPtr =
        ams::ResourceManager::allocate<FPTypeValue>(bElements,
                                                    AMSResourceType::PINNED);
    // Total inner vector dimensions (inputs and outputs)
    size_t totalDims = inputs.size() + outputs.size();
    // Compute number of elements of each outer dimension that fit in buffer
    size_t elPerDim = static_cast<int>(floor(bElements / totalDims));

    for (int i = 0; i < inputs.size(); i++)
      hInputs.push_back(&pPtr[i * elPerDim]);

    for (int i = 0; i < outputs.size(); i++)
      hOutputs.push_back(&pPtr[(i + inputs.size()) * elPerDim]);

    // Iterate over all chunks
    for (int i = 0; i < num_elements; i += elPerDim) {
      size_t actualElems = std::min(elPerDim, num_elements - i);
      // Copy input data to host
      for (int k = 0; k < numIn; k++) {
        ams::ResourceManager::copy(&inputs[k][i],
                                   hInputs[k],
                                   actualElems * sizeof(FPTypeValue));
      }

      // Copy output data to host
      for (int k = 0; k < numIn; k++) {
        ams::ResourceManager::copy(&outputs[k][i],
                                   hOutputs[k],
                                   actualElems * sizeof(FPTypeValue));
      }

      // Store to database
      DB->store(actualElems, hInputs, hOutputs);
    }
    ams::ResourceManager::deallocate(pPtr, AMSResourceType::PINNED);

    return;
  }

public:
  AMSWorkflow()
      : AppCall(nullptr),
        hdcache(nullptr),
        surrogate(nullptr),
        DB(nullptr),
        dbType(AMSDBType::None),
        appDataLoc(AMSResourceType::HOST),
        ePolicy(AMSExecPolicy::UBALANCED)
  {

#ifdef __ENABLE_DB__
    DB = createDB<FPTypeValue>("miniApp_data.txt", dbType, 0);
    CFATAL(WORKFLOW, !DB, "Cannot create database");
#endif
  }

  AMSWorkflow(AMSPhysicFn _AppCall,
              char *uq_path,
              char *surrogate_path,
              char *db_path,
              const AMSDBType dbType,
              AMSResourceType appDataLoc,
              FPTypeValue threshold,
              const AMSUQPolicy uqPolicy,
              const int nClusters,
              int _pId = 0,
              int _wSize = 1,
              AMSExecPolicy policy = AMSExecPolicy::UBALANCED)
      : AppCall(_AppCall),
        dbType(dbType),
        rId(_pId),
        wSize(_wSize),
        appDataLoc(appDataLoc),
        uqPolicy(uqPolicy),
        ePolicy(policy)
  {

    surrogate = nullptr;
    if (surrogate_path) {
      bool is_DeltaUQ = ((uqPolicy == AMSUQPolicy::DeltaUQ_Max ||
                          uqPolicy == AMSUQPolicy::DeltaUQ_Mean)
                             ? true
                             : false);
      surrogate = SurrogateModel<FPTypeValue>::getInstance(
          surrogate_path,
          appDataLoc,
          is_DeltaUQ);
    }

    UQ<FPTypeValue>::setThreshold(threshold);
    // TODO: Fix magic number. 10 represents the number of neighbours I am
    // looking at.
    if (uq_path)
      hdcache = HDCache<FPTypeValue>::getInstance(
          uq_path, appDataLoc, uqPolicy, nClusters, threshold);
    else
      // This is a random hdcache returning true %threshold queries
      hdcache = HDCache<FPTypeValue>::getInstance(appDataLoc, threshold);

    DB = nullptr;
    if (db_path) {
      DBG(Workflow, "Creating Database");
      DB = getDB<FPTypeValue>(db_path, dbType, rId);
    }
  }

  void set_physics(AMSPhysicFn _AppCall) { AppCall = _AppCall; }

  void set_surrogate(SurrogateModel<FPTypeValue> *_surrogate)
  {
    surrogate = _surrogate;
  }

  void set_hdcache(HDCache<FPTypeValue> *_hdcache) { hdcache = _hdcache; }

  ~AMSWorkflow() { DBG(Workflow, "Destroying Workflow Handler"); }


  /** @brief This is the main entry point of AMSLib and replaces the original
   * execution path of the application.
   * @param[in] probDescr an opaque type that will be forwarded to the
   * application upcall
   * @param[in] totalElements the total number of elements to apply the SPMD
   * function on
   * @param[in] inputs the inputs of the computation.
   * @param[out] outputs the computed outputs.
   * @param[in] Comm The MPI Communicatotor for all ranks participating in the
   * SPMD execution.
   *
   * @details The function corresponds to the main driver of the AMSLib.
   * Assuming an original 'foo' function void foo ( void *cls, int numElements,
   * void **inputs, void **outputs){ parallel_for(I : numElements){
   *       cls->physics(inputs[0][I], outputs[0][I]);
   *    }
   * }
   *
   * The AMS transformation would functionaly look like this:
   * void AMSfoo ( void *cls, int numElements, void **inputs, void **outputs){
   *    parallel_for(I : numElements){
   *       if ( UQ (I) ){
   *          Surrogate(inputs[0][I], outputs[0][I])
   *       }
   *       else{
   *        cls->physics(inputs[0][I], outputs[0][I]);
   *        DB->Store(inputs[0][I], outputs[0][I]);
   *       }
   *    }
   * }
   *
   * Yet, AMS assumes a SPMD physics function (in the example cls->physics).
   * Therefore, the AMS transformation is taking place at the level of the SPMD
   * execution. The following transformation is equivalent void AMSfoo( void
   * *cls, int numElements, void **inputs, void **outputs){ predicates =
   * UQ(inputs, numElements); modelInputs, physicsInputs = partition(predicates,
   * inputs); modelOuputs, physicsOutputs = partition(predicates, output);
   *    foo(cls, physicsInputs.size(), physicsInputs, physicsOutputs);
   *    surrogate(modelInputs, modelOuputs, modelOuputs.size());
   *    DB->Store(physicsInputs, physicsOutputs);
   *    concatenate(outptuts, modelOuputs, predicate);
   * }
   *
   * This transformation can exploit the parallel nature of all the required
   * steps.
   */
  void evaluate(void *probDescr,
                const int totalElements,
                const FPTypeValue **inputs,
                FPTypeValue **outputs,
                int inputDim,
                int outputDim,
                MPI_Comm Comm = nullptr)
  {
    CDEBUG(Workflow,
           rId == 0,
           "Entering Evaluate "
           "with problem dimensions [(%d, %d, %d, %d)]",
           totalElements,
           inputDim,
           totalElements,
           outputDim);
    // To move around the inputs, outputs we bundle them as std::vectors
    std::vector<const FPTypeValue *> origInputs(inputs, inputs + inputDim);
    std::vector<FPTypeValue *> origOutputs(outputs, outputs + outputDim);

    REPORT_MEM_USAGE(Workflow, "Start")

    if (!surrogate) {
      FPTypeValue **tmpInputs = const_cast<FPTypeValue **>(inputs);

      std::vector<FPTypeValue *> tmpIn(tmpInputs, tmpInputs + inputDim);
      DBG(Workflow, "No-Model, I am calling Physics code (for all data)");
      AppCall(probDescr,
              totalElements,
              reinterpret_cast<const void **>(origInputs.data()),
              reinterpret_cast<void **>(origOutputs.data()));
      if (DB) {
        CALIPER(CALI_MARK_BEGIN("DBSTORE");)
        Store(totalElements, tmpIn, origOutputs);
        CALIPER(CALI_MARK_END("DBSTORE");)
      }
      return;
    }
    // The predicate with which we will split the data on a later step
    bool *p_ml_acceptable = ams::ResourceManager::allocate<bool>(totalElements, appDataLoc);

    // -------------------------------------------------------------
    // STEP 1: call the hdcache to look at input uncertainties
    //         to decide if making a ML inference makes sense
    // -------------------------------------------------------------
    CALIPER(CALI_MARK_BEGIN("UQ_MODULE");)
    UQ<FPTypeValue>::evaluate(uqPolicy,
                 totalElements,
                 origInputs,
                 origOutputs,
                 hdcache,
                 surrogate,
                 p_ml_acceptable);
    CALIPER(CALI_MARK_END("UQ_MODULE");)

    DBG(Workflow, "Computed Predicates")

    // Pointer values which store input data values
    // to be computed using the eos function.
    std::vector<FPTypeValue *> packedInputs;

    for (int i = 0; i < inputDim; i++) {
      packedInputs.emplace_back(
          ams::ResourceManager::allocate<FPTypeValue>(totalElements, appDataLoc));
    }

    DBG(Workflow, "Allocated input resources")

    bool *predicate = p_ml_acceptable;

    // -----------------------------------------------------------------
    // STEP 3: call physics module only where d_dense_need_phys = true
    // -----------------------------------------------------------------
    // ---- 3a: we need to pack the sparse data based on the uq flag
    const long packedElements =
        data_handler::pack(appDataLoc, predicate, totalElements, origInputs, packedInputs);

    // Pointer values which store output data values
    // to be computed using the eos function.
    std::vector<FPTypeValue *> packedOutputs;
    for (int i = 0; i < outputDim; i++) {
      packedOutputs.emplace_back(
          ams::ResourceManager::allocate<FPTypeValue>(packedElements, appDataLoc));
    }

    {
      void **iPtr = reinterpret_cast<void **>(packedInputs.data());
      void **oPtr = reinterpret_cast<void **>(packedOutputs.data());
      long lbElements = packedElements;

#ifdef __ENABLE_MPI__
      CALIPER(CALI_MARK_BEGIN("LOAD BALANCE MODULE");)
      AMSLoadBalancer<FPTypeValue> lBalancer(
          rId, wSize, packedElements, Comm, inputDim, outputDim, appDataLoc);
      if (ePolicy == AMSExecPolicy::BALANCED && Comm) {
        lBalancer.scatterInputs(packedInputs, appDataLoc);
        iPtr = reinterpret_cast<void **>(lBalancer.inputs());
        oPtr = reinterpret_cast<void **>(lBalancer.outputs());
        lbElements = lBalancer.getBalancedSize();
      }
      CALIPER(CALI_MARK_END("LOAD BALANCE MODULE");)
#endif

      // ---- 3b: call the physics module and store in the data base
      if (packedElements > 0) {
        CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
        AppCall(probDescr, lbElements, iPtr, oPtr);
        CALIPER(CALI_MARK_END("PHYSICS MODULE");)
      }

#ifdef __ENABLE_MPI__
      CALIPER(CALI_MARK_BEGIN("LOAD BALANCE MODULE");)
      if (ePolicy == AMSExecPolicy::BALANCED && Comm) {
        lBalancer.gatherOutputs(packedOutputs, appDataLoc);
      }
      CALIPER(CALI_MARK_END("LOAD BALANCE MODULE");)
#endif
    }

    // ---- 3c: unpack the data
    data_handler::unpack(appDataLoc, predicate, totalElements, packedOutputs, origOutputs);

    DBG(Workflow, "Finished physics evaluation")

    if (DB) {
      CALIPER(CALI_MARK_BEGIN("DBSTORE");)
      DBG(Workflow,
          "Storing data (#elements = %d) to database",
          packedElements);
      Store(packedElements, packedInputs, packedOutputs);
      CALIPER(CALI_MARK_END("DBSTORE");)
    }

    // -----------------------------------------------------------------
    // Deallocate temporal data
    // -----------------------------------------------------------------
    for (int i = 0; i < inputDim; i++)
      ams::ResourceManager::deallocate(packedInputs[i], appDataLoc);
    for (int i = 0; i < outputDim; i++)
      ams::ResourceManager::deallocate(packedOutputs[i], appDataLoc);

    ams::ResourceManager::deallocate(p_ml_acceptable, appDataLoc);

    DBG(Workflow, "Finished AMSExecution")
    CINFO(Workflow,
          rId == 0,
          "Computed %ld "
          "using physics out of the %ld items (%.2f)",
          packedElements,
          totalElements,
          (float)(packedElements) / float(totalElements))

    REPORT_MEM_USAGE(Workflow, "End")
  }
};


}  // namespace ams
#endif
