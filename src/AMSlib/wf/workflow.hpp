/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_WORKFLOW_HPP__
#define __AMS_WORKFLOW_HPP__

#ifdef __AMS_ENABLE_CALIPER__
#include <caliper/cali_macros.h>
#endif

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "AMS.h"
#include "ml/uq.hpp"
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
  std::unique_ptr<UQ<FPTypeValue>> UQModel;

  /** The metric/type of UQ we will use to select between physics and ml computations **/
  const AMSUQPolicy uqPolicy = AMSUQPolicy::AMSUQPolicy_END;

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
    auto &rm = ams::ResourceManager::getInstance();

    // No database, so just de-allocate and return
    if (!DB) return;

    std::vector<FPTypeValue *> hInputs, hOutputs;

    if (appDataLoc == AMSResourceType::HOST)
      return DB->store(num_elements, inputs, outputs);

    // Compute number of elements that fit inside the buffer
    size_t bElements = bSize / sizeof(FPTypeValue);
    FPTypeValue *pPtr =
        rm.allocate<FPTypeValue>(bElements, AMSResourceType::PINNED);
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
        rm.copy(&inputs[k][i], hInputs[k], actualElems * sizeof(FPTypeValue));
      }

      // Copy output data to host
      for (int k = 0; k < numIn; k++) {
        rm.copy(&outputs[k][i], hOutputs[k], actualElems * sizeof(FPTypeValue));
      }

      // Store to database
      DB->store(actualElems, hInputs, hOutputs);
    }
    rm.deallocate(pPtr, AMSResourceType::PINNED);

    return;
  }

public:
  AMSWorkflow()
      : AppCall(nullptr),
        DB(nullptr),
        dbType(AMSDBType::None),
        appDataLoc(AMSResourceType::HOST),
        ePolicy(AMSExecPolicy::UBALANCED)
  {
#ifdef __ENABLE_DB__
    auto &dbm = ams::DBManager<FPTypeValue>::getInstance();
    DB = dbm.createDB("miniApp_data.txt", dbType, 0);
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
    DB = nullptr;
    if (db_path) {
      DBG(Workflow, "Creating Database");
      auto &dbm = ams::DBManager<FPTypeValue>::getInstance();
      DB = dbm.getDB(db_path, dbType, rId);
    }

    UQModel = std::make_unique<UQ<FPTypeValue>>(
        appDataLoc, uqPolicy, uq_path, nClusters, surrogate_path, threshold);
  }

  void set_physics(AMSPhysicFn _AppCall) { AppCall = _AppCall; }

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
    CALIPER(CALI_MARK_BEGIN("AMSEvaluate");)

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
    auto &rm = ams::ResourceManager::getInstance();

    REPORT_MEM_USAGE(Workflow, "Start")

    if (!UQModel->hasSurrogate()) {
      FPTypeValue **tmpInputs = const_cast<FPTypeValue **>(inputs);

      std::vector<FPTypeValue *> tmpIn(tmpInputs, tmpInputs + inputDim);
      DBG(Workflow, "No-Model, I am calling Physics code (for all data)");
      CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
      AppCall(probDescr,
              totalElements,
              reinterpret_cast<const void **>(origInputs.data()),
              reinterpret_cast<void **>(origOutputs.data()));
      CALIPER(CALI_MARK_END("PHYSICS MODULE");)
      if (DB) {
        CALIPER(CALI_MARK_BEGIN("DBSTORE");)
        Store(totalElements, tmpIn, origOutputs);
        CALIPER(CALI_MARK_END("DBSTORE");)
      }
      CALIPER(CALI_MARK_END("AMSEvaluate");)
      return;
    }

    if (DB && DB->updateModel()) {
      UQModel->updateModel("");
    }

    // The predicate with which we will split the data on a later step
    bool *p_ml_acceptable = rm.allocate<bool>(totalElements, appDataLoc);

    // -------------------------------------------------------------
    // STEP 1: call the UQ module to look at input uncertainties
    //         to decide if making a ML inference makes sense
    // -------------------------------------------------------------
    CALIPER(CALI_MARK_BEGIN("UQ_MODULE");)
    UQModel->evaluate(totalElements, origInputs, origOutputs, p_ml_acceptable);
    CALIPER(CALI_MARK_END("UQ_MODULE");)

    DBG(Workflow, "Computed Predicates")

    // Pointer values which store input data values
    // to be computed using the eos function.
    std::vector<FPTypeValue *> packedInputs;

    for (int i = 0; i < inputDim; i++) {
      packedInputs.emplace_back(
          rm.allocate<FPTypeValue>(totalElements, appDataLoc));
    }

    DBG(Workflow, "Allocated input resources")

    bool *predicate = p_ml_acceptable;

    // -----------------------------------------------------------------
    // STEP 3: call physics module only where predicate = false
    // -----------------------------------------------------------------
    // ---- 3a: we need to pack the sparse data based on the uq flag
    CALIPER(CALI_MARK_BEGIN("PACK");)
    const long packedElements = data_handler::pack(
        appDataLoc, predicate, totalElements, origInputs, packedInputs);
    CALIPER(CALI_MARK_END("PACK");)

    // Pointer values which store output data values
    // to be computed using the eos function.
    std::vector<FPTypeValue *> packedOutputs;
    for (int i = 0; i < outputDim; i++) {
      packedOutputs.emplace_back(
          rm.allocate<FPTypeValue>(packedElements, appDataLoc));
    }

    {
      void **iPtr = reinterpret_cast<void **>(packedInputs.data());
      void **oPtr = reinterpret_cast<void **>(packedOutputs.data());
      long lbElements = packedElements;

#ifdef __ENABLE_MPI__
      CALIPER(CALI_MARK_BEGIN("LOAD BALANCE MODULE");)
      AMSLoadBalancer<FPTypeValue> lBalancer(
          rId, wSize, packedElements, Comm);
      if (ePolicy == AMSExecPolicy::BALANCED && Comm) {
        lBalancer.init(inputDim, outputDim, appDataLoc);
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
    CALIPER(CALI_MARK_BEGIN("UNPACK");)
    data_handler::unpack(
        appDataLoc, predicate, totalElements, packedOutputs, origOutputs);
    CALIPER(CALI_MARK_END("UNPACK");)

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
      rm.deallocate(packedInputs[i], appDataLoc);
    for (int i = 0; i < outputDim; i++)
      rm.deallocate(packedOutputs[i], appDataLoc);

    rm.deallocate(p_ml_acceptable, appDataLoc);

    DBG(Workflow, "Finished AMSExecution")
    CINFO(Workflow,
          rId == 0,
          "Computed %ld "
          "using physics out of the %ld items (%.2f)",
          packedElements,
          totalElements,
          (float)(packedElements) / float(totalElements))

    REPORT_MEM_USAGE(Workflow, "End")
    CALIPER(CALI_MARK_END("AMSEvaluate");)
  }
};


}  // namespace ams
#endif
