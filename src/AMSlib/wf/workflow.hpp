/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_WORKFLOW_HPP__
#define __AMS_WORKFLOW_HPP__

#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>

#include <memory>
#include <stdexcept>

#include "AMS.h"
#include "ArrayRef.hpp"
#include "SmallVector.hpp"
#include "interface.hpp"
#include "macro.h"
#include "ml/surrogate.hpp"
#include "resource_manager.hpp"
#include "utils.hpp"
#include "wf/basedb.hpp"
#include "wf/debug.h"

//! ----------------------------------------------------------------------------
//! AMS Workflow class
//! the purpose of this class is to expose an "evaluate" function
//!     which has the same interface as the physics evaluation
//! the intention is that we can easily switch in and out AMS wf in app code
//! ----------------------------------------------------------------------------
namespace ams
{
class AMSWorkflow
{
  // Friend declarations for graph surrogate execution access to private MLModel
  friend bool tryGraphSurrogate(AMSWorkflow*,
                                const AMSHomogeneousGraph&,
                                AMSHomogeneousGraphFields&);
  friend bool tryGraphSurrogate(AMSWorkflow*,
                                const AMSHeterogeneousGraph&,
                                AMSHeterogeneousGraphFields&);

  /** @brief A string identifier describing the domain-model being solved. */
  std::string domainName;

  /** @brief The module that performs uncertainty quantification (UQ) */
  std::shared_ptr<SurrogateModel> MLModel;

  /** @brief The database to store data for which we cannot apply the current
     * model */
  std::shared_ptr<ams::db::BaseDB> DB;

  /** @brief The process id. For MPI runs this is the rank */
  const int rId;

  /** @brief The total number of processes participating in the simulation
     * (world_size for MPI) */
  int wSize;

  /** @brief execution policy of the distributed system. Load balance or not. */
  AMSExecPolicy ePolicy;


  /** @brief The maximum distance of the predicate for a sample prediction to be considered as valid **/
  const float threshold;

  /** @brief whether we should store data **/
  bool storeData;

#ifdef __AMS_ENABLE_MPI__
  /** @brief MPI Communicator for all ranks that call collectively the evaluate function **/
  MPI_Comm comm;
#endif

  /** @brief Is the evaluate a distributed execution **/
  bool isDistributed;

  void storeComputedData(ArrayRef<torch::Tensor> Ins,
                         ArrayRef<torch::Tensor> InOutsBefore,
                         ArrayRef<torch::Tensor> Outs,
                         ArrayRef<torch::Tensor> InOutsAfter)
  {
    CALIPER(CALI_MARK_BEGIN("DBSTORE");)
    SmallVector<torch::Tensor> StoreInputTensors(Ins.begin(), Ins.end());
    SmallVector<torch::Tensor> StoreOutputTensors(Outs.begin(), Outs.end());
    for (auto Tensor : InOutsBefore)
      StoreInputTensors.push_back(Tensor);
    for (auto Tensor : InOutsAfter) {
      StoreOutputTensors.push_back(Tensor);
    }

    AMS_DBG(Workflow,
            "Storing data (#elements = {}) to database",
            StoreInputTensors[0].sizes()[0]);
    DB->store(StoreInputTensors, StoreOutputTensors);
    CALIPER(CALI_MARK_END("DBSTORE");)
  }

  void storeGraphData(const ams::AMSHomogeneousGraph& graph,
                      const ams::AMSHomogeneousGraphFields& outputs)
  {
    if (!DB) {
      AMS_WARNING(Workflow,
                  "Cannot store graph data: database not initialized");
      return;
    }

    try {
      DB->store(graph, outputs);
      AMS_DBG(Workflow, "Successfully stored homogeneous graph data");
    } catch (const std::exception& e) {
      AMS_WARNING(Workflow,
                  "Failed to store homogeneous graph data: {}",
                  e.what());
    }
  }

  void storeGraphData(const ams::AMSHeterogeneousGraph& graph,
                      const ams::AMSHeterogeneousGraphFields& outputs)
  {
    if (!DB) {
      AMS_WARNING(Workflow,
                  "Cannot store graph data: database not initialized");
      return;
    }

    try {
      DB->store(graph, outputs);
      AMS_DBG(Workflow, "Successfully stored heterogeneous graph data");
    } catch (const std::exception& e) {
      AMS_WARNING(Workflow,
                  "Failed to store heterogeneous graph data: {}",
                  e.what());
    }
  }

  /** \brief Check if we can perform a surrogate model update.
     *  AMS can update surrogate model only when all MPI ranks have received 
     * the latest model from RabbitMQ.
     * @return True if surrogate model can be updated
     */
  bool updateModel()
  {
    if (!DB || !DB->allowModelUpdate()) return false;
    bool local = DB->updateModel();
#ifdef __AMS_ENABLE_MPI__
    bool global = false;
    MPI_Allreduce(&local, &global, 1, MPI_CXX_BOOL, MPI_LAND, comm);
    return global;
#else
    return local;
#endif
  }

public:
  AMSWorkflow(std::string& surrogate_path,
              std::string& domain_name,
              float threshold,
              int _pId = 0,
              int _wSize = 1,
              bool store_data = true)
      : domainName(domain_name),
        rId(_pId),
        wSize(_wSize),
        storeData(store_data),
#ifdef __AMS_ENABLE_MPI__
        comm(MPI_COMM_NULL),
#endif
        threshold(threshold),
        ePolicy(AMSExecPolicy::AMS_UBALANCED)
  {
    DB = nullptr;
    auto& dbm = ams::db::DBManager::getInstance();

    if (storeData) DB = dbm.getDB(domainName, rId);
    MLModel = nullptr;
    if (!surrogate_path.empty())
      MLModel = SurrogateModel::getInstance(surrogate_path);
  }

  std::string getDBFilename() const
  {
    if (!DB) return "";
    return DB->getFilename();
  }


#ifdef __AMS_ENABLE_MPI__
  void set_communicator(MPI_Comm communicator) { comm = communicator; }
#endif

  void set_exec_policy(AMSExecPolicy policy) { ePolicy = policy; }

  bool should_load_balance() const
  {
#ifdef __AMS_ENABLE_MPI__
    return (comm != MPI_COMM_NULL && ePolicy == AMSExecPolicy::AMS_BALANCED);
#else
    return false;
#endif
  }


  static SmallVector<torch::Tensor> subSelectTensors(
      ArrayRef<torch::Tensor> Tensors,
      torch::Tensor& Mask)
  {
    SmallVector<torch::Tensor> NewVector;
    for (auto O : Tensors) {
      NewVector.push_back(O.index({Mask}).view({-1, O.sizes()[O.dim() - 1]}));
    }
    return NewVector;
  }

  static void ScatterPhysicOutputsToOrigDomain(
      ArrayRef<torch::Tensor> computedDomain,
      torch::Tensor& Predicate,
      MutableArrayRef<torch::Tensor> entireDomain)
  {
    if (computedDomain.size() != entireDomain.size()) {
      throw std::runtime_error(
          "Expecting equal sized tensors when composing Original and domain "
          "memories\n");
    }
    for (int i = 0; i < computedDomain.size(); i++) {
      auto indexed_shape = computedDomain[i].sizes();
      entireDomain[i].index_put_({Predicate},
                                 computedDomain[i].view(indexed_shape));
    }
  }


  static int MLDomainToApplication(torch::Tensor Src,
                                   MutableArrayRef<torch::Tensor> Dest,
                                   torch::Tensor Predicate,
                                   int offset)
  {
    int outerDim = Src.dim() - 1;
    for (auto& dst : Dest) {
      int ConcatAxisSize = dst.sizes()[dst.dim() - 1];
      torch::Tensor Slice =
          Src.narrow(outerDim, offset, ConcatAxisSize).to(dst.options());
      dst.index_put_({Predicate}, Slice.index({Predicate}));
      offset += ConcatAxisSize;
    }
    return offset;
  }


  ~AMSWorkflow()
  {
    AMS_DBG(Workflow, "Destroying Workflow Handler, DB: {}", DB.use_count());
    if (DB.use_count() == 2) {
      auto& dbm = ams::db::DBManager::getInstance();
      dbm.dropDB(domainName, rId);
    }
  }

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
  void evaluate(DomainLambda CallBack,
                ams::MutableArrayRef<torch::Tensor> Ins,
                ams::MutableArrayRef<torch::Tensor> InOuts,
                ams::MutableArrayRef<torch::Tensor> Outs)
  {
    CALIPER(CALI_MARK_BEGIN("AMSEvaluate");)
    AMS_DBG(Workflow,
            "Entering Workflow with TorchIn:{}, TorchInOut:{}, TorchOut:{}",
            Ins.size(),
            InOuts.size(),
            Outs.size());

    std::string msg{"ApplicationInput: [ "};
    for (auto& TI : Ins)
      msg += shapeToString(TI) + " ";
    msg += "]";
    AMS_DBG(Workflow, "{}", msg);

    msg = "ApplicationInOut: [ ";
    for (auto& TIO : InOuts)
      msg += shapeToString(TIO) + " ";
    msg += "]";
    AMS_DBG(Workflow, "{}", msg);

    msg = "ApplicationOutput: [ ";
    for (auto& TO : Outs)
      msg += shapeToString(TO) + " ";
    msg += "]";
    AMS_DBG(Workflow, "{}", msg);


    SmallVector<torch::Tensor> InputTensors(Ins.begin(), Ins.end());
    SmallVector<torch::Tensor> OutputTensors(Outs.begin(), Outs.end());
    AMS_DBG(Workflow,
            "Entering Workflow with TorchIn:{}, TorchOut:{}",
            InputTensors.size(),
            OutputTensors.size());
    for (auto Tensor : InOuts) {
      InputTensors.push_back(Tensor);
      OutputTensors.push_back(Tensor);
    }

    // Here we create a copy of the inputs/outputs. This is "necessary". To correctly handle
    // input-output cases and to also to set them to right precision.

    REPORT_MEM_USAGE(Workflow, "Start")

    if (!MLModel) {
      AMS_DBG(Workflow, "Model does not exist, calling entire application");
      // We need to clone only inout data to guarantee
      // we have a copy of them when writting the database
      SmallVector<torch::Tensor> PhysicInOutsBefore;
      for (auto S : InOuts)
        PhysicInOutsBefore.push_back(S.clone());

      // We call the application here
      CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
      callApplication(CallBack, Ins, InOuts, Outs);
      CALIPER(CALI_MARK_END("PHYSICS MODULE");)
      if (DB) storeComputedData(Ins, PhysicInOutsBefore, Outs, InOuts);
      CALIPER(CALI_MARK_END("AMSEvaluate");)
      return;
    }

    CALIPER(CALI_MARK_BEGIN("UPDATEMODEL");)
    if (updateModel()) {
      auto model = DB->getLatestModel();
      AMS_CINFO(Workflow, rId == 0, "Updating surrogate model with {}", model)
    }
    CALIPER(CALI_MARK_END("UPDATEMODEL");)

    // -------------------------------------------------------------
    // STEP 1: call the ML Model to get both the prediction and the predicates.
    // -------------------------------------------------------------
    CALIPER(CALI_MARK_BEGIN("SURROGATE");)
    // The predicate with which we will split the data on a lateMLInputsr step
    auto [MLOutputs, Predicate] = MLModel->evaluate(InputTensors, threshold);

    CALIPER(CALI_MARK_END("SURROGATE");)

    //Copy out the results of the ML Model to the correct indices, this needs to happen
    // NOTE: squeezing the predicate is important for the operations next. As they require
    // this shape. We cannot call the inpace operator as the Predicate is generated by the model,
    // in inference mode, and thus it has the read-only property set.
    Predicate = Predicate.squeeze();
    CALIPER(CALI_MARK_BEGIN("MLDomainToApplication");)
    int offset = MLDomainToApplication(MLOutputs, Outs, Predicate, 0);
    MLDomainToApplication(MLOutputs, InOuts, Predicate, offset);
    CALIPER(CALI_MARK_END("MLDomainToApplication");)

    // Revert pedicates and use it to pick the Physic points outputs.
    auto WrongMLIndices = torch::logical_not(Predicate);

    if (WrongMLIndices.sum().item<int64_t>() == 0) return;

    // Physis* tensors have the points which the model could not accurately predict
    CALIPER(CALI_MARK_BEGIN("PACK");)
    SmallVector<torch::Tensor> PhysicIns(subSelectTensors(Ins, WrongMLIndices));
    SmallVector<torch::Tensor> PhysicInOuts(
        subSelectTensors(InOuts, WrongMLIndices));
    // TODO: Outs does not need sub select, we will write all of these from scratch
    SmallVector<torch::Tensor> PhysicOuts(
        subSelectTensors(Outs, WrongMLIndices));
    CALIPER(CALI_MARK_END("PACK");)

    // Copy and clone. This important to take place before AppCall is executed. To keep a copy of the input values
    // that will be overwritten.
    SmallVector<torch::Tensor> PhysicInOutsBefore;
    for (auto S : PhysicInOuts)
      PhysicInOutsBefore.push_back(S.clone());


    // We call the application here
    CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
    callApplication(CallBack, PhysicIns, PhysicInOuts, PhysicOuts);
    CALIPER(CALI_MARK_END("PHYSICS MODULE");)


    CALIPER(CALI_MARK_BEGIN("UNPACK");)
    // Copy out the computation results to the original tensors/buffers
    ScatterPhysicOutputsToOrigDomain(PhysicOuts,
                                     WrongMLIndices.squeeze_(),
                                     Outs);
    ScatterPhysicOutputsToOrigDomain(PhysicInOuts, WrongMLIndices, InOuts);
    CALIPER(CALI_MARK_END("UNPACK");)


    AMS_DBG(Workflow, "Finished physics evaluation")

    if (DB) {
      storeComputedData(PhysicIns,
                        PhysicInOutsBefore,
                        PhysicOuts,
                        PhysicInOuts);
    }

    AMS_DBG(Workflow, "Finished AMSExecution")

    auto sizePhysics = (PhysicIns.size() > 0) ? PhysicIns[0].sizes()[0] : 0;
    auto sizeInput = (InputTensors.size() > 0) ? InputTensors[0].sizes()[0] : 0;
    float ratioComputedPhysics = 0.0;
    if (sizePhysics > 0 && sizeInput > 0)
      ratioComputedPhysics =
          (float)(PhysicIns[0].sizes()[0]) / float(InputTensors[0].sizes()[0]);

    AMS_CINFO(Workflow,
              rId == 0,
              "Computed {} elems"
              "using physics out of the {} items ({})",
              sizePhysics,
              sizeInput,
              ratioComputedPhysics);

    REPORT_MEM_USAGE(Workflow, "End")
    CALIPER(CALI_MARK_END("AMSEvaluate");)
  }

  // Graph-based evaluate methods (mirror tensor pattern)
  void evaluate(HomogeneousGraphDomainFn CallBack,
                const AMSHomogeneousGraph& graph_input,
                AMSHomogeneousGraphFields& outputs)
  {
    CALIPER(CALI_MARK_BEGIN("AMSEvaluateGraph");)

    // Try surrogate first
    bool surrogate_used = tryGraphSurrogate(this, graph_input, outputs);
    if (surrogate_used) {
      CALIPER(CALI_MARK_END("AMSEvaluateGraph");)
      return;
    }

    // Fallback to physics
    CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
    CallBack(graph_input, outputs);
    CALIPER(CALI_MARK_END("PHYSICS MODULE");)

    // Store data after physics computation
    storeGraphData(graph_input, outputs);

    CALIPER(CALI_MARK_END("AMSEvaluateGraph");)
  }

  void evaluate(HeterogeneousGraphDomainFn CallBack,
                const AMSHeterogeneousGraph& graph_input,
                AMSHeterogeneousGraphFields& outputs)
  {
    CALIPER(CALI_MARK_BEGIN("AMSEvaluateGraph");)

    // Try surrogate first
    bool surrogate_used = tryGraphSurrogate(this, graph_input, outputs);
    if (surrogate_used) {
      CALIPER(CALI_MARK_END("AMSEvaluateGraph");)
      return;
    }

    // Fallback to physics
    CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
    CallBack(graph_input, outputs);
    CALIPER(CALI_MARK_END("PHYSICS MODULE");)

    // Store data after physics computation
    storeGraphData(graph_input, outputs);

    CALIPER(CALI_MARK_END("AMSEvaluateGraph");)
  }

  std::string getDBName()
  {
    if (!DB) return "";
    return DB->getFilename();
  }
};


}  // namespace ams
#endif
