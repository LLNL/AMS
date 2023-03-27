// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __AMS_WORKFLOW_HPP__
#define __AMS_WORKFLOW_HPP__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "AMS.h"
#include "ml/hdcache.hpp"
#include "ml/surrogate.hpp"
#include "wf/basedb.hpp"

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
  HDCache<FPTypeValue> *hdcache;

  /** @brief The torch surrogate model to replace the original physics function
   */
  SurrogateModel<FPTypeValue> *surrogate;

  /** @brief The database to store data for which we cannot apply the current
   * model */
  BaseDB<FPTypeValue> *DB;

  /** @brief The type of the database we will use (HDF5, CSV, etc) */
  AMSDBType dbType = AMSDBType::None;

  /** @brief The process id. For MPI runs this is the rank */
  const int rId;

  /** @brief The total number of processes participating in the simulation
   * (world_size for MPI) */
  int wSize;

  /** @brief  Whether data and simulation takes place on CPU or GPU*/
  bool isCPU;

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
    static const long bSize = 1 * 1024 * 1024;
    const int numIn = inputs.size();
    const int numOut = outputs.size();

    // No database, so just de-allocate and return
    if (DB == nullptr) {
      ams::ResourceManager::deallocate(inputs);
      ams::ResourceManager::deallocate(outputs);
      return;
    }

    std::vector<FPTypeValue *> hInputs, hOutputs;

    if (isCPU) return DB->store(num_elements, inputs, outputs);

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
    for (int i = 0; i < num_elements; i += bElements) {
      size_t actualElems = std::min(bElements, num_elements - i);
      // Copy input data to host
      for (int k = 0; k < numIn; k++) {
        ams::ResourceManager::copy(&inputs[k][i], hInputs[k], actualElems);
      }

      // Copy output data to host
      for (int k = 0; k < numIn; k++) {
        ams::ResourceManager::copy(&outputs[k][i], hOutputs[k], actualElems);
      }

      // Store to database
      DB->store(actualElems, hInputs, hOutputs);
    }
    ams::ResourceManager::deallocate(pPtr);

    return;
  }

public:
  AMSWorkflow()
      : AppCall(nullptr),
        hdcache(nullptr),
        surrogate(nullptr),
        DB(nullptr),
        dbType(AMSDBType::None),
        isCPU(false)
  {
#ifdef __ENABLE_DB__
    DB = createDB<FPTypeValue>("miniApp_data.txt", dbType, 0);
    if (!DB) {
      std::cout << "Cannot create static database\n";
    }
#endif
  }

  AMSWorkflow(AMSPhysicFn _AppCall,
              char *uq_path,
              char *surrogate_path,
              char *db_path,
              const AMSDBType dbType,
              bool is_cpu,
              FPTypeValue threshold,
              int _pId = 0,
              int _wSize = 1)
      : AppCall(_AppCall),
        dbType(dbType),
        rId(_pId),
        wSize(_wSize),
        isCPU(is_cpu)
  {
    surrogate = nullptr;
    if (surrogate_path != nullptr)
      surrogate = new SurrogateModel<FPTypeValue>(surrogate_path, is_cpu);

    // TODO: Fix magic number. 10 represents the number of neighbours I am
    // looking at.
    if (uq_path != nullptr)
      hdcache = new HDCache<FPTypeValue>(uq_path, 10, !is_cpu, threshold);
    else
      // This is a random hdcache returning true %threshold queries
      hdcache = new HDCache<FPTypeValue>(2, 10, !is_cpu, threshold);

    DB = nullptr;
    if (db_path != nullptr) {
      DB = createDB<FPTypeValue>(db_path, dbType, rId);
    }
  }

  void set_physics(AMSPhysicFn _AppCall) { AppCall = _AppCall; }

  void set_surrogate(SurrogateModel<FPTypeValue> *_surrogate)
  {
    surrogate = _surrogate;
  }

  void set_hdcache(HDCache<FPTypeValue> *_hdcache) { hdcache = _hdcache; }

  ~AMSWorkflow()
  {
    if (hdcache) delete hdcache;

    if (surrogate) delete surrogate;

    if (DB) delete DB;
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
  void evaluate(void *probDescr,
                const int totalElements,
                const FPTypeValue **inputs,
                FPTypeValue **outputs,
                int inputDim,
                int outputDim,
                MPI_Comm Comm = nullptr)
  {

    // To move around the inputs, outputs we bundle them as std::vectors
    std::vector<const FPTypeValue *> origInputs(inputs, inputs + inputDim);
    std::vector<FPTypeValue *> origOutputs(outputs, outputs + outputDim);

    // The predicate with which we will split the data on a later step
    bool *p_ml_acceptable = ams::ResourceManager::allocate<bool>(totalElements);

    // -------------------------------------------------------------
    // STEP 1: call the hdcache to look at input uncertainties
    //         to decide if making a ML inference makes sense
    // -------------------------------------------------------------
    if (hdcache != nullptr) {
      CALIPER(CALI_MARK_BEGIN("UQ_MODULE");)
      hdcache->evaluate(totalElements, origInputs, p_ml_acceptable);
      CALIPER(CALI_MARK_END("UQ_MODULE");)
    }

    // Pointer values which store input data values
    // to be computed using the eos function.
    std::vector<FPTypeValue *> packedInputs;

    for (int i = 0; i < inputDim; i++) {
      packedInputs.emplace_back(
          ams::ResourceManager::allocate<FPTypeValue>(totalElements));
    }

    // Pointer values which store output data values
    // to be computed using the eos function.
    std::vector<FPTypeValue *> packedOutputs;
    for (int i = 0; i < outputDim; i++) {
      packedOutputs.emplace_back(
          ams::ResourceManager::allocate<FPTypeValue>(totalElements));
    }

    bool *predicate = p_ml_acceptable;
    // null surrogate means we should call physics module
    if (surrogate == nullptr) {
      std::cout << "Calling application cause I dont have model\n";
      AppCall(probDescr,
              totalElements,
              reinterpret_cast<void **>(origInputs.data()),
              reinterpret_cast<void **>(origOutputs.data()));
    } else {
      std::cout << "Calling model\n";
      CALIPER(CALI_MARK_BEGIN("SURROGATE");)
      // We need to call the model on all data values.
      // Because we expect it to be faster.
      // I guess we may need to add some policy to do this
      surrogate->evaluate(totalElements, origInputs, origOutputs);
      CALIPER(CALI_MARK_END("SURROGATE");)
    }


    // -----------------------------------------------------------------
    // STEP 3: call physics module only where d_dense_need_phys = true
    // -----------------------------------------------------------------
    // ---- 3a: we need to pack the sparse data based on the uq flag
    const long packedElements =
        data_handler::pack(predicate, totalElements, origInputs, packedInputs);

#ifdef __ENABLE_MPI__
    if (Comm) {
      MPI_Barrier(Comm);
    }
#endif

    std::cout << std::setprecision(2) << "Physics Computed elements / Surrogate computed elements "
                 "(Fraction of Physics elements) ["
              << packedElements << "/" << totalElements - packedElements << " ("
              << static_cast<double>(packedElements) / static_cast<double>(totalElements)
              << ")]\n";

    // ---- 3b: call the physics module and store in the data base
    if (packedElements > 0 ) {
      CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
      AppCall(probDescr,
              packedElements,
              reinterpret_cast<void **>(packedInputs.data()),
              reinterpret_cast<void **>(packedOutputs.data()));
      CALIPER(CALI_MARK_END("PHYSICS MODULE");)
    }
#ifdef __ENABLE_MPI__
    // TODO: Here we need to load balance. Each rank may have a different
    // number of PackedElemets. Thus we need to distribute the packedOutputs
    // to all ranks
    if (Comm) {
      MPI_Barrier(Comm);
    }
#endif
    // ---- 3c: unpack the data
    data_handler::unpack(predicate, totalElements, packedOutputs, origOutputs);

    if (DB != nullptr) {
      CALIPER(CALI_MARK_BEGIN("DBSTORE");)
      Store(packedElements, packedInputs, packedOutputs);
      CALIPER(CALI_MARK_END("DBSTORE");)
      std::cout << "Stored " << packedElements
                << " physics-computed elements in " << DB->type() << std::endl;
    }

    // -----------------------------------------------------------------
    // Deallocate temporal data
    // -----------------------------------------------------------------
    for (int i = 0; i < inputDim; i++)
      ams::ResourceManager::deallocate(packedInputs[i]);
    for (int i = 0; i < outputDim; i++)
      ams::ResourceManager::deallocate(packedOutputs[i]);

    ams::ResourceManager::deallocate(p_ml_acceptable);
  }
};


}  // namespace ams
#endif
