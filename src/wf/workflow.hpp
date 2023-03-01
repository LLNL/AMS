// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __AMS_WORKFLOW_HPP__
#define __AMS_WORKFLOW_HPP__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

#include "AMS.h"
#include "ml/surrogate.hpp"
#include "ml/hdcache.hpp"

#include "wf/basedb.hpp"

//! ----------------------------------------------------------------------------
//! AMS Workflow class
//! the purpose of this class is to expose an "evaluate" function
//!     which has the same interface as the physics evaluation
//! the intention is that we can easily switch in and out AMS wf in app code
//! ----------------------------------------------------------------------------
namespace ams {
template<typename FPTypeValue>
class AMSWorkflow {

    static_assert (std::is_floating_point<FPTypeValue>::value,
                  "HDCache supports floating-point values (floats, doubles, and long doubles) only!");

    using data_handler = ams::DataHandler<FPTypeValue>;

    AMSPhysicFn AppCall;

    // added to include ML
    HDCache<FPTypeValue>* hdcache;
    SurrogateModel<FPTypeValue>* surrogate;

    // Added to include an offline DB
    // (currently implemented as a file or a Redis backend)
    BaseDB<FPTypeValue>* DB;
    AMSDBType dbType = AMSDBType::None;

    // process Id and total number of processes
    // in this run
    int rId;
    int wSize;

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
               std::vector<FPTypeValue*>& inputs,
               std::vector<FPTypeValue*>& outputs) {
      // 1 MB of buffer size;
      // TODO: Fix magic number
      static const long bSize = 1 * 1024 * 1024;
      const int numIn = inputs.size();
      const int numOut = outputs.size();

      // No database, so just de-allocate and return
      if ( DB == nullptr ){
        ams::ResourceManager::deallocate(inputs);
        ams::ResourceManager::deallocate(outputs);
        return;
      }

      std::vector<FPTypeValue *> hInputs, hOutputs;
      if ( !isCPU ){
        // Compute number of elements that fit inside the buffer
        size_t bElements = bSize / sizeof(FPTypeValue);
        FPTypeValue *pPtr = ams::ResourceManager::allocate<FPTypeValue>(bElements, AMSResourceType::PINNED);
        // Total inner vector dimensions (inputs and outputs)
        size_t totalDims = inputs.size() + outputs.size();
        // Compute number of elements of each outer dimension that fit in buffer
        size_t elPerDim = static_cast<int>(floor(bElements / totalDims));

        for ( int i = 0 ; i < inputs.size(); i++)
          hInputs.push_back(&pPtr[i*elPerDim]);

        for ( int i = 0 ; i < outputs.size(); i++)
          hOutputs.push_back(&pPtr[( i+inputs.size()) * elPerDim]);

        // Iterate over all chunks
        for ( int i = 0; i < num_elements; i+= bElements ){
          size_t actualElems = std::min(bElements, num_elements - i );
          // Copy input data
          for ( int k = 0; k < numIn; k++ ){
            ams::ResourceManager::copy(&inputs[k][i], hInputs[k], actualElems);
          }

          // Copy output data
          for ( int k = 0; k < numIn; k++ ){
            ams::ResourceManager::copy(&outputs[k][i], hOutputs[k], actualElems);
          }

          //Store to database
          DB->store(actualElems, hInputs, hOutputs);
        }
        ams::ResourceManager::deallocate(pPtr);
      }
      else {
        DB->store(num_elements, inputs, outputs);
      }

      return;
    }

public:
    // -------------------------------------------------------------------------
    // constructor and destructor
    // -------------------------------------------------------------------------
    AMSWorkflow() : AppCall(nullptr), hdcache(nullptr), surrogate(nullptr), DB(nullptr), dbType(AMSDBType::None), isCPU(false) {
#ifdef __ENABLE_DB__
        DB = createDB<FPTypeValue>("miniApp_data.txt", dbType, 0);
        if (!DB) {
            std::cout << "Cannot create static database\n";
        }
#endif
    }

    AMSWorkflow(AMSPhysicFn _AppCall, char *uq_path,
        char *surrogate_path, char *db_path, const AMSDBType dbType,
        bool is_cpu, FPTypeValue threshold,
        int _pId = 0, int _wSize = 1) :
        AppCall(_AppCall), dbType(dbType), rId(_pId), wSize(_wSize) , isCPU(is_cpu){
          surrogate = nullptr;
          if ( surrogate_path != nullptr )
            surrogate = new SurrogateModel<FPTypeValue>(surrogate_path, is_cpu);

          if ( uq_path != nullptr )
            hdcache = new HDCache<FPTypeValue>(uq_path, 10, !is_cpu, threshold);
          else
            hdcache = new HDCache<FPTypeValue>(2, 10, !is_cpu, threshold);

          DB = nullptr;
          if ( db_path != nullptr ) {
            DB = createDB<FPTypeValue>(db_path, dbType, rId);
          }
    }

    void set_physics(AMSPhysicFn _AppCall){
      AppCall = _AppCall;
    }

    void set_surrogate(SurrogateModel<FPTypeValue>* _surrogate) {
        surrogate = _surrogate;
    }

    void set_hdcache(HDCache<FPTypeValue>* _hdcache) {
        hdcache = _hdcache;
    }

    ~AMSWorkflow() {
      if ( hdcache )
        delete hdcache;

      if ( surrogate )
        delete surrogate;

      if ( DB ){
        //std::cerr << "Deleting DB\n";
        delete DB;
      }
    }

    // -------------------------------------------------------------------------
    // the main evaluate function of the ams workflow
    // -------------------------------------------------------------------------
    // todo: inputs should be const!
    void evaluate(void *probDescr, const int num_data,
                  const FPTypeValue **inputs, FPTypeValue **outputs,
                  int inputDim, int outputDim, MPI_Comm Comm=nullptr){

        std::vector<const FPTypeValue *> origInputs(inputs, inputs+inputDim);
        std::vector<FPTypeValue *> origOutputs(outputs, outputs + outputDim);

        /* The allocate function always allocates on the default device. The default device
         * can be set by calling setDefaultDataAllocator. Otherwise we can explicitly control
         * the location of the data by calling allocate(size, AMSDevice).
         */
        bool* p_ml_acceptable = ams::ResourceManager::allocate<bool>(num_data);

        // -------------------------------------------------------------
        // STEP 1: call the hdcache to look at input uncertainties
        //         to decide if making a ML inference makes sense
        // -------------------------------------------------------------
        // ideally, we should do step 1 and step 2 async!

        if (hdcache != nullptr) {
            CALIPER(CALI_MARK_BEGIN("UQ_MODULE");)
            hdcache->evaluate(num_data, origInputs, p_ml_acceptable);
            CALIPER(CALI_MARK_END("UQ_MODULE");)
        }

        int partitionElements = data_handler::computePartitionSize(2, 4);
        // FIXME: We need to either remove the idea of partioning the outer data and parallelizing
        // or take into account the implications of such an approach.
        // The current implementation has bugs in the case of MPI execution. The for loop can execute
        // A different number of times. Thus blocking any MPI all to all functions. The line below
        // disables that feature. It still not bullet proof. There is some chance for some specific
        // node to have a no element at all and it can result into a deadlock
        partitionElements = num_data;

        for (int pId = 0; pId < num_data; pId += partitionElements) {

            // Pointer values which store data values
            // to be computed using the eos function.
            const int elements = std::min(partitionElements, num_data - pId);
            std::vector<FPTypeValue *> packedInputs;
            // TODO: Do not drop the const modifier here.
            // I am using const cast later
            std::vector<FPTypeValue *> sparseInputs;

            for (int i = 0; i < inputDim; i++){
              packedInputs.emplace_back(ams::ResourceManager::allocate<FPTypeValue>(elements));
              sparseInputs.emplace_back(const_cast<FPTypeValue *>(&origInputs[i][pId]));
            }

            std::vector<FPTypeValue *> packedOutputs;
            std::vector<FPTypeValue *> sparseOutputs;
            for (int i = 0; i < outputDim; i++){
              packedOutputs.emplace_back(ams::ResourceManager::allocate<FPTypeValue>(elements));
              sparseOutputs.emplace_back(&origOutputs[i][pId]);
            }

            bool* predicate = &p_ml_acceptable[pId];
            // null surrogate means we should call physics module
            if (surrogate == nullptr) {
                std::cout << "Calling application cause I dont have model\n";
                AppCall( probDescr, elements,
                          reinterpret_cast<void**>(sparseInputs.data()),
                          reinterpret_cast<void**>(sparseOutputs.data()));
            }
            else {
                std::cout << "Calling model\n";
                CALIPER(CALI_MARK_BEGIN("SURROGATE");)
                //We need to call the model on all data values.
                //Because we expect it to be faster.
                //I guess we may need to add some policy to do this
                surrogate->evaluate(elements, sparseInputs, sparseOutputs);
                CALIPER(CALI_MARK_END("SURROGATE");)
            }


            // -----------------------------------------------------------------
            // STEP 3: call physics module only where d_dense_need_phys = true
            // -----------------------------------------------------------------
            // ---- 3a: we need to pack the sparse data based on the uq flag
            const long packedElements =
                data_handler::pack(predicate, elements, sparseInputs, packedInputs);

            // TODO: Here we need to load balance. Each rank may have a different
            // number of PackedElemets. Thus we need to distribute the packedInputs
            // to all ranks
#ifdef __ENABLE_MPI__
            if ( Comm ){
              MPI_Barrier(Comm);
            }
#endif

            std::cout << std::setprecision(2)
                      << "[" << static_cast<int>(pId/partitionElements)  << "] Physics Computed elements / Surrogate computed elements "
                         "(Fraction of Physics elements) ["
                      << packedElements << "/" << elements - packedElements << " ("
                      << static_cast<double>(packedElements) / static_cast<double>(elements)
                      << ")]\n";

            // ---- 3b: call the physics module and store in the data base
            if (packedElements > 0) {
                CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
                AppCall(probDescr, elements,
                          reinterpret_cast<void **>(packedInputs.data()),
                          reinterpret_cast<void **>(packedOutputs.data()));
                CALIPER(CALI_MARK_END("PHYSICS MODULE");)
            }
#ifdef __ENABLE_MPI__
            // TODO: Here we need to load balance. Each rank may have a different
            // number of PackedElemets. Thus we need to distribute the packedOutputs
            // to all ranks
            if ( Comm ){
              MPI_Barrier(Comm);
            }
#endif
            // ---- 3c: unpack the data
            data_handler::unpack(predicate, elements, packedOutputs, sparseOutputs);

            if (DB != nullptr) {
                CALIPER(CALI_MARK_BEGIN("DBSTORE");)
                Store(packedElements, packedInputs, packedOutputs);
                CALIPER(CALI_MARK_END("DBSTORE");)
                std::cout << "Stored " << packedElements << " physics-computed elements in " << DB->type() << std::endl;
            }

            // -----------------------------------------------------------------
            // Deallocate temporal data
            // -----------------------------------------------------------------
            for (int i = 0; i < inputDim; i++)
              ams::ResourceManager::deallocate(packedInputs[i]);
            for (int i  = 0; i < outputDim; i++)
              ams::ResourceManager::deallocate(packedOutputs[i]);
        }
        ams::ResourceManager::deallocate(p_ml_acceptable);
    }
};


}   // end of namespace
#endif
