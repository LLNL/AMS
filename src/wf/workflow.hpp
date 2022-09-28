#ifndef __AMS_WORKFLOW_HPP__
#define __AMS_WORKFLOW_HPP__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    // (currently implemented as a file)
    BaseDB* DB;

public:
    // -------------------------------------------------------------------------
    // constructor and destructor
    // -------------------------------------------------------------------------
    AMSWorkflow() : AppCall(nullptr), hdcache(nullptr), surrogate(nullptr), DB (nullptr) {
#ifdef __ENABLE_DB__
        DB = new BaseDB("miniApp_data.txt");
        if (!DB) {
            std::cout << "Cannot create static database\n";
        }
#endif
    }

    AMSWorkflow(AMSPhysicFn _AppCall, char *uq_path,
        char *surrogate_path, char *db_path,
        bool is_cpu, FPTypeValue threshold) :
        AppCall(_AppCall) {
          surrogate = nullptr;
          if ( surrogate_path != nullptr )
            surrogate = new SurrogateModel<FPTypeValue>(surrogate_path, is_cpu);

          if ( uq_path != nullptr )
            hdcache = new HDCache<FPTypeValue>(uq_path, 10, false, threshold);
          else
            hdcache = new HDCache<FPTypeValue>(2, 10, false, threshold);

#ifdef __ENABLE_DB__
          DB = nullptr;
          if ( db_path != nullptr )
            DB = new BaseDB(db_path);
#endif
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

      if ( DB )
        delete DB;
    }

    // -------------------------------------------------------------------------
    // the main evaluate function of the ams workflow
    // -------------------------------------------------------------------------
    // todo: inputs should be const!
    void evaluate(void *probDescr, const int num_data,
                  const FPTypeValue **inputs, FPTypeValue **outputs,
                  int inputDim, int outputDim){

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
                AppCall( probDescr, elements,
                          reinterpret_cast<void**>(sparseInputs.data()),
                          reinterpret_cast<void**>(sparseOutputs.data()));
            }
            else {
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

            std::cout << std::setprecision(2)
                      << "[" << static_cast<int>(pId/partitionElements)  << "] Physics Computed elements / Surrogate computed elements "
                         "(Fraction of Physysics elements) ["
                      << packedElements << "/" << elements - packedElements << " ("
                      << static_cast<double>(packedElements) / static_cast<double>(elements)
                      << ")]\n";

            // ---- 3b: call the physics module and store in the dat base
            if (packedElements > 0) {
                CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
                AppCall(probDescr, elements,
                          reinterpret_cast<void **>(packedInputs.data()),
                          reinterpret_cast<void **>(packedOutputs.data()));
                CALIPER(CALI_MARK_END("PHYSICS MODULE");)

                if (DB != nullptr) {
                    CALIPER(CALI_MARK_BEGIN("DBSTORE");)
                    DB->Store(packedElements, packedInputs, packedOutputs);
                    CALIPER(CALI_MARK_END("DBSTORE");)
                }
            }

            // ---- 3c: unpack the data
            data_handler::unpack(predicate, elements, packedOutputs, sparseOutputs);

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
