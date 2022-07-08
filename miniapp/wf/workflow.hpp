#ifndef __AMS_WORKFLOW_HPP__
#define __AMS_WORKFLOW_HPP__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "app/eos.hpp"
#include "ml/hdcache.hpp"
#include "ml/surrogate.hpp"
#include "wf/basedb.hpp"
#include "utils/utils_data.hpp"
#include "utils/utils_caliper.hpp"
#include "utils/allocator.hpp"


//! ----------------------------------------------------------------------------
//! AMS Workflow class
//! the purpose of this class is to expose an "evaluate" function
//!     which has the same interface as the physics evaluation
//! the intention is that we can easily switch in and out AMS wf in app code
//! ----------------------------------------------------------------------------
namespace ams {
class AMSWorkflow {

    using TypeValue = double;       // todo: should template the class
    using data_handler = ams::DataHandler<TypeValue>;

    const int num_mats;

    // eos computations
    std::vector<EOS*> eoses;

    // added to include ML
    std::vector<HDCache<TypeValue>*> hdcaches;
    std::vector<SurrogateModel<TypeValue>*> surrogates;

    // Added to include an offline DB
    // (currently implemented as a file)
    BaseDB* DB = nullptr;

public:
    // -------------------------------------------------------------------------
    // constructor and destructor
    // -------------------------------------------------------------------------
    AMSWorkflow(int _num_mats) : num_mats(_num_mats) {

        // setup eos
        eoses.resize(num_mats, nullptr);
        hdcaches.resize(num_mats, nullptr);
        surrogates.resize(num_mats, nullptr);

        // default database is null
        DB = nullptr;
#ifdef __ENABLE_DB__
        DB = new BaseDB("miniApp_data.txt");
        if (!DB) {
            std::cout << "Cannot create static database\n";
        }
#endif
    }

    void set_eos(int mat_idx, EOS* eos) {
        eoses[mat_idx] = eos;
    }
    void set_surrogate(int mat_idx, SurrogateModel<TypeValue>* surrogate) {
        surrogates[mat_idx] = surrogate;
    }
    void set_hdcache(int mat_idx, HDCache<TypeValue>* hdcache) {
        hdcaches[mat_idx] = hdcache;
    }

    ~AMSWorkflow() {
        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
            delete eoses[mat_idx];
            delete hdcaches[mat_idx];
            delete surrogates[mat_idx];
        }
        delete DB;
    }

    // -------------------------------------------------------------------------
    // the main evaluate function of the ams workflow
    // -------------------------------------------------------------------------
    // todo: inputs should be const!
    void evaluate(const int num_data,
                  TypeValue* pDensity, TypeValue* pEnergy,          // inputs
                  TypeValue* pPressure, TypeValue* pSoundSpeed2,    // outputs
                  TypeValue* pBulkmod, TypeValue* pTemperature,     // outputs
                  const int mat_idx = 0) {

        // we want to make this function equivalent to the eos call
        // the only difference is "mat_idx", which is now an optional parameters
        // we need mat_idx only to figure out which instances to use
        auto eos = eoses[mat_idx];
        auto hdcache = hdcaches[mat_idx];
        auto surrogate = surrogates[mat_idx];


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
            hdcache->print();
            hdcache->evaluate(num_data, {pDensity, pEnergy}, p_ml_acceptable);
            CALIPER(CALI_MARK_END("UQ_MODULE");)
        }

        // -------------------------------------------------------------
        // STEP 2: let's call surrogate for everything
        // ideally, we should do step 1 and step 2 async!
        // -------------------------------------------------------------

        /*
         At this point I am puzzled with how allocations should be done
         in regards to packing. The worst case scenario and simlest policy
         would require "length" *("Num Input Vectors" + "Num Output Vectors" + 1).
         This can be fine in the case of CPU execution. It is definetely too high
         for GPU execution. I will start a partioning scheme that limits the memory
         usage to a user defined size "PARTITION_SIZE". Setting the size to length
         should operate as the worst case scenario.
        */

        int partitionElements = data_handler::computePartitionSize(2, 4);

        /*
         The way partioning is working now we can have "inbalance" across
         iterations. As we only check the "uq" vector for the next
         partionElements. Thus, the vectors will be filled in up to that size.
         However, most times the vector will be half-empty.
        */
        for (int pId = 0; pId < num_data; pId += partitionElements) {

            // Pointer values which store data values
            // to be computed using the eos function.
            const int elements = std::min(partitionElements, num_data - pId);

            TypeValue *packed_density = ams::ResourceManager::allocate<TypeValue>(elements);
            TypeValue *packed_energy = ams::ResourceManager::allocate<TypeValue>(elements);
            TypeValue *packed_pressure = ams::ResourceManager::allocate<TypeValue>(elements);
            TypeValue *packed_soundspeed2 = ams::ResourceManager::allocate<TypeValue>(elements);
            TypeValue *packed_bulkmod = ams::ResourceManager::allocate<TypeValue>(elements);
            TypeValue *packed_temperature = ams::ResourceManager::allocate<TypeValue>(elements);

            std::vector<TypeValue*> sparse_inputs({&pDensity[pId], &pEnergy[pId]});
            std::vector<TypeValue*> sparse_outputs({&pPressure[pId], &pSoundSpeed2[pId],
                                                    &pBulkmod[pId], &pTemperature[pId]});

            std::vector<TypeValue*> packed_inputs({packed_density, packed_energy});
            std::vector<TypeValue*> packed_outputs({packed_pressure, packed_soundspeed2,
                                                    packed_bulkmod, packed_temperature});

            bool* predicate = &p_ml_acceptable[pId];

            // null surrogate means we should call physics module
            if (surrogate == nullptr) {

                // Because we dont have py-torch we call the actual physics code.
                // This helps debugging. Disabling torch should give identicalresults.
                eos->Eval(elements,
                          &pEnergy[pId], &pDensity[pId],          // inputs
                          &pPressure[pId], &pSoundSpeed2[pId],    // outputs
                          &pBulkmod[pId], &pTemperature[pId]);    // outputs
            }

            else {
                /*
                 One of the benefits of the packing is that we indirectly limit the size
                 of the model. As it will perform inference on up to "elements" points.
                 Thus, we indirectly control the maximum memory of the model.
                */
                CALIPER(CALI_MARK_BEGIN("SURROGATE");)
                surrogate->Eval(elements, sparse_inputs, sparse_outputs);
                CALIPER(CALI_MARK_END("SURROGATE");)

#ifdef __SURROGATE_DEBUG__
                // TODO: I will revisit the RMSE later. We need to compute it only
                // for point which we have low uncertainty.
                eos->computeRMSE(num_elems_for_mat * num_qpts,
                                 &d_dense_density(0, 0), &d_dense_energy(0, 0),
                                 &d_dense_pressure(0, 0), &d_dense_soundspeed2(0, 0),
                                 &d_dense_bulkmod(0, 0), &d_dense_temperature(0, 0));
#endif
            }


            // -----------------------------------------------------------------
            // STEP 3: call physics module only where d_dense_need_phys = true
            // -----------------------------------------------------------------

            // ---- 3a: we need to pack the sparse data based on the uq flag
            const long packedElements =
                data_handler::pack(predicate, elements, sparse_inputs, packed_inputs);

            std::cout << std::setprecision(2)
                      << "Physis Computed elements / Surrogate computed elements "
                         "(Fraction) ["
                      << packedElements << "/" << elements - packedElements << " ("
                      << static_cast<double>(packedElements) / static_cast<double>(elements)
                      << ")]\n";

            // ---- 3b: call the physics module and store in the dat base
            if (packedElements > 0) {

                CALIPER(CALI_MARK_BEGIN("PHYSICS MODULE");)
                eos->Eval(packedElements,
                          packed_energy, packed_density,             // inputs
                          packed_pressure, packed_soundspeed2,       // outputs
                          packed_bulkmod, packed_temperature);       // outputs
                CALIPER(CALI_MARK_END("PHYSICS MODULE");)

                if (DB != nullptr) {
                    CALIPER(CALI_MARK_BEGIN("DBSTORE");)
                    DB->Store(packedElements, packed_inputs, packed_outputs);
                    CALIPER(CALI_MARK_END("DBSTORE");)
                }
            }

            // ---- 3c: unpack the data
            data_handler::unpack(predicate, elements, packed_outputs, sparse_outputs);


            // -----------------------------------------------------------------
            // Deallocate temporal data
            // -----------------------------------------------------------------
            ams::ResourceManager::deallocate(packed_density);
            ams::ResourceManager::deallocate(packed_energy);
            ams::ResourceManager::deallocate(packed_pressure);
            ams::ResourceManager::deallocate(packed_soundspeed2);
            ams::ResourceManager::deallocate(packed_bulkmod);
            ams::ResourceManager::deallocate(packed_temperature);
        }
        ams::ResourceManager::deallocate(p_ml_acceptable);
    }
};
}   // end of namespace
#endif
