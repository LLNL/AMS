#ifndef __MINIAPP_HPP__
#define __MINIAPP_HPP__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "app/eos.hpp"
#include "app/eos_constant_on_host.hpp"
#include "app/eos_idealgas.hpp"
#include "app/utils_mfem.hpp"

// this macro completely bypasses all AMS functionality
// this allows us to check how easy is it to test ams
#define USE_AMS

#ifdef USE_AMS
#include "AMS.h"
#endif

//! ----------------------------------------------------------------------------
//! mini app class
//! ----------------------------------------------------------------------------
namespace ams {

template <typename TypeValue = double>
class MiniApp {

private:
    const int num_mats;
    const int num_elems;
    const int num_qpts;
    const std::string device_name;
    const bool use_device;
    const bool pack_sparse_mats;

    std::vector<EOS*> eoses;

#ifdef USE_AMS
    AMSExecutor* workflow;
#endif

    CALIPER(cali::ConfigManager mgr;)

public:
    // -------------------------------------------------------------------------
    // constructor and destructor
    // -------------------------------------------------------------------------
    MiniApp(int _num_mats, int _num_elems, int _num_qpts,
            const std::string _device_name, bool _pack_sparse_mats) :
            num_mats(_num_mats), num_elems(_num_elems), num_qpts(_num_qpts),
            device_name(_device_name), use_device(device_name != "cpu"),
            pack_sparse_mats(_pack_sparse_mats && !use_device) {

        eoses.resize(num_mats, nullptr);

#ifdef USE_AMS
        // We need a AMSExecutor for each material.
        // This implicitly implies that we are going to
        // have a differnt ml model for each type of material.
        // If we have a single model for all materials then we
        // need to make a single model here.
        workflow = new AMSExecutor[num_mats]();
#endif
        CALIPER(mgr.start();)
    }

    ~MiniApp() {
#ifdef USE_AMS
        delete [] workflow;
#endif
        CALIPER(mgr.flush());
    }

    // -------------------------------------------------------------------------
    // setup the miniapp
    // -------------------------------------------------------------------------
    void setup(const std::string eos_name, const std::string model_path,
               const std::string hdcache_path, TypeValue threshold = 0.5) {

        const std::string &alloc_name_host(AMSGetAllocatorName(AMSResourceType::HOST));

        // ---------------------------------------------------------------------
        // setup EOSes
        // ---------------------------------------------------------------------
        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
            if (eos_name == "ideal_gas") {
                eoses[mat_idx] = new IdealGas(1.6, 1.4);
            } else if (eos_name == "constant_host") {
                eoses[mat_idx] = new ConstantEOSOnHost(alloc_name_host.c_str(), 1.0);
            } else {
                std::cerr << "unknown eos `" << eos_name << "'" << std::endl;
                return;
            }
        }

        // ---------------------------------------------------------------------
        // setup AMS workflow (surrogate and cache)
        // ---------------------------------------------------------------------
#ifdef USE_AMS
        char *uq_path = nullptr;
        char *surrogate_path = nullptr;
        char *db_path = nullptr;

#ifdef __ENABLE_FAISS__
        if ( hdcache_path.size() > 0 )
          uq_path = const_cast<char*>(hdcache_path.c_str());
#endif

#ifdef __ENABLE_TORCH__
        if ( model_path.size() > 0 )
          surrogate_path = const_cast<char*>(model_path.c_str());
#endif

#ifdef __ENABLE_DB__
        db_path = "miniapp.txt";
#endif
        AMSResourceType device = AMSResourceType::HOST;
        if ( use_device )
          device = AMSResourceType::DEVICE;


        AMSConfig amsConf = {
        AMSExecPolicy::SinglePass,
        AMSDType::Double,
        device,
        callBack,
        surrogate_path,
        uq_path,
        db_path,
        threshold};

        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
          workflow[mat_idx] = AMSCreateExecutor(amsConf);
        }
#endif
    }

    // -------------------------------------------------------------------------
    // miniapp evaluation (all cycles)
    // -------------------------------------------------------------------------
    void evaluate(int stop_cycle,
                  TypeValue* density_in, TypeValue* energy_in,
                  const bool* indicators_in) {

        // the inputs here are raw pointers to input data
        // these are on the host
        // dimensions of input data is assumed to be (num_mats, num_elems, num_qpts)
        if (0) {
            print_tensor_array("density_in", density_in, {num_mats, num_elems, num_qpts});
            print_tensor_array("energy_in", energy_in, {num_mats, num_elems, num_qpts});
            print_tensor_array("indicators_in", indicators_in, {1, num_mats, num_elems});
        }

        // -------------------------------------------------------------------------
        // initialize inputs and outputs as mfem tensors
        // -------------------------------------------------------------------------
        // our mfem::DenseTensor will have the shape (qpts x elems x mat)
        // this stores mat 2D "DenseMatrix" each of size qpts x elems

        // let's just use mfem to wrap the tensor around the input pointer
        mfem::DenseTensor density;
        mfem::DenseTensor energy;
        density.UseExternalData(density_in, num_qpts, num_elems, num_mats);
        energy.UseExternalData(energy_in, num_qpts, num_elems, num_mats);

        if (0) {
            print_dense_tensor("density", density);
            print_dense_tensor("density", density, indicators_in);
            // print_dense_tensor("energy", energy);
            // print_dense_tensor("energy", energy, indicators_in);
        }

        // outputs
        mfem::DenseTensor pressure(num_qpts, num_elems, num_mats);
        mfem::DenseTensor soundspeed2(num_qpts, num_elems, num_mats);
        mfem::DenseTensor bulkmod(num_qpts, num_elems, num_mats);
        mfem::DenseTensor temperature(num_qpts, num_elems, num_mats);

        // ---------------------------------------------------------------------
        // need a way to store indices of active elements for each material
        // will use a linearized array for this
        // first num_mat elements will store the total count of all elements thus far
        //                                                    (actually, + num_mats)
        // after that, store the indices of those elements in order
        mfem::Array<int> sparse_elem_indices;
        sparse_elem_indices.SetSize(num_mats);
        sparse_elem_indices.Reserve(num_mats * num_elems);

        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
            for (int elem_idx = 0; elem_idx < num_elems; ++elem_idx) {
                if (indicators_in[elem_idx + num_elems * mat_idx]) {
                    sparse_elem_indices.Append(elem_idx);
                }
            }
            sparse_elem_indices[mat_idx] = sparse_elem_indices.Size();
        }

        // ---------------------------------------------------------------------
        // run through the cycles (time-steps)
        for (int c = 0; c <= stop_cycle; ++c) {
            std::cout << "\n--> cycle " << c << std::endl;
            evaluate(density, energy,                   // inputs
                     sparse_elem_indices,               // aux data
                     pressure, soundspeed2,             // outputs
                     bulkmod, temperature);             // outputs
        }

        // pressure.HostRead();
        // print_dense_tensor("pressure", pressure);
    }

private:
    // -------------------------------------------------------------------------
    // miniapp evaluation (one cycle)
    // -------------------------------------------------------------------------
    void evaluate(mfem::DenseTensor& density, mfem::DenseTensor& energy,
                  mfem::Array<int>& sparse_elem_indices, mfem::DenseTensor& pressure,
                  mfem::DenseTensor& soundspeed2, mfem::DenseTensor& bulkmod,
                  mfem::DenseTensor& temperature) {

        CALIPER(CALI_MARK_FUNCTION_BEGIN;)

        // move/allocate data on the device.
        // if the data is already on the device this is basically a noop
        const auto d_density = mfemReshapeTensor3(density, Read);
        const auto d_energy = mfemReshapeTensor3(energy, Read);
        const auto d_pressure = mfemReshapeTensor3(pressure, Write);
        const auto d_soundspeed2 = mfemReshapeTensor3(soundspeed2, Write);
        const auto d_bulkmod = mfemReshapeTensor3(bulkmod, Write);
        const auto d_temperature = mfemReshapeTensor3(temperature, Write);

        const auto d_sparse_elem_indices = mfemReshapeArray1(sparse_elem_indices, Write);


        // ---------------------------------------------------------------------
        // for each material
        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
            const int offset_curr = mat_idx == 0 ? num_mats : sparse_elem_indices[mat_idx - 1];
            const int offset_next = sparse_elem_indices[mat_idx];

            const int num_elems_for_mat = offset_next - offset_curr;
            if (num_elems_for_mat == 0) {
                continue;
            }

            // -----------------------------------------------------------------
            // NOTE: we've found it's faster to do sparse lookups on GPUs but on CPUs
            // the dense packing->looked->unpacking is better if we're using expensive
            // eoses. in the future we may just use dense representations everywhere
            // but for now we use sparse ones.
            if (pack_sparse_mats && num_elems_for_mat < num_elems) {
                std::cout << " material " << mat_idx << ": using sparse packing for "
                          << num_elems_for_mat << " elems\n";

                // -------------------------------------------------------------
                // TODO: I think Tom mentiond we can allocate these outside the loop
                // check again
                mfem::Array<TypeValue> dense_density(num_elems_for_mat * num_qpts);
                mfem::Array<TypeValue> dense_energy(num_elems_for_mat * num_qpts);
                mfem::Array<TypeValue> dense_pressure(num_elems_for_mat * num_qpts);
                mfem::Array<TypeValue> dense_soundspeed2(num_elems_for_mat * num_qpts);
                mfem::Array<TypeValue> dense_bulkmod(num_elems_for_mat * num_qpts);
                mfem::Array<TypeValue> dense_temperature(num_elems_for_mat * num_qpts);

                // these are device tensors!
                auto d_dense_density = mfemReshapeArray2(dense_density, Write, num_qpts, num_elems_for_mat);
                auto d_dense_energy = mfemReshapeArray2(dense_energy, Write, num_qpts, num_elems_for_mat);
                auto d_dense_pressure = mfemReshapeArray2(dense_pressure, Write, num_qpts, num_elems_for_mat);
                auto d_dense_soundspeed2 = mfemReshapeArray2(dense_soundspeed2, Write, num_qpts, num_elems_for_mat);
                auto d_dense_bulkmod = mfemReshapeArray2(dense_bulkmod, Write, num_qpts, num_elems_for_mat);
                auto d_dense_temperature = mfemReshapeArray2(dense_temperature, Write, num_qpts, num_elems_for_mat);

                // -------------------------------------------------------------
                // sparse -> dense
                CALIPER(CALI_MARK_BEGIN("SPARSE_TO_DENSE");)
                pack_ij(mat_idx, num_qpts, num_elems_for_mat, offset_curr,
                          d_sparse_elem_indices, d_density, d_dense_density, d_energy,
                          d_dense_energy);
                CALIPER(CALI_MARK_END("SPARSE_TO_DENSE");)
                // -------------------------------------------------------------
                std::vector<const double *> inputs = { &d_dense_density(0, 0), &d_dense_energy(0, 0) };
                std::vector<double *> outputs = {&d_dense_pressure(0, 0), &d_dense_soundspeed2(0, 0), &d_dense_bulkmod(0, 0), &d_dense_temperature(0, 0)};

#ifdef USE_AMS
                AMSExecute(workflow[mat_idx], static_cast<void *>(eoses[mat_idx]),
                                  num_elems_for_mat * num_qpts,
                                  reinterpret_cast<const void **>(inputs.data()),
                                  reinterpret_cast<void **>(outputs.data()),
                                  inputs.size(), outputs.size());
#else
                eoses[mat_idx]->Eval(num_elems_for_mat * num_qpts,
                                     &d_dense_density(0, 0), &d_dense_energy(0, 0),
                                     &d_dense_pressure(0, 0), &d_dense_soundspeed2(0, 0),
                                     &d_dense_bulkmod(0, 0), &d_dense_temperature(0, 0));
#endif
                // -------------------------------------------------------------
                // dense -> sparse
                CALIPER(CALI_MARK_BEGIN("DENSE_TO_SPARSE");)
                unpack_ij(mat_idx, num_qpts, num_elems_for_mat, offset_curr,
                          d_sparse_elem_indices, d_dense_pressure, d_pressure,
                          d_dense_soundspeed2, d_soundspeed2, d_dense_bulkmod,
                          d_bulkmod, d_dense_temperature, d_temperature);
                CALIPER(CALI_MARK_END("DENSE_TO_SPARSE");)
                // -------------------------------------------------------------

            } else {
#ifdef USE_AMS
                std::cout << " material " << mat_idx << ": using dense packing for "
                          << num_elems << " elems\n";

                std::vector<const double *> inputs = { &d_density(0, 0, mat_idx), &d_energy(0, 0, mat_idx) };
                std::vector<double *> outputs = { &d_pressure(0, 0, mat_idx), &d_soundspeed2(0, 0, mat_idx),
                                                  &d_bulkmod(0, 0, mat_idx), &d_temperature(0, 0, mat_idx)};

                AMSExecute(workflow[mat_idx], static_cast<void*>(eoses[mat_idx]),
                    num_elems_for_mat * num_qpts,
                    reinterpret_cast<const void **>(inputs.data()),
                    reinterpret_cast<void **>(outputs.data()),
                    inputs.size(), outputs.size());

#else
                eoses[mat_idx]->Eval(num_elems * num_qpts,
                                     &d_density(0, 0, mat_idx), &d_energy(0, 0, mat_idx),
                                     &d_pressure(0, 0, mat_idx), &d_soundspeed2(0, 0, mat_idx),
                                     &d_bulkmod(0, 0, mat_idx), &d_temperature(0, 0, mat_idx));
#endif
            }
        }
        CALIPER(CALI_MARK_FUNCTION_END);
    }
};
}   // end of namespace
#endif
