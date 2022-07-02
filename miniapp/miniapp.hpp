#ifndef __MINIAPP_HPP__
#define __MINIAPP_HPP__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/dtensor.hpp"
#include "umpire/ResourceManager.hpp"

using mfem::ForallWrap;

#include "app/eos.hpp"
#include "app/eos_constant_on_host.hpp"
#include "app/eos_idealgas.hpp"
#include "ml/hdcache.hpp"
#include "ml/surrogate.hpp"
#include "wf/workflow.hpp"
#include "utils/utils_data.hpp"
#include "utils/utils_mfem.hpp"
#include "utils/utils_caliper.hpp"
#include "utils/allocator.hpp"


#ifdef __ENABLE_FAISS__
  #include "ml/hdcache_faiss.hpp"
  template <typename T>
  using TypeHDCache = HDCache_Faiss<T>;
#else
  #include "ml/hdcache_random.hpp"
  template <typename T>
  using TypeHDCache = HDCache_Random<T>;
#endif



//! ----------------------------------------------------------------------------
//! mini app class
//! ----------------------------------------------------------------------------
template <typename TypeValue = double>
class MiniApp {
    using data_handler = DataHandler<TypeValue>;

private:
    const int num_mats;
    const int num_elems;
    const int num_qpts;
    const std::string device_name;
    const bool use_device;
    const bool pack_sparse_mats;


    AMSWorkflow *workflow;
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

        workflow = new AMSWorkflow(_num_mats);
        CALIPER(mgr.start();)
    }
    ~MiniApp() {
        delete workflow;
        CALIPER(mgr.flush());
    }

    // -------------------------------------------------------------------------
    // setup the miniapp
    // -------------------------------------------------------------------------
    void setup(const std::string eos_name, const std::string model_path) {

        // -------------------------------------------------------------------------
        // setup resource manager (data allocators)
        // -------------------------------------------------------------------------

        AMS::ResourceManager::setup(use_device);
        auto host_alloc_name = AMS::ResourceManager::getHostAllocatorName();
        auto device_alloc_name = AMS::ResourceManager::getDeviceAllocatorName();

        // set up mfem memory manager
        mfem::MemoryManager::SetUmpireHostAllocatorName(host_alloc_name.c_str());
        if (use_device) {
            mfem::MemoryManager::SetUmpireDevice2AllocatorName(device_alloc_name.c_str());
        }
        mfem::Device::SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE, mfem::MemoryType::DEVICE_UMPIRE);
        mfem::Device device(device_name);

        std::cout << std::endl;
        device.Print();
        std::cout << std::endl;


        // -------------------------------------------------------------------------
        // setup eos, surrogate models, and hdcaches
        // TODO: keeping it consistent with Tom's existing code currently
        // Do we want different surrogates and caches for different materials?
        // -------------------------------------------------------------------------
        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
            if (eos_name == "ideal_gas") {
                workflow->set_eos(mat_idx, new IdealGas(1.6, 1.4));
            } else if (eos_name == "constant_host") {
                workflow->set_eos(mat_idx, new ConstantEOSOnHost(host_alloc_name.c_str(), 1.0));
            } else {
                std::cerr << "unknown eos `" << eos_name << "'" << std::endl;
                return;
            }
        }

        const int cache_dim = 2;
        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
            if (model_path.size() > 0) {
                workflow->set_surrogate(mat_idx,
                                      new SurrogateModel<double>(model_path.c_str(), !use_device));
            } else {
                workflow->set_surrogate(mat_idx, nullptr);
            }
            workflow->set_hdcache(mat_idx, new TypeHDCache<TypeValue>(cache_dim, 10, false));
        }
    }

    // -------------------------------------------------------------------------
    // miniapp evaluation (all cycles)
    // -------------------------------------------------------------------------
    void evaluate(int stop_cycle,
                  TypeValue* density_in, TypeValue* energy_in,
                  const bool* indicators_in) {

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

        // -------------------------------------------------------------------------
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

        // -------------------------------------------------------------------------
        // run through the cycles (time-steps)
        for (int c = 0; c <= stop_cycle; ++c) {
            std::cout << "\n--> cycle " << c << std::endl;
            evaluate(density, energy,                   // inputs
                     sparse_elem_indices,               // aux data
                     pressure, soundspeed2,             // outputs
                     bulkmod, temperature);             // outputs
            // break;
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
                data_handler::pack_ij(mat_idx, num_qpts, num_elems_for_mat, offset_curr,
                                      d_sparse_elem_indices, d_density, d_dense_density, d_energy,
                                      d_dense_energy);
                CALIPER(CALI_MARK_END("SPARSE_TO_DENSE");)
                // -------------------------------------------------------------

                workflow->evaluate(num_elems_for_mat * num_qpts,
                                   &d_dense_density(0, 0), &d_dense_energy(0, 0),
                                   &d_dense_pressure(0, 0), &d_dense_soundspeed2(0, 0),
                                   &d_dense_bulkmod(0, 0), &d_dense_temperature(0, 0),
                                   mat_idx);

                // -------------------------------------------------------------
                // dense -> sparse
                CALIPER(CALI_MARK_BEGIN("DENSE_TO_SPARSE");)
                data_handler::unpack_ij(mat_idx, num_qpts, num_elems_for_mat, offset_curr,
                                        d_sparse_elem_indices, d_dense_pressure, d_pressure,
                                        d_dense_soundspeed2, d_soundspeed2, d_dense_bulkmod,
                                        d_bulkmod, d_dense_temperature, d_temperature);
                CALIPER(CALI_MARK_END("DENSE_TO_SPARSE");)
                // -------------------------------------------------------------

            } else {
                workflow->evaluate(num_elems * num_qpts,
                                   const_cast<TypeValue*>(&d_density(0, 0, mat_idx)),
                                   const_cast<TypeValue*>(&d_energy(0, 0, mat_idx)),
                                   &d_pressure(0, 0, mat_idx), &d_soundspeed2(0, 0, mat_idx),
                                   &d_bulkmod(0, 0, mat_idx), &d_temperature(0, 0, mat_idx),
                                   mat_idx);
            }
        }

        CALIPER(CALI_MARK_FUNCTION_END);
    }
};

#endif
