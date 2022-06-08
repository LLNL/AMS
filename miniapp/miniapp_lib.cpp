#include <cstdio>
#include <cstdlib>
#include <vector>

#include <umpire/Umpire.hpp>

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "app/eos.hpp"
#include "app/eos_constant_on_host.hpp"
#include "app/eos_idealgas.hpp"
#include "app/mfem_utils.hpp"

#include "ml/hdcache.hpp"
#include "ml/surrogate.hpp"
#include "wf/utilities.hpp"

#include "miniapp.hpp"

//! ----------------------------------------------------------------------------
//! the main miniapp function that is exposed to the shared lib
extern "C" void miniapp_lib(bool is_cpu, const char *device_name, int stop_cycle,
                            bool pack_sparse_mats, int num_mats, int num_elems, int num_qpts,
                            const char *model_path, const std::string &eos_name, double *density_in,
                            double *energy_in, bool *indicators_in) {
    // create an object of the miniapp
    MiniApp miniapp(num_mats, num_elems, num_qpts, is_cpu, pack_sparse_mats);

    // setup device
    auto &rm = umpire::ResourceManager::getInstance();

    const bool use_device = device_name != "cpu";

    rm.makeAllocator<umpire::strategy::QuickPool, true>(AMS::utilities::getHostAllocatorName(),
                                                        rm.getAllocator("HOST"));
    mfem::MemoryManager::SetUmpireHostAllocatorName(AMS::utilities::getHostAllocatorName());
    if (use_device) {
        rm.makeAllocator<umpire::strategy::QuickPool, true>(
            AMS::utilities::getDeviceAllocatorName(), rm.getAllocator("DEVICE"));
        mfem::MemoryManager::SetUmpireDevice2AllocatorName(
            AMS::utilities::getDeviceAllocatorName());
    }

    mfem::Device::SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE, mfem::MemoryType::DEVICE_UMPIRE);

    std::cout << std::endl;
    mfem::Device device(device_name);
    device.Print();
    std::cout << std::endl;

    // -------------------------------------------------------------------------
    // setup eos, surrogare models, and hdcaches
    // TODO: keeping it consistent with Tom's existing code currently
    // Do we want different surrogates and caches for different materials?
    IdealGas *temp_eos = new IdealGas(1.6, 1.4);
    int cache_dim = 2;
    for (int mat_idx = 0; mat_idx < miniapp.num_mats; ++mat_idx) {
        {
            if (eos_name == "ideal_gas") {
                miniapp.eoses[mat_idx] = new IdealGas(1.6, 1.4);
            } else if (eos_name == "constant_host") {
                miniapp.eoses[mat_idx] =
                    new ConstantEOSOnHost(AMS::utilities::getHostAllocatorName(), 1.0);
            } else {
                std::cerr << "unknown eos `" << eos_name << "'" << std::endl;
                return;
            }
        }
        if (strlen(model_path) > 0) {
            miniapp.surrogates[mat_idx] = new SurrogateModel(temp_eos, model_path, is_cpu);
            miniapp.hdcaches[mat_idx] = new HDCache(cache_dim);
        } else {
            miniapp.surrogates[mat_idx] = nullptr;
            miniapp.hdcaches[mat_idx] = nullptr;
        }
    }

    // -------------------------------------------------------------------------
    // initialize inputs and outputs as mfem tensors
    // -------------------------------------------------------------------------
    // inputs

    // mfem::DenseTensor has shapw (i,j,k)
    //  contains k 2D "DenseMatrix" each of size ixj

#if 0
    // if the data has been defined in C order (mat x elems x qpts)
    // the following conversion works
    mfem::DenseTensor density(density_in, num_qpts, num_elems, num_mats);
    mfem::DenseTensor energy(energy_in, num_qpts, num_elems, num_mats);

#else
    mfem::DenseTensor density(num_qpts, num_elems, num_mats);
    mfem::DenseTensor energy(num_mats, num_elems, num_qpts);

    // init inputs
    density = 0;
    energy = 0;

    // TODO: what are good initial values?
    for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
        for (int elem_idx = 0; elem_idx < num_elems; ++elem_idx) {
            const int me = mat_idx * num_elems + elem_idx;
            if (!indicators_in[me]) {
                continue;
            }
            for (int qpt_idx = 0; qpt_idx < num_qpts; ++qpt_idx) {
                density(qpt_idx, elem_idx, mat_idx) = density_in[qpt_idx + me * num_qpts];
                energy(qpt_idx, elem_idx, mat_idx) = energy_in[qpt_idx + me * num_qpts];
            }
        }
    }
#endif

    if (0) {
        // print_tensor_array("indicators_in", indicators_in, {1, num_mats,
        // num_elems}); print_tensor_array("density_in", density_in, {num_mats,
        // num_elems, num_qpts}); print_tensor_array("energy_in", energy_in,
        // {num_mats, num_elems, num_qpts});

        // print_dense_tensor("density", density);
        // print_dense_tensor("density", density, indicators_in);
        // print_dense_tensor("energy", energy);
        // print_dense_tensor("energy", energy, indicators_in);
    }

    // outputs
    mfem::DenseTensor pressure(num_qpts, num_elems, num_mats);
    mfem::DenseTensor soundspeed2(num_qpts, num_elems, num_mats);
    mfem::DenseTensor bulkmod(num_qpts, num_elems, num_mats);
    mfem::DenseTensor temperature(num_qpts, num_elems, num_mats);

    // -------------------------------------------------------------------------
    // run through the cycles (time-steps)
    printf("\n");
    miniapp.start();
    for (int c = 0; c <= stop_cycle; ++c) {
        std::cout << "--> cycle " << c << std::endl;
        miniapp.evaluate(density, energy, indicators_in, pressure, soundspeed2, bulkmod,
                         temperature);
        // break;
    }

    // pressure.HostRead();
    // print_dense_tensor("pressure", pressure);
}

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
