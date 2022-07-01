#include <cstdio>
#include <cstdlib>
#include <vector>

#include <umpire/Umpire.hpp>

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "app/eos.hpp"
#include "app/eos_constant_on_host.hpp"
#include "app/eos_idealgas.hpp"
#include "ml/hdcache.hpp"
#include "ml/surrogate.hpp"

#include "utils/utils_data.hpp"
#include "utils/utils_mfem.hpp"
#include "utils/allocator.hpp"

#include "miniapp.hpp"


#ifdef __ENABLE_FAISS__
  #include "ml/hdcache_faiss.hpp"
  template <typename T>
  using TypeHDCache = HDCache_Faiss<T>;
#else
  #include "ml/hdcache_random.hpp"
  template <typename T>
  using TypeHDCache = HDCache_Random<T>;
#endif

using TypeValue = double;

//! ----------------------------------------------------------------------------
//! the main miniapp function that is exposed to the shared lib
extern "C" void miniapp_lib(const std::string& device_name, const std::string& eos_name,
                            const std::string& model_path, int stop_cycle, bool pack_sparse_mats,
                            int num_mats, int num_elems, int num_qpts, double* density_in,
                            double* energy_in, bool* indicators_in) {

    // dimensions of input data is assumed to be (num_mats, num_elems, num_qpts)
    if (0) {
        print_tensor_array("density_in", density_in, {num_mats, num_elems, num_qpts});
        print_tensor_array("energy_in", energy_in, {num_mats, num_elems, num_qpts});
        print_tensor_array("indicators_in", indicators_in, {1, num_mats, num_elems});
    }

    const bool use_device = device_name != "cpu";

    // -------------------------------------------------------------------------
    // setup device
    auto& rm = umpire::ResourceManager::getInstance();

    auto host_alloc_name = AMS::utilities::getHostAllocatorName();
    auto device_alloc_name = AMS::utilities::getDeviceAllocatorName();

    rm.makeAllocator<umpire::strategy::QuickPool, true>(host_alloc_name, rm.getAllocator("HOST"));
    mfem::MemoryManager::SetUmpireHostAllocatorName(host_alloc_name);
    if (use_device) {
        std::cout << " Setting up default allocator to: " << device_alloc_name << "\n";
        rm.makeAllocator<umpire::strategy::QuickPool, true>(device_alloc_name,
                                                            rm.getAllocator("DEVICE"));
        mfem::MemoryManager::SetUmpireDevice2AllocatorName(device_alloc_name);
        AMS::utilities::setDefaultDataAllocator(AMS::utilities::AMSDevice::DEVICE);
    }

    mfem::Device::SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE, mfem::MemoryType::DEVICE_UMPIRE);

    mfem::Device device(device_name);

    std::cout << std::endl;
    device.Print();
    std::cout << std::endl;

    // -------------------------------------------------------------------------
    // create an object of the miniapp
    MiniApp miniapp(num_mats, num_elems, num_qpts, !use_device, pack_sparse_mats);

    // -------------------------------------------------------------------------
    // setup eos, surrogare models, and hdcaches
    // TODO: keeping it consistent with Tom's existing code currently
    // Do we want different surrogates and caches for different materials?
    const int cache_dim = 2;
    for (int mat_idx = 0; mat_idx < miniapp.num_mats; ++mat_idx) {
        {
            if (eos_name == "ideal_gas") {
                miniapp.eoses[mat_idx] = new IdealGas(1.6, 1.4);
            } else if (eos_name == "constant_host") {
                miniapp.eoses[mat_idx] = new ConstantEOSOnHost(host_alloc_name, 1.0);
            } else {
                std::cerr << "unknown eos `" << eos_name << "'" << std::endl;
                return;
            }
        }
        if (model_path.size() > 0) {
            miniapp.surrogates[mat_idx] =
                new SurrogateModel<double>(model_path.c_str(), !use_device);
        } else {
            miniapp.surrogates[mat_idx] = nullptr;
        }
        miniapp.hdcaches[mat_idx] = new TypeHDCache<TypeValue>(cache_dim, 10, false);
    }

    // -------------------------------------------------------------------------
    // create fake data for training the faiss index!
    if (0) {
        std::cout << " Creating fake data for training faiss index!\n";
        const size_t nfakedata = 159744;
        std::vector<TypeValue*> fake_data(2);
        for (int f = 0; f < 2; f++) {
          fake_data[f] = new TypeValue[nfakedata];
          for (int i = 0; i < nfakedata; i++) {
            fake_data[f][i] = .1 + unitrand();
          }
        }
        miniapp.hdcaches[0]->train(nfakedata, fake_data);
        miniapp.hdcaches[0]->add(nfakedata, fake_data);
        miniapp.hdcaches[0]->save_cache("test_index.idx");
    }
    else {
      const std::string idx_path = "/usr/workspace/AMS/miniapp_resources/test_index.idx";
      for (int mat_idx = 0; mat_idx < miniapp.num_mats; ++mat_idx) {
        miniapp.hdcaches[mat_idx]->load_cache(idx_path);
      }
    }
    // -------------------------------------------------------------------------

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
    miniapp.start();
    for (int c = 0; c <= stop_cycle; ++c) {
        std::cout << "\n--> cycle " << c << std::endl;
        miniapp.evaluate(density, energy, sparse_elem_indices, pressure, soundspeed2, bulkmod,
                         temperature);
        // break;
    }

    // pressure.HostRead();
    // print_dense_tensor("pressure", pressure);
}

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
