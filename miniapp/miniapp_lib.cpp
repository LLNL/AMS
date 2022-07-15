#include <cstdio>
#include <cstdlib>

#include <mfem.hpp>

#include "miniapp.hpp"
#include "miniapp_lib.hpp"
#include "wf/resource_manager.hpp"


//! ----------------------------------------------------------------------------
//! the main miniapp function that is exposed to the shared lib
//! ----------------------------------------------------------------------------
extern "C" void miniapp_lib(const std::string& device_name,
                            const std::string& eos_name,
                            const std::string& model_path,
                            const std::string &hdcache_path,
                            int stop_cycle, bool pack_sparse_mats,
                            int num_mats, int num_elems, int num_qpts,
                            TypeValue* density_in, TypeValue* energy_in,
                            bool* indicators_in) {

    // -------------------------------------------------------------------------
    // setting up data allocators
    // -------------------------------------------------------------------------
    ams::ResourceManager::setup(device_name);


    // -------------------------------------------------------------------------
    // mfem memory manager
    // -------------------------------------------------------------------------
    // hardcoded names!
    const std::string &alloc_name_host = ams::ResourceManager::getHostAllocatorName();
    const std::string &alloc_name_device = ams::ResourceManager::getDeviceAllocatorName();

    mfem::MemoryManager::SetUmpireHostAllocatorName(alloc_name_host.c_str());
    if (device_name != "cpu")
        mfem::MemoryManager::SetUmpireDeviceAllocatorName(alloc_name_device.c_str());

    mfem::Device::SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE, mfem::MemoryType::DEVICE_UMPIRE);

    mfem::Device device(device_name);
    std::cout << std::endl;
    device.Print();
    std::cout << std::endl;

    ams::ResourceManager::list_allocators();

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    ams::MiniApp<TypeValue> miniapp(num_mats, num_elems, num_qpts, device_name, pack_sparse_mats);
    miniapp.setup(eos_name, model_path, hdcache_path);
    miniapp.evaluate(stop_cycle, density_in, energy_in, indicators_in);
}

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
