#include "allocator.hpp"
#include <umpire/Umpire.hpp>

namespace ams {

// default allocator
ResourceManager::ResourceType ResourceManager::default_resource = ResourceManager::ResourceType::HOST;
int ResourceManager::allocator_ids[ResourceType::RSEND] = {-1, -1};

const std::string
ResourceManager::getDeviceAllocatorName() {  return "mmp-device-quickpool"; }

const std::string
ResourceManager::getHostAllocatorName() {    return "mmp-host-quickpool";   }


void
ResourceManager::setDefaultDataAllocator(ResourceManager::ResourceType location) {
    ResourceManager::default_resource = location;
}

ResourceManager::ResourceType
ResourceManager::getDefaultDataAllocator() {
    return ResourceManager::default_resource;
}



bool
ResourceManager::isDeviceExecution() {
    return ResourceManager::default_resource == ResourceManager::ResourceType::DEVICE;
}


void
ResourceManager::setup(bool use_device) {

    std::cout << "Setting up ams::ResourceManager\n";

    auto host_alloc_name = ResourceManager::getHostAllocatorName();
    auto device_alloc_name = ResourceManager::getDeviceAllocatorName();

    auto& rm = umpire::ResourceManager::getInstance();
    auto h_alloc = rm.makeAllocator<umpire::strategy::QuickPool, true>(host_alloc_name, rm.getAllocator("HOST"));
    allocator_ids[ResourceType::HOST] = h_alloc.getId();

    std::cout << "  > created allocator["<<ResourceType::HOST<<"] = "
              << h_alloc.getId() << ": " << h_alloc.getName() << "\n";

    if (use_device) {
        auto d_alloc = rm.makeAllocator<umpire::strategy::QuickPool, true>(device_alloc_name, rm.getAllocator("DEVICE"));
        allocator_ids[ResourceType::DEVICE] = d_alloc.getId();
        std::cout << "  > created allocator["<<ResourceType::DEVICE<<"] = "
                  << d_alloc.getId() << ": " << d_alloc.getName() << "\n";
    }

    // set the default
    if (use_device) {
        ResourceManager::setDefaultDataAllocator(ResourceType::DEVICE);
        std::cout << "  > default allocator = (" << device_alloc_name << ")\n";
    }
    else {
        ResourceManager::setDefaultDataAllocator(ResourceType::HOST);
        std::cout << "  default allocator = (" << host_alloc_name << ")\n";
    }
}
}  // namespace AMS
