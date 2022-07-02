#include "allocator.hpp"
#include <umpire/Umpire.hpp>

namespace AMS {


#ifdef USE_NEW_ALLOCATOR

// default allocator
ResourceManager::ResourceType ResourceManager::default_resource = ResourceManager::ResourceType::HOST;

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

    std::cout << "Setting up ams::allocator\n";

    auto host_alloc_name = ResourceManager::getHostAllocatorName();
    auto device_alloc_name = ResourceManager::getDeviceAllocatorName();


    auto& rm = umpire::ResourceManager::getInstance();
    rm.makeAllocator<umpire::strategy::QuickPool, true>(host_alloc_name, rm.getAllocator("HOST"));
    if (use_device) {
        rm.makeAllocator<umpire::strategy::QuickPool, true>(device_alloc_name, rm.getAllocator("DEVICE"));
    }

    // set the default
    if (use_device) {
        ResourceManager::setDefaultDataAllocator(ResourceType::DEVICE);
        std::cout << "  default allocator = (" << device_alloc_name << ")\n";
    }
    else {
        ResourceManager::setDefaultDataAllocator(ResourceType::HOST);
        std::cout << "  default allocator = (" << host_alloc_name << ")\n";
    }
}
#endif






#ifndef USE_NEW_ALLOCATOR
namespace utilities {

AMSDevice defaultDloc = AMSDevice::HOST;

void setDefaultDataAllocator(AMSDevice location) {
    defaultDloc = location;
}

AMSDevice getDefaultDataAllocator() {
    return defaultDloc;
}

const char* getDeviceAllocatorName() {
    return "mmp-device-quickpool";
}

const char* getHostAllocatorName() {
    return "mmp-host-quickpool";
}

void deallocate(void* ptr) {
    static auto& rm = umpire::ResourceManager::getInstance();
    if (defaultDloc == HOST) {
        static auto cpuAllocator = rm.getAllocator(getHostAllocatorName());
        cpuAllocator.deallocate(ptr);
    } else if (defaultDloc == DEVICE) {
        static auto deviceAllocator = rm.getAllocator(getDeviceAllocatorName());
        deviceAllocator.deallocate(ptr);
    }
}

void deallocate(void* ptr, AMSDevice dev) {
    static auto& rm = umpire::ResourceManager::getInstance();
    if (dev == HOST) {
        static auto cpuAllocator = rm.getAllocator(getHostAllocatorName());
        cpuAllocator.deallocate(ptr);
    } else if (dev == DEVICE) {
        static auto deviceAllocator = rm.getAllocator(getDeviceAllocatorName());
        deviceAllocator.deallocate(ptr);
    }
}

void* allocate(size_t bytes) {
    static auto& rm = umpire::ResourceManager::getInstance();
    if (defaultDloc == HOST) {
        static auto cpuAllocator = rm.getAllocator(getHostAllocatorName());
        return cpuAllocator.allocate(bytes);
    } else if (defaultDloc == DEVICE) {
        static auto deviceAllocator = rm.getAllocator(getDeviceAllocatorName());
        return deviceAllocator.allocate(bytes);
    }
    return nullptr;
}

void* allocate(size_t bytes, AMSDevice dev) {
    static auto& rm = umpire::ResourceManager::getInstance();
    if (dev == HOST) {
        static auto cpuAllocator = rm.getAllocator(getHostAllocatorName());
        return cpuAllocator.allocate(bytes);
    } else if (dev == DEVICE) {
        static auto deviceAllocator = rm.getAllocator(getDeviceAllocatorName());
        return deviceAllocator.allocate(bytes);
    }
    return nullptr;
}

const char* getDefaultAllocatorName() {
    switch (defaultDloc) {
        case AMSDevice::HOST:
            return getHostAllocatorName();
        case AMSDevice::DEVICE:
            return getDeviceAllocatorName();
    }
    return "unknown";
}

bool isDeviceExecution() {
    return defaultDloc == DEVICE;
}
}  // namespace utilities
#endif
}  // namespace AMS
