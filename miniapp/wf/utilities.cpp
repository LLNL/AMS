#include "utilities.hpp"
#include <umpire/Umpire.hpp>

namespace AMS {
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
}  // namespace AMS
