#include <cstring>
#include <iostream>
#include <umpire/Umpire.hpp>
#include "wf/utilities.hpp"

using namespace AMS::utilities;

int main(int argc, char* argv[]) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto host_alloc_name = AMS::utilities::getHostAllocatorName();
    auto device_alloc_name = AMS::utilities::getDeviceAllocatorName();

    rm.makeAllocator<umpire::strategy::QuickPool, true>(host_alloc_name, rm.getAllocator("HOST"));
    rm.makeAllocator<umpire::strategy::QuickPool, true>(device_alloc_name,
                                                        rm.getAllocator("DEVICE"));

    if (strcmp(argv[1], "device") == 0) {
        std::cout << "Starting allocation[Done]\n";
        double* data = static_cast<double*>(
            AMS::utilities::allocate(sizeof(double), AMSDevice::DEVICE));
        auto found_allocator = rm.getAllocator(data);
        if (strcmp(getDeviceAllocatorName(), found_allocator.getName().data()) != 0) {
            std::cout << "Device Allocator Name" << getDeviceAllocatorName() << "Actual Allocation "
                      << found_allocator.getName() << "\n";
            return 1;
        }
        std::cout << "Explicit device allocation[Done]\n";
        
        deallocate(data, AMSDevice::DEVICE);
        std::cout << "Explicit device de-allocation[Done]\n";

        setDefaultDataAllocator(AMSDevice::DEVICE);

        if ( getDefaultDataAllocator() != AMSDevice::DEVICE ){
          std::cout<<"Default allocator not set correctly\n"; 
          return 2;
        }
        std::cout << "Set default allocator to device[Done]\n";

        data = static_cast<double*>(
            AMS::utilities::allocate(sizeof(double)));

        found_allocator = rm.getAllocator(data);
        if (strcmp(getDeviceAllocatorName(), found_allocator.getName().data()) != 0) {
            std::cout << "Device Allocator Name" << getDeviceAllocatorName() << "Actual Allocation "
                      << found_allocator.getName() << "\n";
            return 3;
        }
        std::cout << "Implicit device allocation [Done]\n";
    } else if (strcmp(argv[1], "host") == 1) {
        std::cout << "Starting allocation[Done]\n";
        double* data = static_cast<double*>(
            AMS::utilities::allocate(sizeof(double), AMSDevice::HOST));
        auto found_allocator = rm.getAllocator(data);
        if (strcmp(getHostAllocatorName(), found_allocator.getName().data()) != 0) {
            std::cout << "Host Allocator Name" << getHostAllocatorName() << "Actual Allocation "
                      << found_allocator.getName() << "\n";
            return 1;
        }
        std::cout << "Explicit device allocation[Done]\n";
        
        deallocate(data, AMSDevice::HOST);
        std::cout << "Explicit device de-allocation[Done]\n";

        setDefaultDataAllocator(AMSDevice::HOST);

        if ( getDefaultDataAllocator() != AMSDevice::HOST ){
          std::cout<<"Default allocator not set correctly\n"; 
          return 2;
        }
        std::cout << "Set default allocator to device[Done]\n";

        data = static_cast<double*>(
            AMS::utilities::allocate(sizeof(double)));

        found_allocator = rm.getAllocator(data);
        if (strcmp(getHostAllocatorName(), found_allocator.getName().data()) != 0) {
            std::cout << "Host Allocator Name" << getHostAllocatorName() << "Actual Allocation "
                      << found_allocator.getName() << "\n";
            return 3;
        }
        std::cout << "Implicit device allocation [Done]\n";
    }
}
