#include <cstring>
#include <iostream>
#include <umpire/Umpire.hpp>
#include "utils/allocator.hpp"


int main(int argc, char* argv[]) {
    using namespace ams;
    auto& rm = umpire::ResourceManager::getInstance();
    auto host_alloc_name = ams::ResourceManager::getHostAllocatorName().c_str();
    auto device_alloc_name = ams::ResourceManager::getDeviceAllocatorName().c_str();

    
    if (strcmp(argv[1], "device") == 0) {
        ams::ResourceManager::setup(1);
        std::cout << "Starting allocation[Done]\n";
        double* data = 
            ams::ResourceManager::allocate<double>(1, ResourceManager::ResourceType::DEVICE);
        auto found_allocator = rm.getAllocator(data);
        if (strcmp(ResourceManager::getDeviceAllocatorName().c_str(), found_allocator.getName().data()) != 0) {
            std::cout << "Device Allocator Name" << ResourceManager::getDeviceAllocatorName() << "Actual Allocation "
                      << found_allocator.getName() << "\n";
            return 1;
        }
        std::cout << "Explicit device allocation[Done]\n";
        
        ResourceManager::deallocate(data, ResourceManager::ResourceType::DEVICE);
        std::cout << "Explicit device de-allocation[Done]\n";

        ResourceManager::setDefaultDataAllocator(ResourceManager::ResourceType::DEVICE);

        if ( ResourceManager::getDefaultDataAllocator() != ResourceManager::ResourceType::DEVICE ){
          std::cout<<"Default allocator not set correctly\n"; 
          return 2;
        }
        std::cout << "Set default allocator to device[Done]\n";

        data = ams::ResourceManager::allocate<double>(1);

        found_allocator = rm.getAllocator(data);
        if (strcmp(ResourceManager::getDeviceAllocatorName().c_str(), found_allocator.getName().data()) != 0) {
            std::cout << "Device Allocator Name" << ResourceManager::getDeviceAllocatorName() << "Actual Allocation "
                      << found_allocator.getName() << "\n";
            return 3;
        }
        std::cout << "Implicit device allocation [Done]\n";
    } else if (strcmp(argv[1], "host") == 1) {
        ams::ResourceManager::setup(0);
        std::cout << "Starting allocation[Done]\n";
        double* data = ams::ResourceManager::allocate<double>(1, ResourceManager::ResourceType::HOST);
        auto found_allocator = rm.getAllocator(data);
        if (strcmp(ResourceManager::getHostAllocatorName().c_str(), found_allocator.getName().data()) != 0) {
            std::cout << "Host Allocator Name" << ResourceManager::getHostAllocatorName() << "Actual Allocation "
                      << found_allocator.getName() << "\n";
            return 1;
        }
        std::cout << "Explicit device allocation[Done]\n";
        
        ResourceManager::deallocate(data, ResourceManager::ResourceType::HOST);
        std::cout << "Explicit device de-allocation[Done]\n";

        ResourceManager::setDefaultDataAllocator(ResourceManager::ResourceType::HOST);

        if ( ResourceManager::getDefaultDataAllocator() != ResourceManager::ResourceType::HOST ){
          std::cout<<"Default allocator not set correctly\n"; 
          return 2;
        }
        std::cout << "Set default allocator to device[Done]\n";

        data = ams::ResourceManager::allocate<double>(1);

        found_allocator = rm.getAllocator(data);
        if (strcmp(ResourceManager::getHostAllocatorName().c_str(), found_allocator.getName().data()) != 0) {
            std::cout << "Host Allocator Name" << ResourceManager::getHostAllocatorName() << "Actual Allocation "
                      << found_allocator.getName() << "\n";
            return 3;
        }
        std::cout << "Implicit device allocation [Done]\n";
    }
}
