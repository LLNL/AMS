#include <AMS.h>

#include <cstring>
#include <iostream>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <wf/resource_manager.hpp>

int main(int argc, char* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();
  int device = std::atoi(argv[1]);

  AMSSetupAllocator(AMSResourceType::HOST);

  if (device == 1) {
    AMSSetupAllocator(AMSResourceType::DEVICE);
    AMSSetDefaultAllocator(AMSResourceType::DEVICE);
    std::cout << "Starting allocation[Done]\n";
    double* data =
        ams::ResourceManager::allocate<double>(1, AMSResourceType::DEVICE);
    auto found_allocator = rm.getAllocator(data);
    if (strcmp(ams::ResourceManager::getDeviceAllocatorName(),
               found_allocator.getName().data()) != 0) {
      std::cout << "Device Allocator Name"
                << ams::ResourceManager::getDeviceAllocatorName()
                << "Actual Allocation " << found_allocator.getName() << "\n";
      return 1;
    }
    std::cout << "Explicit device allocation[Done]\n";

    ams::ResourceManager::deallocate(data, AMSResourceType::DEVICE);
    std::cout << "Explicit device de-allocation[Done]\n";

    ams::ResourceManager::setDefaultDataAllocator(AMSResourceType::DEVICE);

    if (ams::ResourceManager::getDefaultDataAllocator() !=
        AMSResourceType::DEVICE) {
      std::cout << "Default allocator not set correctly\n";
      return 2;
    }
    std::cout << "Set default allocator to device[Done]\n";

    data = ams::ResourceManager::allocate<double>(1);

    found_allocator = rm.getAllocator(data);
    if (strcmp(ams::ResourceManager::getDeviceAllocatorName(),
               found_allocator.getName().data()) != 0) {
      std::cout << "Device Allocator Name"
                << ams::ResourceManager::getDeviceAllocatorName()
                << "Actual Allocation " << found_allocator.getName() << "\n";
      return 3;
    }
    std::cout << "Implicit device allocation [Done]\n";
  } else if (device == 0) {
    AMSSetDefaultAllocator(AMSResourceType::HOST);
    std::cout << "Starting allocation[Done]\n";
    double* data =
        ams::ResourceManager::allocate<double>(1, AMSResourceType::HOST);
    auto found_allocator = rm.getAllocator(data);
    if (strcmp(ams::ResourceManager::getHostAllocatorName(),
               found_allocator.getName().data()) != 0) {
      std::cout << "Host Allocator Name"
                << ams::ResourceManager::getHostAllocatorName()
                << "Actual Allocation " << found_allocator.getName() << "\n";
      return 1;
    }
    std::cout << "Explicit device allocation[Done]\n";

    ams::ResourceManager::deallocate(data, AMSResourceType::HOST);
    std::cout << "Explicit device de-allocation[Done]\n";

    ams::ResourceManager::setDefaultDataAllocator(AMSResourceType::HOST);

    if (ams::ResourceManager::getDefaultDataAllocator() !=
        AMSResourceType::HOST) {
      std::cout << "Default allocator not set correctly\n";
      return 2;
    }
    std::cout << "Set default allocator to device[Done]\n";

    data = ams::ResourceManager::allocate<double>(1);

    found_allocator = rm.getAllocator(data);
    if (strcmp(ams::ResourceManager::getHostAllocatorName(),
               found_allocator.getName().data()) != 0) {
      std::cout << "Host Allocator Name"
                << ams::ResourceManager::getHostAllocatorName()
                << "Actual Allocation " << found_allocator.getName() << "\n";
      return 3;
    }
    std::cout << "Implicit device allocation [Done]\n";
  }
}
