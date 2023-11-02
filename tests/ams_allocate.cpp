/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

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

  if (device == 1) {
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


    found_allocator = rm.getAllocator(data);
    if (strcmp(ams::ResourceManager::getDeviceAllocatorName(),
               found_allocator.getName().data()) != 0) {
      std::cout << "Device Allocator Name"
                << ams::ResourceManager::getDeviceAllocatorName()
                << "Actual Allocation " << found_allocator.getName() << "\n";
      return 3;
    }

    ams::ResourceManager::deallocate(data, AMSResourceType::DEVICE);
    std::cout << "Explicit device de-allocation[Done]\n";
  } else if (device == 0) {
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


    found_allocator = rm.getAllocator(data);
    if (strcmp(ams::ResourceManager::getHostAllocatorName(),
               found_allocator.getName().data()) != 0) {
      std::cout << "Host Allocator Name"
                << ams::ResourceManager::getHostAllocatorName()
                << "Actual Allocation " << found_allocator.getName() << "\n";
      return 3;
    }
    data = ams::ResourceManager::allocate<double>(1, AMSResourceType::HOST);
    std::cout << "Implicit device allocation [Done]\n";
  }
}
