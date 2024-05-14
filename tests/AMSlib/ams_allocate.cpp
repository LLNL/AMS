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
#include <umpire/strategy/QuickPool.hpp>
#include <wf/resource_manager.hpp>

int test_allocation(AMSResourceType resource, std::string pool_name)
{
  std::cout << "Testing Pool: " << pool_name << "\n";
  auto& rm = umpire::ResourceManager::getInstance();
  auto& ams_rm = ams::ResourceManager::getInstance();
  double* data = ams_rm.allocate<double>(1, resource);
  auto found_allocator = rm.getAllocator(data);
  if (ams_rm.getAllocatorName(resource) != found_allocator.getName()) {
    std::cout << "Allocator Name" << ams_rm.getAllocatorName(resource)
              << "Actual Allocation " << found_allocator.getName() << "\n";
    return 1;
  }


  if (ams_rm.getAllocatorName(resource) != pool_name) {
    std::cout << "Allocator Name" << ams_rm.getAllocatorName(resource)
              << "is not equal to pool name " << pool_name << "\n";
    return 1;
  }

  //  found_allocator = rm.getAllocator(data);
  //  if (ams_rm.getAllocatorName(resource) !=
  //      found_allocator.getName().data()) {
  //    std::cout << "Device Allocator Name"
  //              << ams_rm.getAllocatorName(resource)
  //              << "Actual Allocation " << found_allocator.getName() << "\n";
  //    return 3;
  //  }

  ams_rm.deallocate(data, resource);
  return 0;
}

int main(int argc, char* argv[])
{
  int device = std::atoi(argv[1]);

  // Testing with global umpire allocators
  auto& ams_rm = ams::ResourceManager::getInstance();
  ams_rm.init();
  if (device == 1) {
    if (test_allocation(AMSResourceType::AMS_DEVICE, "DEVICE") != 0) return 1;
  } else if (device == 0) {
    if (test_allocation(AMSResourceType::AMS_HOST, "HOST") != 0) return 1;
  }

  // Testing with pools

  if (device == 1) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
        "test-device", rm.getAllocator("DEVICE"));
    ams_rm.setAllocator("test-device", AMSResourceType::AMS_DEVICE);
    if (test_allocation(AMSResourceType::AMS_DEVICE, "test-device") != 0)
      return 1;
  } else if (device == 0) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
        "test-host", rm.getAllocator("HOST"));
    ams_rm.setAllocator("test-host", AMSResourceType::AMS_HOST);
    if (test_allocation(AMSResourceType::AMS_HOST, "test-host") != 0) return 1;
  }

  return 0;
}
