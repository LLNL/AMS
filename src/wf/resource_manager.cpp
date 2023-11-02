/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "resource_manager.hpp"

#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>

#include "debug.h"

namespace ams
{

std::string AMSAllocator::getName() { return allocator.getName(); }

void *AMSAllocator::allocate(size_t num_bytes)
{
  void *ptr = allocator.allocate(num_bytes);
  CFATAL(ResourceManager,
         ptr == nullptr,
         "Failed to allocated %ld values using allocator %s",
         num_bytes,
         getName().c_str());
  return ptr;
}

void AMSAllocator::deallocate(void *ptr) { allocator.deallocate(ptr); }

void AMSAllocator::registerPtr(void *ptr, size_t nBytes)
{
  auto &rm = umpire::ResourceManager::getInstance();
  rm.registerAllocation(ptr,
                        umpire::util::AllocationRecord(
                            ptr, nBytes, allocator.getAllocationStrategy()));
}

std::vector<AMSAllocator *> ResourceManager::RMAllocators = {nullptr,
                                                             nullptr,
                                                             nullptr};
// -----------------------------------------------------------------------------
// set up the resource manager
// -----------------------------------------------------------------------------
}  // namespace ams
