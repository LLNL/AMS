/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>

#include "resource_manager.hpp"

namespace ams
{

//! --------------------------------------------------------------------------
const char* ResourceManager::getDeviceAllocatorName()
{
  return "mmp-device-quickpool";
}

const char* ResourceManager::getHostAllocatorName()
{
  return "mmp-host-quickpool";
}

const char* ResourceManager::getPinnedAllocatorName()
{
  return "mmp-pinned-quickpool";
}

const char* ResourceManager::getAllocatorName(AMSResourceType Resource)
{
  if (Resource == AMSResourceType::HOST)
    return ResourceManager::getHostAllocatorName();
  else if (Resource == AMSResourceType::DEVICE)
    return ResourceManager::getDeviceAllocatorName();
  else if (Resource == AMSResourceType::PINNED)
    return ResourceManager::getPinnedAllocatorName();
  else {
    FATAL(ResourceManager, "Request allocator for resource that does not exist (%d)", Resource)
    return nullptr;
  }
}

//! --------------------------------------------------------------------------
// maintain a list of allocator ids
int ResourceManager::allocator_ids[AMSResourceType::RSEND] = {-1, -1, -1};

// maintain a list of allocator names
std::string ResourceManager::allocator_names[AMSResourceType::RSEND] = {"HOST",
                                                                        "DEVICE",
                                                                        "PINNED"};


// -----------------------------------------------------------------------------
// set up the resource manager
// -----------------------------------------------------------------------------
PERFFASPECT()
void ResourceManager::setup(const AMSResourceType Resource)
{
  if (Resource < AMSResourceType::HOST || Resource >= AMSResourceType::RSEND) {
    throw std::runtime_error("Resource does not exist\n");
  }

  // use umpire resource manager
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc_name = ResourceManager::getAllocatorName(Resource);
  auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
      alloc_name, rm.getAllocator(allocator_names[Resource]));

  DBG(ResourceManager,
      "Setting up ams::ResourceManager::%s:%d",
      allocator_names[Resource].c_str(),
      Resource);

  allocator_ids[Resource] = alloc_resource.getId();
}
}  // namespace ams
