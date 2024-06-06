/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <cstdlib>
#include <cstring>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>

#include "debug.h"
#include "device.hpp"
#include "resource_manager.hpp"

namespace ams
{

template<typename T>
static T roundUp(T num_to_round, int multiple) 
{
    return ((num_to_round + multiple- 1) / multiple) * multiple;
}

std::string AMSAllocator::getName() { return name; }


struct AMSDefaultDeviceAllocator final : AMSAllocator {
  AMSDefaultDeviceAllocator(std::string name) : AMSAllocator(name){};
  ~AMSDefaultDeviceAllocator() = default;

  void *allocate(size_t num_bytes) { return DeviceAllocate(num_bytes); }

  void deallocate(void *ptr) { return DeviceFree(ptr); }
};

struct AMSDefaultHostAllocator final : AMSAllocator {
  AMSDefaultHostAllocator(std::string name) : AMSAllocator(name) {}
  ~AMSDefaultHostAllocator() = default;

  void *allocate(size_t num_bytes) { return aligned_alloc(8, roundUp(num_bytes, 8)); }

  void deallocate(void *ptr) { free(ptr); }
};

struct AMSDefaultPinnedAllocator final : AMSAllocator {
  AMSDefaultPinnedAllocator(std::string name) : AMSAllocator(name) {}
  ~AMSDefaultPinnedAllocator() = default;

  void *allocate(size_t num_bytes) { return DevicePinnedAlloc(num_bytes); }

  void deallocate(void *ptr) { DeviceFreePinned(ptr); }
};


namespace internal
{
void _raw_copy(void *src,
               AMSResourceType src_dev,
               void *dest,
               AMSResourceType dest_dev,
               size_t num_bytes)
{
  if (src_dev == AMSResourceType::AMS_HOST) {
    if (dest_dev == AMSResourceType::AMS_HOST) {
      std::memcpy(dest, src, num_bytes);
    } else if (dest_dev == AMSResourceType::AMS_DEVICE) {
      HtoDMemcpy(dest, src, num_bytes);
    } else if (dest_dev == AMSResourceType::AMS_PINNED) {
      std::memcpy(dest, src, num_bytes);
    } else {
      FATAL(AMSResource, "Unknown copy dest")
    }
  } else if (src_dev == AMSResourceType::AMS_DEVICE) {
    if (dest_dev == AMSResourceType::AMS_HOST) {
      DtoHMemcpy(dest, src, num_bytes);
    } else if (dest_dev == AMSResourceType::AMS_DEVICE) {
      DtoDMemcpy(dest, src, num_bytes);
    } else if (dest_dev == AMSResourceType::AMS_PINNED) {
      DtoHMemcpy(dest, src, num_bytes);
    } else {
      FATAL(AMSResource, "Unknown copy dest")
    }
  } else if (src_dev == AMSResourceType::AMS_PINNED) {
    if (dest_dev == AMSResourceType::AMS_HOST) {
      std::memcpy(dest, src, num_bytes);
    } else if (dest_dev == AMSResourceType::AMS_DEVICE) {
      HtoDMemcpy(dest, src, num_bytes);
    } else if (dest_dev == AMSResourceType::AMS_PINNED) {
      std::memcpy(dest, src, num_bytes);
    } else {
      FATAL(AMSResource, "Unknown copy dest")
    }
  }
}

AMSAllocator *_get_allocator(std::string alloc_name, AMSResourceType resource)
{
  if (resource == AMSResourceType::AMS_DEVICE) {
    return new AMSDefaultDeviceAllocator(alloc_name);
  } else if (resource == AMSResourceType::AMS_HOST) {
    return new AMSDefaultHostAllocator(alloc_name);
  } else if (resource == AMSResourceType::AMS_PINNED) {
    return new AMSDefaultPinnedAllocator(alloc_name);
  } else {
    FATAL(ResourceManager,
          "Requested allocator %s for Unknown resource type",
          alloc_name.c_str());
  }
}

void _release_allocator(AMSAllocator *allocator) { delete allocator; }

}  // namespace internal
}  // namespace ams
