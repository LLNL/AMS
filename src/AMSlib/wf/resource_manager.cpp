/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <cstdlib>
#include <cstring>

#include "debug.h"
#include "device.hpp"
#include "resource_manager.hpp"

namespace ams
{

template <typename T>
static T roundUp(T num_to_round, int multiple)
{
  return ((num_to_round + multiple - 1) / multiple) * multiple;
}

const std::string AMSAllocator::getName() const { return name; }


struct AMSDefaultDeviceAllocator final : AMSAllocator {
  AMSDefaultDeviceAllocator(std::string name) : AMSAllocator(name){};
  ~AMSDefaultDeviceAllocator() = default;

  void *allocate(size_t num_bytes) { return DeviceAllocate(num_bytes); }

  void deallocate(void *ptr) { return DeviceFree(ptr); }
};

struct AMSDefaultHostAllocator final : AMSAllocator {
  AMSDefaultHostAllocator(std::string name) : AMSAllocator(name) {}
  ~AMSDefaultHostAllocator() = default;

  void *allocate(size_t num_bytes)
  {
    return aligned_alloc(8, roundUp(num_bytes, 8));
  }

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
  switch (src_dev) {
    case AMSResourceType::AMS_HOST:
    case AMSResourceType::AMS_PINNED:
      switch (dest_dev) {
        case AMSResourceType::AMS_HOST:
        case AMSResourceType::AMS_PINNED:
          std::memcpy(dest, src, num_bytes);
          break;
        case AMSResourceType::AMS_DEVICE:
          HtoDMemcpy(dest, src, num_bytes);
          break;
        default:
          FATAL(ResourceManager, "Unknown device type to copy to from HOST");
          break;
      }
      break;
    case AMSResourceType::AMS_DEVICE:
      switch (dest_dev) {
        case AMSResourceType::AMS_DEVICE:
          DtoDMemcpy(dest, src, num_bytes);
          break;
        case AMSResourceType::AMS_HOST:
        case AMSResourceType::AMS_PINNED:
          DtoHMemcpy(dest, src, num_bytes);
          break;
        default:
          FATAL(ResourceManager, "Unknown device type to copy to from DEVICE");
          break;
      }
    default:
      FATAL(ResourceManager, "Unknown device type to copy from");
  }
}

AMSAllocator *_get_allocator(std::string &alloc_name, AMSResourceType resource)
{
  switch (resource) {
    case AMSResourceType::AMS_DEVICE:
      return new AMSDefaultDeviceAllocator(alloc_name);
      break;
    case AMSResourceType::AMS_HOST:
      return new AMSDefaultHostAllocator(alloc_name);
      break;
    case AMSResourceType::AMS_PINNED:
      return new AMSDefaultPinnedAllocator(alloc_name);
      break;
    default:
      FATAL(ResourceManager,
            "Unknown resource type to create an allocator for");
  }
}

void _release_allocator(AMSAllocator *allocator) { delete allocator; }

}  // namespace internal
}  // namespace ams
