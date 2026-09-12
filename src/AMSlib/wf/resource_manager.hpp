/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_ALLOCATOR__
#define __AMS_ALLOCATOR__

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "AMS.h"
#include "macro.h"
#include "wf/debug.h"


namespace ams
{
// Forward decl.
struct AMSAllocator;
namespace internal
{
void _raw_copy(void* src,
               AMSResourceType src_dev,
               void* dest,
               AMSResourceType dest_dev,
               size_t num_bytes);

AMSAllocator* _get_allocator(std::string& alloc_name, AMSResourceType resource);
void _release_allocator(AMSAllocator* allocator);

}  // namespace internal
/**
 * @brief A "utility" class that provides
 * a unified interface to the umpire library for memory allocations
 * and data movements/copies.
 */


struct AMSAllocator {
  std::string name;
  AMSAllocator(std::string& alloc_name) : name(alloc_name) {}
  virtual ~AMSAllocator() = default;

  virtual void* allocate(size_t num_bytes, size_t alignment) = 0;
  virtual void deallocate(void* ptr) = 0;

  const std::string getName() const;
};


class ResourceManager
{
private:
  /** @brief  Used internally to map resource types (Device, host, pinned memory) to
   * umpire allocator ids. */
  mutable std::vector<std::shared_ptr<AMSAllocator>> RMAllocators;
  mutable std::mutex Mutex;
  ResourceManager() : RMAllocators({nullptr, nullptr, nullptr}) {};

public:
  ~ResourceManager() = default;
  ResourceManager(const ResourceManager&) = delete;
  ResourceManager(ResourceManager&&) = delete;
  ResourceManager& operator=(const ResourceManager&) = delete;
  ResourceManager& operator=(ResourceManager&&) = delete;

  static ResourceManager& getInstance()
  {
    static ResourceManager instance;
    return instance;
  }

  /** @brief return the name of an allocator */
  const std::string getAllocatorName(AMSResourceType resource) const
  {
    return getAllocator(resource)->getName();
  }

  std::shared_ptr<AMSAllocator> getAllocator(AMSResourceType resource) const
  {
    if (resource < AMS_HOST || resource > AMS_PINNED)
      throw std::invalid_argument("Invalid AMS memory resource");
    std::lock_guard<std::mutex> lock(Mutex);
    auto allocator = RMAllocators[resource];
    if (!allocator) {
      if (resource != AMS_HOST) {
#if !defined(__AMS_ENABLE_CUDA__) && !defined(__AMS_ENABLE_HIP__)
        throw std::invalid_argument(
            "Requested AMS device/pinned memory resource is unavailable");
#endif
      }
      std::string name = resource == AMS_HOST     ? "HOST"
                         : resource == AMS_DEVICE ? "DEVICE"
                                                  : "PINNED";
      allocator.reset(ams::internal::_get_allocator(name, resource));
      RMAllocators[resource] = allocator;
    }
    return allocator;
  }

  /** @brief Allocates nvalues on the specified device.
   *  @tparam TypeInValue The type of pointer to allocate.
   *  @param[in] nvalues Number of elements to allocate.
   *  @param[in] dev Resource to allocate memory from.
   *  @return Pointer to allocated elements.
   */
  template <typename TypeInValue>
  PERFFASPECT()
  TypeInValue* allocate(size_t nvalues,
                        AMSResourceType dev,
                        size_t alignment = sizeof(TypeInValue))
  {
    auto allocator = getAllocator(dev);
    return static_cast<TypeInValue*>(
        allocator->allocate(nvalues * sizeof(TypeInValue), alignment));
  }

  /** @brief deallocates pointer from the specified device.
   *  @tparam TypeInValue The type of pointer to de-allocate.
   *  @param[in] data pointer to deallocate.
   *  @param[in] dev device to de-allocate from .
   *  @return void.
   */
  template <typename TypeInValue>
  PERFFASPECT()
  void deallocate(TypeInValue* data, AMSResourceType dev)
  {
    getAllocator(dev)->deallocate(data);
  }

  /** @brief copy values from src to destination regardless of their memory location.
   *  @tparam TypeInValue type of pointers
   *  @param[in] src Source memory pointer.
   *  @param[out] dest destination memory pointer.
   *  @param[in] size number of values to copy (It performs a shallow copy of nested pointers).
   *  @return void.
   */
  template <typename TypeInValue>
  PERFFASPECT()
  void copy(TypeInValue* src,
            AMSResourceType src_dev,
            TypeInValue* dest,
            AMSResourceType dest_dev,
            size_t nvalues)
  {
    ams::internal::_raw_copy(static_cast<void*>(src),
                             src_dev,
                             static_cast<void*>(dest),
                             dest_dev,
                             nvalues * sizeof(TypeInValue));
  }

  /** @brief Utility function that deallocates all C-Vectors inside the vector.
   *  @tparam TypeInValue type of pointers
   *  @param[in] dPtr vector containing pointers to C-vectors to be allocated.
   *  @return void.
   */
  template <typename T>
  void deallocate(std::vector<T*>& dPtr, AMSResourceType resource)
  {
    for (auto* I : dPtr)
      getAllocator(resource)->deallocate(I);
  }

  void init()
  {
    AMS_DBG(ResourceManager, "Initialization of allocators");
    std::string host_alloc("HOST");
    std::string device_alloc("DEVICE");
    std::string pinned_alloc("PINNED");
    if (!RMAllocators[AMSResourceType::AMS_HOST])
      setAllocator(host_alloc, AMSResourceType::AMS_HOST);
#if defined(__AMS_ENABLE_CUDA__) || defined(__AMS_ENABLE_HIP__)
    if (!RMAllocators[AMSResourceType::AMS_DEVICE])
      setAllocator(device_alloc, AMSResourceType::AMS_DEVICE);

    if (!RMAllocators[AMSResourceType::AMS_PINNED])
      setAllocator(pinned_alloc, AMSResourceType::AMS_PINNED);
#endif
  }

  void setAllocator(std::string& alloc_name, AMSResourceType resource)
  {
    if (resource < AMS_HOST || resource > AMS_PINNED)
      throw std::invalid_argument("Invalid AMS memory resource");
    std::lock_guard<std::mutex> lock(Mutex);
    RMAllocators[resource].reset(
        ams::internal::_get_allocator(alloc_name, resource));
    AMS_DBG(ResourceManager,
            "Set Allocator [{}] to pool with name : {}",
            resource,
            RMAllocators[resource]->getName());
  }

  bool isActive(AMSResourceType resource)
  {
    if (resource < AMS_HOST || resource > AMS_PINNED) return false;
    std::lock_guard<std::mutex> lock(Mutex);
    return RMAllocators[resource] != nullptr;
  }

  /** @brief Returns the memory consumption of the given resource as measured from Umpire.
   *  @param[in] resource The memory pool to get the consumption from.
   *  @param[out] wm the highest memory allocation that umpire has performed until now.
   *  @param[out] cs The current size of the pool. This can be smaller than the actual size.
   *  @param[out] as The actual size of the pool..
   *  @return void.
   */
  void getAllocatorStats(AMSResourceType resource,
                         size_t& wm,
                         size_t& cs,
                         size_t& as)
  {
    return;
  }

  //! ------------------------------------------------------------------------
};


}  // namespace ams

#endif
