/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_ALLOCATOR__
#define __AMS_ALLOCATOR__

#include <cstddef>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>

#include "AMS.h"
#include "wf/debug.h"


namespace ams
{
/**
 * @brief A "utility" class that provides
 * a unified interface to the umpire library for memory allocations
 * and data movements/copies.
 */
class ResourceManager
{
public:
private:
  /** @brief  Used internally to map resource types (Device, host, pinned memory) to
   * umpire allocator ids. */
  static int allocator_ids[AMSResourceType::RSEND];

  /** @brief The names of the user defined allocators */
  static std::string allocator_names[AMSResourceType::RSEND];

public:
  ResourceManager() = delete;
  ResourceManager(const ResourceManager&) = delete;
  ResourceManager(ResourceManager&&) = delete;
  ResourceManager& operator=(const ResourceManager&) = delete;
  ResourceManager& operator=(ResourceManager&&) = delete;

  /** @brief The names of the user defined allocators */
  static const char* getDeviceAllocatorName();

  /** @brief Get the name of the Host allocator */
  static const char* getHostAllocatorName();

  /** @brief Get the name of the Pinned memory Allocator */
  static const char* getPinnedAllocatorName();

  /** @brief Get the name of the Pinned memory Allocator */
  static const char* getAllocatorName(AMSResourceType Resource);

  /** @brief setup allocators in the resource manager */
  static void setup(const AMSResourceType resource);

  /** @brief Check if pointer is allocatd through
   *  @tparam TypeInValue The type of pointer being tested.
   *  @param[in] data pointer to memory.
   *  @return Boolean value describing whether the pointer has
   *  been allocated through an internal allocator
   */
  template <typename TypeInValue>
  static bool hasAllocator(const TypeInValue* data)
  {
    static auto& rm = umpire::ResourceManager::getInstance();
    void* vdata = static_cast<void*>(const_cast<TypeInValue*>(data));
    return rm.hasAllocator(vdata);
  }

  /** @brief Returns the id of the allocator allocated the defined memory.
   *  @tparam TypeInValue The type of pointer being tested.
   *  @param[in] data pointer to memory.
   *  @return Allocator Id.
   */
  template <typename TypeInValue>
  static int getDataAllocationId(const TypeInValue* data)
  {
    static auto& rm = umpire::ResourceManager::getInstance();
    void* vdata = static_cast<void*>(const_cast<TypeInValue*>(data));
    return hasAllocator(vdata) ? rm.getAllocator(vdata).getId() : -1;
  }

  /** @brief Returns the name of the allocator allocated the defined memory.
   *  @tparam TypeInValue The type of pointer being tested.
   *  @param[in] data pointer to memory.
   *  @return Allocator name if allocated through umpire else "unknown".
   */
  template <typename TypeInValue>
  static const std::string getDataAllocationName(const TypeInValue* data)
  {
    static auto& rm = umpire::ResourceManager::getInstance();
    void* vdata = static_cast<void*>(const_cast<TypeInValue*>(data));
    return rm.hasAllocator(vdata) ? rm.getAllocator(vdata).getName()
                                  : "unknown";
  }

  /** @brief Allocates nvalues on the specified device.
   *  @tparam TypeInValue The type of pointer to allocate.
   *  @param[in] nvalues Number of elements to allocate.
   *  @param[in] dev Resource to allocate memory from.
   *  @return Pointer to allocated elements.
   */
  template <typename TypeInValue>
  PERFFASPECT()
  static TypeInValue* allocate(size_t nvalues, AMSResourceType dev)
  {
    static auto& rm = umpire::ResourceManager::getInstance();
    DBG(ResourceManager,
        "Requesting to allocate %ld values using allocator :%s",
        nvalues,
        getAllocatorName(dev));
    auto alloc = rm.getAllocator(allocator_ids[dev]);
    TypeInValue* ret = static_cast<TypeInValue*>(
        alloc.allocate(nvalues * sizeof(TypeInValue)));
    CFATAL(ResourceManager,
           ret == nullptr,
           "Failed to allocated %ld values on device %d",
           nvalues,
           dev);
    return ret;
  }

  /** @brief deallocates pointer from the specified device.
   *  @tparam TypeInValue The type of pointer to de-allocate.
   *  @param[in] data pointer to deallocate.
   *  @param[in] dev device to de-allocate from .
   *  @return void.
   */
  template <typename TypeInValue>
  PERFFASPECT()
  static void deallocate(TypeInValue* data, AMSResourceType dev)
  {
    static auto& rm = umpire::ResourceManager::getInstance();
    if (hasAllocator(data)) {
      rm.getAllocator(allocator_ids[dev]).deallocate(data);
    }
  }

  /** @brief registers an external pointer in the umpire allocation records.
   *  @param[in] ptr pointer to memory to register.
   *  @param[in] nBytes number of bytes to register.
   *  @param[in] dev resource to register the memory to.
   *  @return void.
   */
  PERFFASPECT()
  static void registerExternal(void* ptr, size_t nBytes, AMSResourceType dev)
  {
    auto& rm = umpire::ResourceManager::getInstance();
    auto alloc = rm.getAllocator(allocator_ids[dev]);
    rm.registerAllocation(ptr,
                          umpire::util::AllocationRecord(
                              ptr, nBytes, alloc.getAllocationStrategy()));
  }

  /** @brief removes a registered external pointer from the umpire allocation records.
   *  @param[in] ptr pointer to memory to de-register.
   *  @return void.
   */
  static void deregisterExternal(void* ptr)
  {
    auto& rm = umpire::ResourceManager::getInstance();
    rm.deregisterAllocation(ptr);
  }

  /** @brief copy values from src to destination regardless of their memory location.
   *  @tparam TypeInValue type of pointers
   *  @param[in] src Source memory pointer.
   *  @param[out] dest destination memory pointer.
   *  @param[in] size number of bytes to copy. (When 0 copies entire allocated area)
   *  @return void.
   */
  template <typename TypeInValue>
  PERFFASPECT()
  static void copy(TypeInValue* src, TypeInValue* dest, size_t size = 0)
  {
    static auto& rm = umpire::ResourceManager::getInstance();
    rm.copy(dest, src, size);
  }

  /** @brief Utility function that deallocates all C-Vectors inside the vector.
   *  @tparam TypeInValue type of pointers
   *  @param[in] dPtr vector containing pointers to C-vectors to be allocated.
   *  @return void.
   */
  template <typename T>
  static void deallocate(std::vector<T*>& dPtr)
  {
    for (auto* I : dPtr)
      deallocate(I);
  }

  /** @brief Returns the memory consumption of the given resource as measured from Umpire.
   *  @param[in] resource The memory pool to get the consumption from.
   *  @param[out] wm the highest memory allocation that umpire has performed until now.
   *  @param[out] cs The current size of the pool. This can be smaller than the actual size.
   *  @param[out] as The actual size of the pool..
   *  @return void.
   */
  static void getAllocatorStats(AMSResourceType resource,
                                size_t& wm,
                                size_t& cs,
                                size_t& as)
  {
    auto& rm = umpire::ResourceManager::getInstance();
    auto alloc = rm.getAllocator(allocator_ids[resource]);
    wm = alloc.getHighWatermark();
    cs = alloc.getCurrentSize();
    as = alloc.getActualSize();
    return;
  }
  //! ------------------------------------------------------------------------
};

}  // namespace ams

#endif
