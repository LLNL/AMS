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

struct AMSAllocator {
  int id;
  umpire::Allocator allocator;

  AMSAllocator(std::string alloc_name)
  {
    auto& rm = umpire::ResourceManager::getInstance();
    allocator = rm.getAllocator(alloc_name);
  }

  void* allocate(size_t num_bytes);
  void deallocate(void* ptr);

  void setAllocator(umpire::Allocator& alloc);

  std::string getName();

  void registerPtr(void* ptr, size_t nBytes);
  static void deregisterPtr(void* ptr)
  {
    auto& rm = umpire::ResourceManager::getInstance();
    rm.deregisterAllocation(ptr);
  }

  void getAllocatorStats(size_t& wm, size_t& cs, size_t& as);
};

class ResourceManager
{
public:
private:
  /** @brief  Used internally to map resource types (Device, host, pinned memory) to
   * umpire allocator ids. */
  static std::vector<AMSAllocator*> RMAllocators;

public:
  ResourceManager() = delete;
  ResourceManager(const ResourceManager&) = delete;
  ResourceManager(ResourceManager&&) = delete;
  ResourceManager& operator=(const ResourceManager&) = delete;
  ResourceManager& operator=(ResourceManager&&) = delete;

  /** @brief return the name of an allocator */
  static std::string getAllocatorName(AMSResourceType resource)
  {
    return RMAllocators[resource]->getName();
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
    return static_cast<TypeInValue*>(
        RMAllocators[dev]->allocate(nvalues * sizeof(TypeInValue)));
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
    RMAllocators[dev]->deallocate(data);
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
    RMAllocators[dev]->registerPtr(ptr, nBytes);
  }

  /** @brief removes a registered external pointer from the umpire allocation records.
   *  @param[in] ptr pointer to memory to de-register.
   *  @return void.
   */
  static void deregisterExternal(void* ptr)
  {
    AMSAllocator::deregisterPtr(ptr);
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
  static void deallocate(std::vector<T*>& dPtr, AMSResourceType resource)
  {
    for (auto* I : dPtr)
      RMAllocators[resource]->deallocate(I);
  }

  static void init()
  {
    DBG(ResourceManager, "Default initialization of allocators");
    if (!RMAllocators[AMSResourceType::HOST])
      setAllocator("HOST", AMSResourceType::HOST);
#ifdef __ENABLE_CUDA__
    if (!RMAllocators[AMSResourceType::DEVICE])
      setAllocator("DEVICE", AMSResourceType::DEVICE);

    if (!RMAllocators[AMSResourceType::PINNED])
      setAllocator("PINNED", AMSResourceType::PINNED);
#endif
  }

  static void setAllocator(std::string alloc_name, AMSResourceType resource)
  {
    if (RMAllocators[resource]) {
      delete RMAllocators[resource];
    }

    RMAllocators[resource] = new AMSAllocator(alloc_name);
    DBG(ResourceManager,
        "Set Allocator [%d] to pool with name : %s",
        resource,
        RMAllocators[resource]->getName().c_str());
  }

  static bool isActive(AMSResourceType resource){
    return RMAllocators[resource] != nullptr;
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
    RMAllocators[resource]->getAllocatorStats(wm, cs, as);
    return;
  }
  //! ------------------------------------------------------------------------
};

}  // namespace ams

#endif
