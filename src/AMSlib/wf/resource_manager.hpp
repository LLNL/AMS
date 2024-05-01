/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_ALLOCATOR__
#define __AMS_ALLOCATOR__

#include <cstddef>
#include <stdexcept>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>

#include "AMS.h"
#include "debug.h"
#include "wf/debug.h"


namespace ams
{

struct DataPtr {
  size_t size;
  AMSResourceType resource;
  DataPtr(size_t s, AMSResourceType res) : size(s), resource(res) {}
};
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
    DBG(AMSAllocator,
        "in AMSAllocator(%d, %s, %p)",
        id,
        alloc_name.c_str(),
        this)
  }

  ~AMSAllocator() { DBG(AMSAllocator, "in ~AMSAllocator(%d, %p)", id, this) }

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
private:
  /** @brief  Used internally to map resource types (Device, host, pinned memory) to
   * umpire allocator ids. */
  std::vector<AMSAllocator*> RMAllocators;
  ResourceManager() : RMAllocators({nullptr, nullptr, nullptr}){};
  std::unordered_map<void*, DataPtr> activePointers;

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
  std::string getAllocatorName(AMSResourceType resource) { return "System"; }

  /** @brief Allocates nvalues on the specified device.
   *  @tparam TypeInValue The type of pointer to allocate.
   *  @param[in] nvalues Number of elements to allocate.
   *  @param[in] dev Resource to allocate memory from.
   *  @return Pointer to allocated elements.
   */
  template <typename TypeInValue>
  PERFFASPECT()
  TypeInValue* allocate(size_t nvalues, AMSResourceType dev)
  {
    void* ptr = malloc(nvalues * sizeof(TypeInValue));
    activePointers.emplace(ptr, DataPtr(nvalues * sizeof(TypeInValue), dev));
    return static_cast<TypeInValue*>(ptr);
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
    activePointers.erase(static_cast<void*>(data));
    free(data);
    //RMAllocators[dev]->deallocate(data);
  }

  /** @brief registers an external pointer in the umpire allocation records.
   *  @param[in] ptr pointer to memory to register.
   *  @param[in] nBytes number of bytes to register.
   *  @param[in] dev resource to register the memory to.
   *  @return void.
   */
  PERFFASPECT()
  void registerExternal(void* ptr, size_t nBytes, AMSResourceType dev)
  {
    activePointers.emplace(ptr, DataPtr(nBytes, dev));
    //RMAllocators[dev]->registerPtr(ptr, nBytes);
  }

  /** @brief removes a registered external pointer from the umpire allocation records.
   *  @param[in] ptr pointer to memory to de-register.
   *  @return void.
   */
  void deregisterExternal(void* ptr)
  {
    activePointers.erase((ptr));
    //AMSAllocator::deregisterPtr(ptr);
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
  void copy(TypeInValue* src, TypeInValue* dest, size_t size = 0)
  {
    auto src_it = activePointers.find(src);
    if (src_it == activePointers.end()) {
      THROW(std::runtime_error,
            "Pointer is not register in activePoints (src)");
    }

    size_t src_size = src_it->second.size;

    auto dest_it = activePointers.find(dest);
    if (dest_it == activePointers.end()) {
      THROW(std::runtime_error,
            "Pointer not registered in activePointers (dest)");
    }

    size_t dest_size = dest_it->second.size;

    if (size == 0 && dest_size != src_size) {
      THROW(std::runtime_error,
            "Pointers have different sizes, I cannot copy them");
    }

    if (size == 0) {
      std::memcpy(dest, src, src_size);
    } else {
      std::memcpy(dest, src, size);
    }
  }

  /** @brief Utility function that deallocates all C-Vectors inside the vector.
   *  @tparam TypeInValue type of pointers
   *  @param[in] dPtr vector containing pointers to C-vectors to be allocated.
   *  @return void.
   */
  template <typename T>
  void deallocate(std::vector<T*>& dPtr, AMSResourceType resource)
  {
    for (auto* I : dPtr) {
      auto dest_it = activePointers.find(I);
      if (dest_it == activePointers.end()) {
        THROW(std::runtime_error, "Cannot find allocation in activePoints");
      }
      activePointers.erase(I);
      free(I);
    }
    //RMAllocators[resource]->deallocate(I);
  }

  void init() {}

  void setAllocator(std::string alloc_name, AMSResourceType resource) {}

  bool isActive(AMSResourceType resource) { return true; }

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
    as = 0;
    cs = 0;
    wm = 0;
    for (auto KV : activePointers) {
      if (KV.second.resource == resource) as += KV.second.size;
    }
    return;
  }
  //! ------------------------------------------------------------------------
};

}  // namespace ams

#endif
