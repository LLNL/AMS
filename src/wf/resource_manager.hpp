// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

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

  /** @brief  Used internally to allocate from the user define default resource (Device, Host Memory) */
  static AMSResourceType default_resource;

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

  /** @brief Print out all available allocators */
  static void list_allocators();

  /** @brief Get the default memory allocator */
  static AMSResourceType getDefaultDataAllocator();

  /** @brief Set the default memory allocator */
  static void setDefaultDataAllocator(AMSResourceType resource);

  /** @brief Check if default allocator is set to Device
   *  @pre The library currently assumes the default memory allocator
   *  also describes the executing device.
   */
  static bool isDeviceExecution();

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

  /** @brief checks whether the data are resident on the device.
   *  @tparam TypeInValue The type of pointer being tested.
   *  @param[in] data pointer to memory.
   *  @return True when data are on Device.
   */
  template <typename TypeInValue>
  static bool is_on_device(const TypeInValue* data)
  {
    auto alloc_id = getDataAllocationId(data);
    return ResourceManager::isDeviceExecution() && alloc_id != -1 &&
           alloc_id == allocator_ids[AMSResourceType::DEVICE];
  }

  /** @brief Allocates nvalues on the specified device.
   *  @tparam TypeInValue The type of pointer to allocate.
   *  @param[in] nvalues Number of elements to allocate.
   *  @param[in] dev Resource to allocate memory from.
   *  @return Pointer to allocated elements.
   */
  template <typename TypeInValue>
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  static TypeInValue* allocate(size_t nvalues, AMSResourceType dev = default_resource)
  {
    static auto& rm = umpire::ResourceManager::getInstance();
    auto alloc = rm.getAllocator(allocator_ids[dev]);
    TypeInValue *ret = static_cast<TypeInValue*>(alloc.allocate(nvalues * sizeof(TypeInValue)));
    CFATAL(ResourceManager, ret == nullptr,
        "Failed to allocated %ld values on device %d", nvalues, dev);
    return ret;
  }

  /** @brief deallocates pointer from the specified device.
   *  @tparam TypeInValue The type of pointer to de-allocate.
   *  @param[in] data pointer to deallocate.
   *  @param[in] dev device to de-allocate from .
   *  @return void.
   */
  template <typename TypeInValue>
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  static void deallocate(TypeInValue* data, AMSResourceType dev = default_resource)
  {
    static auto& rm = umpire::ResourceManager::getInstance();
    if (hasAllocator(data)) {
      rm.getAllocator(data).deallocate(data);
    }
  }

  /** @brief registers an external pointer in the umpire allocation records.
   *  @param[in] ptr pointer to memory to register.
   *  @param[in] nBytes number of bytes to register.
   *  @param[in] dev resource to register the memory to.
   *  @return void.
   */
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  static void registerExternal(void* ptr,
                               size_t nBytes,
                               AMSResourceType dev = default_resource)
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
   *  @param[in] size number of elements to copy. (When 0 copies entire allocated area)
   *  @return void.
   */
  template <typename TypeInValue>
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
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
  //! ------------------------------------------------------------------------
};

}  // namespace ams

#endif
