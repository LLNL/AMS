#ifndef __AMS_ALLOCATOR__
#define __AMS_ALLOCATOR__

#include <cstddef>

#include "AMS.h"
#include <umpire/Umpire.hpp>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>


namespace ams {

class ResourceManager {
public:
private:
    static AMSResourceType default_resource;
    static int allocator_ids[AMSResourceType::RSEND];
    static std::string allocator_names[AMSResourceType::RSEND];

public:
    //! names for these allocators
    static const char* getDeviceAllocatorName();
    static const char* getHostAllocatorName();
    static const char* getAllocatorName(AMSResourceType Resource);

    //! setup allocators in the resource manager
    static void setup(const AMSResourceType resource);

    //! list allocators
    static void list_allocators();

    //! get/set default allocator
    static AMSResourceType getDefaultDataAllocator();
    static void setDefaultDataAllocator(AMSResourceType resource);

    //! check if we are using device
    static bool isDeviceExecution();

    //! ------------------------------------------------------------------------
    //! query an allocated array
    template<typename T>
    static bool
    hasAllocator(const T* data) {
      static auto& rm = umpire::ResourceManager::getInstance();
      void *vdata = static_cast<void*>(const_cast<T*>(data));
      return rm.hasAllocator(vdata);
    }

    template<typename T>
    static int
    getDataAllocationId(const T* data) {
      static auto& rm = umpire::ResourceManager::getInstance();
      void *vdata = static_cast<void*>(const_cast<T*>(data));
      return rm.hasAllocator(vdata) ? rm.getAllocator(vdata).getId() : -1;
    }

    template<typename T>
    static const std::string
    getDataAllocationName(const T* data) {
      static auto& rm = umpire::ResourceManager::getInstance();
      void *vdata = static_cast<void*>(const_cast<T*>(data));
      return rm.hasAllocator(vdata) ? rm.getAllocator(vdata).getName() : "unknown";
    }

    template<typename T>
    static bool
    is_on_device(const T* data) {
        auto alloc_id = getDataAllocationId(data);
        return ResourceManager::isDeviceExecution() &&
                alloc_id != -1 && alloc_id == allocator_ids[AMSResourceType::DEVICE];
    }

    //! ------------------------------------------------------------------------
    //! allocate and deallocate
    template<typename T>
    static T*
    allocate(size_t nvalues, AMSResourceType dev = default_resource) {
        static auto& rm = umpire::ResourceManager::getInstance();
        auto alloc = rm.getAllocator(allocator_ids[dev]);
        return static_cast<T*>(alloc.allocate(nvalues * sizeof(T)));
    }

    template<typename T>
    static void
    deallocate(T* data, AMSResourceType dev = default_resource) {
        static auto& rm = umpire::ResourceManager::getInstance();
        if (hasAllocator(data) ) {
            rm.getAllocator(data).deallocate(data);
        }
    }
    //! ------------------------------------------------------------------------
};

}  // namespace AMS

#endif
