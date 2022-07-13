#ifndef __AMS_ALLOCATOR__
#define __AMS_ALLOCATOR__

#include <cstddef>

#include <umpire/Umpire.hpp>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>


namespace ams {

class ResourceManager {
public:
    typedef enum location { UNKNOWN = -1, HOST = 0, DEVICE = 1, RSEND } ResourceType;
private:
    static ResourceType default_resource;
    static int allocator_ids[ResourceType::RSEND];

public:
    //! names for these allocations
    static const std::string getDeviceAllocatorName();
    static const std::string getHostAllocatorName();


    //! setup allocators in the resource manager
    static void setup(const std::string &device_name);


    //! list the allocators
    static void list_allocators();

    /*
    //! get/set default allocator
    static ResourceType getDefaultDataAllocator();
    static void setDefaultDataAllocator(ResourceType location);*/

    static bool isDeviceExecution();


    //! ------------------------------------------------------------------------
    //! query an allocated array
    template<typename T>
    static bool
    hasAllocator(T* data) {
      static auto& rm = umpire::ResourceManager::getInstance();
      return rm.hasAllocator(data);
    }

    template<typename T>
    static int
    getDataAllocationId(T* data) {
      static auto& rm = umpire::ResourceManager::getInstance();
      return rm.hasAllocator(data) ? rm.getAllocator(data).getId() : -1;
    }

    template<typename T>
    static const std::string
    getDataAllocationName(T* data) {
      static auto& rm = umpire::ResourceManager::getInstance();
      return rm.hasAllocator(data) ? rm.getAllocator(data).getName() : "unknown";
    }

    template<typename T>
    static bool
    is_on_device(T* data) {
        return getDataAllocationId(data) == allocator_ids[ResourceType::DEVICE];
    }

    //! ------------------------------------------------------------------------
    //! allocate and deallocate
    template<typename T>
    static T*
    allocate(size_t nvalues, ResourceType dev = default_resource) {
        static auto& rm = umpire::ResourceManager::getInstance();
        auto alloc = rm.getAllocator(allocator_ids[dev]);
        return static_cast<T*>(alloc.allocate(nvalues * sizeof(T)));
    }

    template<typename T>
    static void
    deallocate(T* data, ResourceType dev = default_resource) {
        static auto& rm = umpire::ResourceManager::getInstance();
        if (rm.hasAllocator(data) ) {
            rm.getAllocator(data).deallocate(data);
        }
    }
    //! ------------------------------------------------------------------------
};

}  // namespace AMS

#endif
