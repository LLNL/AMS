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
    //! names for these allocators
    static const std::string getDeviceAllocatorName();
    static const std::string getHostAllocatorName();

    //! setup allocators in the resource manager
    static void setup(const std::string &device_name);

    //! list allocators
    static void list_allocators();

    //! get/set default allocator
    static ResourceType getDefaultDataAllocator();
    static void setDefaultDataAllocator(ResourceType resource);

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
                alloc_id != -1 && alloc_id == allocator_ids[ResourceType::DEVICE];
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
        if (hasAllocator(data) ) {
            rm.getAllocator(data).deallocate(data);
        }
    }
    //! ------------------------------------------------------------------------
};

}  // namespace AMS

#endif
