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
    static int
    getDataAllocationId(T* data) {
      static auto& rm = umpire::ResourceManager::getInstance();
      return rm.getAllocator(data).getId();
    }

    template<typename T>
    static const std::string
    getDataAllocationName(T* data) {
      static auto& rm = umpire::ResourceManager::getInstance();
      return rm.getAllocator(data).getName();
    }

    template<typename T>
    static bool
    is_on_device(T* data, const std::string &data_name="") {

        static auto& rm = umpire::ResourceManager::getInstance();
        return rm.getAllocator(data).getId() == allocator_ids[ResourceType::DEVICE];

#if 0
        try {
          /*std::cout << "DEBUG: data ("<<data_name<< ":"<<data<<") ::";
          fflush(stdout);
          std::cout << rm.getAllocator(data).getName() << ", ";
          fflush(stdout);
          std::cout << rm.getAllocator(data).getId() << "\n";*/

          return rm.getAllocator(data).getId() == allocator_ids[ResourceType::DEVICE];
        }
        catch (const std::exception& e) {
          std::cerr << "WARNING: Failed to identify device location for ("<<data_name<<"). Assuming host!\n";
          return false;
        }
#endif
    }

    //! ------------------------------------------------------------------------
    //! allocate and deallocate
    template<typename T>
    static T*
    allocate(size_t nvalues, ResourceType dev = default_resource) {

        /*std::cout << " ---> allocating (" << nvalues << " values) of (size " << sizeof(T) << ") "
                  << "on (" << dev << ": " << (dev == 0 ? "host" : dev == 1? "device" : "unknown") << ") :: ";
        fflush(stdout);*/

        static auto& rm = umpire::ResourceManager::getInstance();
        auto alloc = rm.getAllocator(allocator_ids[dev]);
        auto p = static_cast<T*>(alloc.allocate(nvalues * sizeof(T)));
        //std::cout << p << "\n";
        return p;
    }

    template<typename T>
    static void
    deallocate(T* data, ResourceType dev = default_resource) {
        auto alloc = umpire::ResourceManager::getInstance().getAllocator(data);
        //auto alloc = umpire::ResourceManager::getInstance().getAllocator(allocator_ids[dev]);
        alloc.deallocate(data);
    }
    //! ------------------------------------------------------------------------
};

}  // namespace AMS

#endif
