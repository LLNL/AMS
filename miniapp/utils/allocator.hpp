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


    //! setup the allocator
    static void setup(bool use_device);


    //! get/set default allocator
    static ResourceType getDefaultDataAllocator();
    static void setDefaultDataAllocator(ResourceType location);


    static bool isDeviceExecution();


    //! query an allocated array
    template<typename T>
    static const std::string
    getDataAllocationName(T* data) {
      auto& rm = umpire::ResourceManager::getInstance();
      return rm.getAllocator(data).getName();
    }


    template<typename T>
    static bool
    is_on_device(T* data, const std::string &data_name="") {
        // TODO: should find out w/o relying on strings
        auto& rm = umpire::ResourceManager::getInstance();

        try {
          std::cout << "DEBUG: data ("<<data_name<< ":"<<data<<") ::"
                    << rm.getAllocator(data).getName() << ", "
                    << rm.getAllocator(data).getId() << "\n";
          return rm.getAllocator(data).getId() == allocator_ids[ResourceType::DEVICE];
        }
        catch (const std::exception& e) {
          std::cerr << "WARNING: Failed to identify device location for ("<<data_name<<"). Assuming host!\n";
          return false;
        }
    }


    //! allocate and deallocate
    template<typename T>
    static T*
    allocate(size_t nvalues, ResourceType dev = default_resource) {
        auto alloc = umpire::ResourceManager::getInstance().getAllocator(allocator_ids[dev]);
        return static_cast<T*>(alloc.allocate(nvalues * sizeof(T)));
    }

    template<typename T>
    static void
    deallocate(T* data, ResourceType dev = default_resource) {
        auto alloc = umpire::ResourceManager::getInstance().getAllocator(allocator_ids[dev]);
        alloc.deallocate(data);
    }
};

}  // namespace AMS

#endif
