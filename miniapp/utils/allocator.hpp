#ifndef __AMS_ALLOCATOR__
#define __AMS_ALLOCATOR__

#include <cstddef>
#include <umpire/Umpire.hpp>

#define USE_NEW_ALLOCATOR

namespace AMS {
namespace utilities {
typedef enum d_location { HOST = 0, DEVICE } AMSDevice;

void setDefaultDataAllocator(AMSDevice location);

AMSDevice getDefaultDataAllocator();

const char* getDeviceAllocatorName();

const char* getHostAllocatorName();

const char* getDefaultAllocatorName();

void* allocate(size_t bytes);
void* allocate(size_t bytes, AMSDevice dev);

void deallocate(void* ptr, AMSDevice dev);
void deallocate(void* ptr);

bool isDeviceExecution();


template<typename T>
bool is_data_on_device(T* data) {

  // todo: we need to do this better! should not rely on strings,
  // but see how Dinos is using the enums!
  auto& rm = umpire::ResourceManager::getInstance();
  auto found_allocator = rm.getAllocator(data);
  auto nm = found_allocator.getName();

  bool is_device = int(nm.find("device")) > 0 || int(nm.find("DEVICE")) > 0;

  //std::cout << " is_data_on_device("<<data<<") = "<< is_device << " ::: " << nm
  //          << " :: " << nm.find("host") << ", " << nm.find("HOST")
  //          << " :: " << nm.find("device") << ", " << nm.find("DEVICE") << "\n";
  return is_device;
}



}  // namespace utilities



#ifdef USE_NEW_ALLOCATOR
class ResourceManager {

    typedef enum location { UNKNOWN = 0, HOST = 1, DEVICE = 2 } ResourceType;
    static ResourceType default_resource;

public:
    //! names for these allocations
    static const std::string getDeviceAllocatorName();
    static const std::string getHostAllocatorName();


    //! setup the allocator
    static void setup(bool use_device);


    //! get/set default allocator
    static ResourceType getDefaultDataAllocator();
    static void setDefaultDataAllocator(ResourceType location);


    //! query an allocated array
    template<typename T>
    static const std::string
    getDataAllocationName(T* data) {
      auto& rm = umpire::ResourceManager::getInstance();
      return rm.getAllocator(data).getName();
    }

    template<typename T>
    static bool
    is_on_device(T* data) {
        // TODO: should find out w/o relying on strings
        auto nm = ResourceManager::getDataAllocationName<T>(data);
        return int(nm.find("device")) > 0 || int(nm.find("DEVICE")) > 0;
    }


    //! allocate and deallocate
    template<typename T>
    static T*
    allocate(size_t nvalues, ResourceType dev = default_resource) {

        const std::string &alloc_name = (dev == HOST) ?
                    getHostAllocatorName() : getDeviceAllocatorName();

        auto alloc = umpire::ResourceManager::getInstance().getAllocator(alloc_name);
        return static_cast<T*>(alloc.allocate(nvalues * sizeof(T)));
    }

    template<typename T>
    static void
    deallocate(T* data, ResourceType dev = default_resource) {

        const std::string &alloc_name = (dev == HOST) ?
                    getHostAllocatorName() : getDeviceAllocatorName();

        auto alloc = umpire::ResourceManager::getInstance().getAllocator(alloc_name);
        alloc.deallocate(data);
    }
};
#endif

}  // namespace AMS

#endif
