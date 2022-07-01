#ifndef __AMS_ALLOCATOR__
#define __AMS_ALLOCATOR__

#include <cstddef>
#include <umpire/Umpire.hpp>

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
}  // namespace AMS

#endif
