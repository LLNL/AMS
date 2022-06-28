#ifndef __UTILITIES__
#define __UTILITIES__
#include <cstddef>

namespace AMS {
namespace utilities {
typedef enum d_location { CPU = 0, DEVICE } AMSDevice;

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

}  // namespace utilities
}  // namespace AMS

#endif
