#ifndef __UTILITIES__
#define __UTILITIES__

namespace AMS {
namespace utilities {
typedef enum d_location { CPU = 0, DEVICE } AMSDevice;

void setDefaultDataAllocator(AMSDevice location);

AMSDevice getDefaultDataAllocator();

const char *getDeviceAllocatorName();

const char *getHostAllocatorName();

const char *getDefaultAllocatorName();

void *allocate(size_t bytes);
void *allocate(size_t bytes, AMSDevice dev);


} // namespace utilities
} // namespace AMS

#endif
