#ifndef __UTILITIES__
#define __UTILITIES__

namespace AMS {
namespace utilities {
typedef enum d_location { CPU = 0, DEVICE } dLocation;

void setDefaultDataAllocator(dLocation location);

const char *getDeviceAllocatorName();

const char *getHostAllocatorName();

const char *getDefaultAllocatorName();

} // namespace utilities
} // namespace AMS

#endif