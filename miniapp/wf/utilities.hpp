#ifndef __UTILITIES__
#define __UTILITIES__

namespace AMS {
namespace utilities {
const char *getDeviceAllocatorName() { return "mmp-device-quickpool"; }

const char *getHostAllocatorName() { return "mmp-host-quickpool"; }
} // namespace utilities
} // namespace AMS

#endif