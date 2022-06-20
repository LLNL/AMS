#include "utilities.hpp"

namespace AMS {
namespace utilities {
dLocation defaultDloc = dLocation::CPU;

void setDefaultDataAllocator(dLocation location) { defaultDloc = location; }

const char *getDeviceAllocatorName() { return "mmp-device-quickpool"; }

const char *getHostAllocatorName() { return "mmp-host-quickpool"; }

const char *getDefaultAllocatorName() {
    switch (defaultDloc) {
    case dLocation::CPU:
        return getHostAllocatorName();
    case dLocation::DEVICE:
        return getDeviceAllocatorName();
    }
}

} // namespace utilities
} // namespace AMS