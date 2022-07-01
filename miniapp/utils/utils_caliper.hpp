#ifndef __AMS_UTILS_CALIPER_HPP__
#define __AMS_UTILS_CALIPER_HPP__


// This is usefull to completely remove
// caliper at compile time.
#ifdef __ENABLE_CALIPER__
#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#define CALIPER(stmt) stmt
#else
#define CALIPER(stmt)
#endif

#endif
