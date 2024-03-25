/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#ifndef __AMS_DEBUG__
#define __AMS_DEBUG__

#include <atomic>
#include <mutex>


#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"

enum AMSVerbosity {
  AMSFATAL = 0x001,
  AMSWARNING = 0x002,
  AMSINFO = 0x004,
  AMSDEBUG = 0x008,
};


void memUsage(double& vm_usage, double& resident_set);

inline std::atomic<uint32_t>& getInfoLevelInternal()
{
  static std::atomic<uint32_t> InfoLevel;
  static std::once_flag Flag{};
  std::call_once(Flag, []() {
    InfoLevel.store(AMSVerbosity::AMSFATAL);
    if (char* EnvStr = getenv("LIBAMS_VERBOSITY_LEVEL"))
      InfoLevel.store(std::stoi(EnvStr));  // We always enable FATAL
  });
  return InfoLevel;
}

inline uint32_t getVerbosityLevel()
{
  return getInfoLevelInternal().load() | AMSFATAL;
}

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)

#define AMSPRINTID(id)                       \
  do {                                       \
    fprintf(stderr, "[ " GETNAME(id) " ] "); \
  } while (0);

#define AMSPRINTMESSAGE(...)      \
  do {                            \
    fprintf(stderr, __VA_ARGS__); \
  } while (0);

#define AMSPRINT(id, condition, vl, color, ...)  \
  if (condition && (getVerbosityLevel() & vl)) { \
    AMSPRINTMESSAGE(color)                       \
    AMSPRINTID(id)                               \
    AMSPRINTMESSAGE(__VA_ARGS__)                 \
    AMSPRINTMESSAGE(RESET "\n")                  \
  }                                              \
  while (0)                                      \
    ;

#define CFATAL(id, condition, ...)                                    \
  if (condition) {                                                    \
    AMSPRINT(id, condition, AMSVerbosity::AMSFATAL, RED, __VA_ARGS__) \
    abort();                                                          \
  }

#define FATAL(id, ...) CFATAL(id, true, __VA_ARGS__)

#define THROW(exception, msg)                                              \
  throw exception(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                  " " + msg)


#ifdef AMS_DEBUG

#define CWARNING(id, condition, ...) \
  AMSPRINT(id, condition, AMSVerbosity::AMSWARNING, YEL, __VA_ARGS__)

#define WARNING(id, ...) CWARNING(id, true, __VA_ARGS__)

#define CINFO(id, condition, ...) \
  AMSPRINT(id, condition, AMSVerbosity::AMSINFO, BLU, __VA_ARGS__)

#define INFO(id, ...) CINFO(id, true, __VA_ARGS__)

#define CDEBUG(id, condition, ...) \
  AMSPRINT(id, condition, AMSVerbosity::AMSDEBUG, MAG, __VA_ARGS__)

#define DBG(id, ...) CDEBUG(id, true, __VA_ARGS__)

#define REPORT_MEM_USAGE(id, phase)                                    \
  do {                                                                 \
    double vm, rs;                                                     \
    size_t watermark, current_size, actual_size;                       \
    auto& rm = ams::ResourceManager::getInstance();                    \
    memUsage(vm, rs);                                                  \
    DBG(id, "Memory usage at %s is VM:%g RS:%g", phase, vm, rs);       \
                                                                       \
    for (int i = 0; i < AMSResourceType::RSEND; i++) {                 \
      if (rm.isActive((AMSResourceType)i)) {                           \
        rm.getAllocatorStats((AMSResourceType)i,                       \
                                                watermark,             \
                                                current_size,          \
                                                actual_size);          \
        DBG(id,                                                        \
            "Allocator: %s HWM:%lu CS:%lu AS:%lu) ",                   \
            rm.getAllocatorName((AMSResourceType)i)                    \
                .c_str(),                                              \
            watermark,                                                 \
            current_size,                                              \
            actual_size);                                              \
      }                                                                \
    }                                                                  \
  } while (0);

#ifdef __ENABLE_CUDA__
// NOTE: Regardless of condition we synchronize. We only emit a message based on condition.
#define _CAMSDebugDeviceSync(id, condition, fn, ln, ...)        \
  do{                                      \
    AMSDeviceSync(fn, ln);                  \
    CDEBUG(id, condition, __VA_ARGS__)     \
  }while(0);

#define CAMSDebugDeviceSync(id, condition, ...)  _CAMSDebugDeviceSync(id, condition, __FILE__, __LINE__, __VA_ARGS__) 
#define AMSDebugDeviceSync(id, ...) _CAMSDebugDeviceSync(id, true, __FILE__, __LINE__, __VA_ARGS__)
#else
#define CAMSDebugDeviceSync(id, condition, ...)
#define AMSDebugDeviceSync(id, ...)
#endif


#else  // LIBAMS_DEBUG is disabled
#define CWARNING(id, condition, ...)

#define WARNING(id, ...)

#define CINFO(id, condition, ...)

#define INFO(id, ...)

#define CDEBUG(id, condition, ...)

#define DBG(id, ...)

#define REPORT_MEM_USAGE(id, phase)                                    \

#define CAMSDebugDeviceSync(id, condition, ...)
#define AMSDebugDeviceSync(id, ...)

#endif  // AMS_DEBUG

#endif // __AMS_DEBUG__
