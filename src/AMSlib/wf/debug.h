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
#include <wf/logger.hpp>

void memUsage(double& vm_usage, double& resident_set);

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)

#define AMSPRINT(id, condition, vl, ...)           \
  do {                                             \
    if (condition && ams::util::shouldPrint(vl)) { \
      fprintf(ams::util::out(vl),                  \
              "[AMS:%s:%s] ",                      \
              ams::util::getVerbosityKey(vl),      \
              GETNAME(id));                        \
      fprintf(ams::util::out(vl), __VA_ARGS__);    \
      fprintf(ams::util::out(vl), "\n");           \
    }                                              \
  } while (0);

#define CFATAL(id, condition, ...)                  \
  do {                                              \
    if (condition) {                                \
      AMSPRINT(id,                                  \
               condition,                           \
               ams::util::LogVerbosityLevel::Error, \
               __VA_ARGS__)                         \
      ams::util::flush_files();                     \
      ams::util::close();                           \
      abort();                                      \
    }                                               \
  } while (0);

#define FATAL(id, ...) CFATAL(id, true, __VA_ARGS__)

#define THROW(exception, msg) \
  FATAL(Throw, "%s %s %s", __FILE__, std::to_string(__LINE__).c_str(), msg)

#ifdef LIBAMS_VERBOSE

#define CWARNING(id, condition, ...) \
  AMSPRINT(id, condition, ams::util::LogVerbosityLevel::Warning, __VA_ARGS__)

#define WARNING(id, ...) CWARNING(id, true, __VA_ARGS__)

#define CINFO(id, condition, ...) \
  AMSPRINT(id, condition, ams::util::LogVerbosityLevel::Info, __VA_ARGS__)

#define INFO(id, ...) CINFO(id, true, __VA_ARGS__)

#define CDEBUG(id, condition, ...) \
  AMSPRINT(id, condition, ams::util::LogVerbosityLevel::Debug, __VA_ARGS__)

#define DBG(id, ...) CDEBUG(id, true, __VA_ARGS__)

// clang-format off
#define REPORT_MEM_USAGE(id, phase)                                    \
  do {                                                                 \
    double vm, rs;                                                     \
    size_t watermark, current_size, actual_size;                       \
    auto& rm = ams::ResourceManager::getInstance();                    \
    memUsage(vm, rs);                                                  \
    DBG(MEM : id, "Memory usage at %s is VM:%g RS:%g", phase, vm, rs); \
                                                                       \
    for (int i = 0; i < AMSResourceType::AMS_RSEND; i++) {             \
      if (rm.isActive((AMSResourceType)i)) {                           \
        rm.getAllocatorStats((AMSResourceType)i,                       \
                             watermark,                                \
                             current_size,                             \
                             actual_size);                             \
        DBG(MEM                                                        \
            : id,                                                      \
              "Allocator: %s HWM:%lu CS:%lu AS:%lu) ",                 \
              rm.getAllocatorName((AMSResourceType)i).c_str(),         \
              watermark,                                               \
              current_size,                                            \
              actual_size);                                            \
      }                                                                \
    }                                                                  \
  } while (0);

// clang-format on

#else  // LIBAMS_VERBOSE is disabled
#define CWARNING(id, condition, ...)

#define WARNING(id, ...)

#define CINFO(id, condition, ...)

#define INFO(id, ...)

#define CDEBUG(id, condition, ...)

#define DBG(id, ...)


#endif  // LIBAMS_VERBOSE

#endif
