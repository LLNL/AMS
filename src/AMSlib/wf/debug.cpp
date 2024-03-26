/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "wf/debug.h"

#if defined(__ENABLE_CUDA__) && defined(__ENABLE_TORCH__) 
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <unistd.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <string>

using namespace std;

/** \brief Get the memory usage as reported in /proc/self/stat in terms
   * of used VM and resident memory.
   * @param[out] vm_usage the amount of memory used in virtual memory
   * @param[out] resident_set the resident set of this process
   */
void memUsage(double& vm_usage, double& resident_set)
{

  vm_usage = 0.0;
  resident_set = 0.0;
  ifstream stat_stream("/proc/self/stat", ios_base::in);  //get info from proc
  string pid, comm, state, ppid, pgrp, session, tty_nr;
  string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  string utime, stime, cutime, cstime, priority, nice;
  string O, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
      tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
      stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >>
      starttime >> vsize >> rss;
  stat_stream.close();
  long page_size = sysconf(_SC_PAGE_SIZE);  // for x86-64 is configured
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size;
}

void dumpTorchDeviceStats()
{
#if defined(__ENABLE_CUDA__) && defined(__ENABLE_TORCH__) 
  c10::cuda::CUDACachingAllocator::emptyCache();
  int curr_device = c10::cuda::current_device();
  c10::cuda::CUDACachingAllocator::DeviceStats stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(curr_device);

  DBG(TorchDeviceStats,
      "Current device according to torch has id : %d",
      curr_device);

  for (auto S : stats.allocated_bytes) {
    DBG(TorchDeviceStats,
        "Allocated Current: %g (MBytes) Peak: %g (MBytes) Allocated: %G "
        "(MBytes) Freed: "
        "%g (MBytes)",
        (double)(S.current) / (1024.0 * 1024.0),
        (double)(S.peak) / (1024.0 * 1024.0),
        (double)(S.allocated) / (1024.0 * 1024.0),
        (double)(S.freed) / (1024.0 * 1024.0));
  }

  for (auto S : stats.reserved_bytes) {
    DBG(TorchDeviceStats,
        "Reserved Current: %g (MBytes) Peak: %g (MBytes) Allocated: %G "
        "(MBytes) Freed: "
        "%g (MBytes)",
        (double)(S.current) / (1024.0 * 1024.0),
        (double)(S.peak) / (1024.0 * 1024.0),
        (double)(S.allocated) / (1024.0 * 1024.0),
        (double)(S.freed) / (1024.0 * 1024.0));
  }

  for (auto S : stats.active_bytes) {
    DBG(TorchDeviceStats,
        "Active Current: %g (MBytes) Peak: %g (MBytes) Allocated: %G "
        "(MBytes) Freed: "
        "%g (MBytes)",
        (double)(S.current) / (1024.0 * 1024.0),
        (double)(S.peak) / (1024.0 * 1024.0),
        (double)(S.allocated) / (1024.0 * 1024.0),
        (double)(S.freed) / (1024.0 * 1024.0));
  }

  for (auto S : stats.inactive_split_bytes) {
    DBG(TorchDeviceStats,
        "Inactive Split Bytes Current: %g (MBytes) Peak: %g (MBytes) "
        "Allocated: %G "
        "(MBytes) Freed: "
        "%g (MBytes)",
        (double)(S.current) / (1024.0 * 1024.0),
        (double)(S.peak) / (1024.0 * 1024.0),
        (double)(S.allocated) / (1024.0 * 1024.0),
        (double)(S.freed) / (1024.0 * 1024.0));
  }
#endif
}
