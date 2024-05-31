/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

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
