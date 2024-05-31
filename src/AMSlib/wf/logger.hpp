/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_LOGGER__
#define __AMS_LOGGER__

#include <cstdio>
#include <iostream>
#include <string>

namespace ams
{

std::ostream& ams_out();
std::ostream& ams_error();


namespace util
{

enum LogVerbosityLevel { Error, Warning, Info, Debug, Num_Levels };

class Logger
{
public:
  static const char* MessageLevelName[LogVerbosityLevel::Num_Levels];

  static Logger* getActiveLogger();

  inline bool logLevelEnabled(LogVerbosityLevel level)
  {
    if (level < 0 || level >= Num_Levels || m_is_enabled[level] == false)
      return false;

    if (level == LogVerbosityLevel::Error)
      return true;
    else if (ams_out == nullptr)
      return false;
    return true;
  };

  ~Logger() noexcept = default;
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;


  void setLoggingMsgLevel(LogVerbosityLevel level);
  void initialize_std_io_err(const bool enable_log, std::string& stdio_fn);

  FILE* out() const { return ams_out; }
  FILE* err() const { return ams_err; }

  void flush();
  void close();

private:
  Logger() noexcept;

  bool m_is_enabled[LogVerbosityLevel::Num_Levels];
  FILE *ams_out, *ams_err;
};


void flush_files();
LogVerbosityLevel getVerbosityLevel(const char* level_str);
void close();
inline bool shouldPrint(LogVerbosityLevel lvl)
{
  return Logger::getActiveLogger()->logLevelEnabled(lvl);
}

inline const char* getVerbosityKey(LogVerbosityLevel lvl)
{
  return Logger::MessageLevelName[lvl];
}

inline FILE* out(LogVerbosityLevel lvl)
{
  if (LogVerbosityLevel::Error == lvl) return Logger::getActiveLogger()->err();
  return Logger::getActiveLogger()->out();
}

}  // namespace util
}  // namespace ams

#endif
