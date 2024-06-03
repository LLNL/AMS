/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include <algorithm>  // for std::equal
#include <cctype>     // for std::toupper
#include <cstdlib>    // for getenv()
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "debug.h"
#include "logger.hpp"
#include "wf/debug.h"
#include "wf/logger.hpp"


namespace ams
{
namespace util
{

// By default AMS prints only errors
static LogVerbosityLevel defaultLevel = LogVerbosityLevel::Error;

const char* Logger::MessageLevelName[LogVerbosityLevel::Num_Levels] = {"ERROR",
                                                                       "WARNIN"
                                                                       "G",
                                                                       "INFO",
                                                                       "DEBUG"};

static int case_insensitive_match(const std::string s1, const std::string s2)
{
  return (s1.size() == s2.size()) &&
         std::equal(s1.begin(), s1.end(), s2.begin(), [](char c1, char c2) {
           return (std::toupper(c1) == std::toupper(c2));
         });
}

Logger::Logger() noexcept
    :  // by default, all message streams are disabled
      m_is_enabled{false, false, false, false}
{
  LogVerbosityLevel level{defaultLevel};
  setLoggingMsgLevel(level);
}

LogVerbosityLevel getVerbosityLevel(const char* level_str)
{
  if (level_str == nullptr) return defaultLevel;

  for (int i = 0; i < LogVerbosityLevel::Num_Levels; ++i) {
    if (case_insensitive_match(level_str, Logger::MessageLevelName[i])) {
      return static_cast<LogVerbosityLevel>(i);
    }
  }

  return defaultLevel;
}

void Logger::setLoggingMsgLevel(LogVerbosityLevel level)
{
  for (int i = 0; i < LogVerbosityLevel::Num_Levels; ++i)
    m_is_enabled[i] = (i <= level);
}

Logger* Logger::getActiveLogger()
{
  static Logger logger;
  return &logger;
}

static inline std::string concat_file_name(const std::string& prefix,
                                           const std::string& suffix)
{
  return prefix + "." + suffix;
}

void Logger::initialize_std_io_err(const bool enable_log, std::string& stdio_fn)
{
  ams_out = nullptr;
  ams_err = stderr;

  if (enable_log) {
    ams_out = stdout;
    // The case we want to just redirect to stdout
    if (!stdio_fn.empty()) {
      const std::string log_filename{concat_file_name(stdio_fn, "log")};
      ams_out = fopen(log_filename.c_str(), "a");
      CFATAL(Logger,
             ams_out == nullptr,
             "Could not open file for stdout redirection");
    }
  }
}

void Logger::flush()
{
  if (ams_out != nullptr && ams_out != stdout) fflush(ams_out);
  fflush(ams_err);
}


void Logger::close()
{

  if (ams_out != nullptr && ams_out != stdout) {
    fclose(ams_out);
    ams_out = nullptr;
  }
}

void close()
{
  auto logger = Logger::getActiveLogger();
  logger->flush();
  logger->close();
}

void flush_files()
{
  auto logger = Logger::getActiveLogger();
  logger->flush();
}

}  // namespace util
}  // namespace ams
