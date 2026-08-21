#ifndef AMS_TEST_DEVICE_HPP
#define AMS_TEST_DEVICE_HPP

#if defined(__AMS_ENABLE_CUDA__)
#include <cuda_runtime.h>
#elif defined(__AMS_ENABLE_HIP__)
#include <hip/hip_runtime.h>
#endif

namespace ams::test
{
inline bool hasRuntimeDevice()
{
#if defined(__AMS_ENABLE_CUDA__)
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#elif defined(__AMS_ENABLE_HIP__)
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#else
  return false;
#endif
}
}  // namespace ams::test

#endif  // AMS_TEST_DEVICE_HPP
