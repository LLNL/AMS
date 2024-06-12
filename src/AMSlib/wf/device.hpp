/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_DEVICE_HPP__
#define __AMS_DEVICE_HPP__

#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "AMS.h"
#include "wf/debug.h"

#define UNDEFINED_FUNC -1

#ifdef __ENABLE_CUDA__
namespace ams
{
void DtoDMemcpy(void *dest, void *src, size_t nBytes);

void HtoHMemcpy(void *dest, void *src, size_t nBytes);

void HtoDMemcpy(void *dest, void *src, size_t nBytes);

void DtoHMemcpy(void *dest, void *src, size_t nBytes);

void *DeviceAllocate(size_t nBytes);

void DeviceFree(void *ptr);

void *DevicePinnedAlloc(size_t nBytes);

void DeviceFreePinned(void *ptr);

void deviceCheckErrors(const char *file, int line);

void device_random_uq(int seed,
                      bool *uq_flags,
                      int ndata,
                      double acceptable_error);

namespace Device
{

template <typename scalar_t>
void computeDeltaUQMeanPredicatesDevice(
    const scalar_t *__restrict__ outputs_stdev,
    bool *__restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold);


template <typename scalar_t>
void computeDeltaUQMaxPredicatesDevice(
    const scalar_t *__restrict__ outputs_stdev,
    bool *__restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold);

void device_compute_predicate(float *data,
                              bool *predicate,
                              size_t nData,
                              const size_t kneigh,
                              float threshold);

template <typename TypeValue>
PERFFASPECT()
void rand_init(bool *predicate, const size_t n, TypeValue threshold);

template <typename TypeInValue, typename TypeOutValue>
void device_linearize(TypeOutValue *output,
                      const TypeInValue *const *inputs,
                      size_t dims,
                      size_t elements);

template <typename T>
int device_compact(bool cond,
                   const T **sparse,
                   T **dense,
                   const bool *dPredicate,
                   const size_t length,
                   int dims,
                   int blockSize,
                   bool isReverse = false);

template <typename T>
int device_compact(bool cond,
                   T **sparse,
                   T **dense,
                   int *indices,
                   const size_t length,
                   int dims,
                   int blockSize,
                   const bool *dPredicate,
                   bool isReverse = false);


PERFFASPECT()
inline void computePredicate(float *data,
                             bool *predicate,
                             size_t nData,
                             const size_t kneigh,
                             float threshold)
{
  return device_compute_predicate(data, predicate, nData, kneigh, threshold);
}


template <typename TypeInValue, typename TypeOutValue>
PERFFASPECT()
inline void linearize(TypeOutValue *output,
                      const TypeInValue *const *inputs,
                      size_t dims,
                      size_t elements)
{
  return device_linearize(output, inputs, dims, elements);
}

template <typename TypeValue>
PERFFASPECT()
inline int pack(bool cond,
                const bool *predicate,
                const size_t n,
                const TypeValue **sparse,
                TypeValue **dense,
                int dims)
{
  return device_compact(cond, sparse, dense, predicate, n, dims, 1024);
}

template <typename TypeValue>
PERFFASPECT()
inline int pack(bool cond,
                const bool *predicate,
                const size_t n,
                TypeValue **sparse,
                TypeValue **dense,
                int *sparse_indices,
                int dims)
{
  return device_compact(
      cond, sparse, dense, sparse_indices, n, dims, 1024, predicate);
}

template <typename TypeValue>
PERFFASPECT()
inline int unpack(bool cond,
                  const bool *predicate,
                  const size_t n,
                  TypeValue **sparse,
                  TypeValue **dense,
                  int dims)
{
  return device_compact(cond,
                        const_cast<const TypeValue **>(sparse),
                        dense,
                        predicate,
                        n,
                        dims,
                        1024,
                        true);
}

template <typename TypeValue>
PERFFASPECT()
inline int unpack(bool cond,
                  const size_t n,
                  TypeValue **sparse,
                  TypeValue **dense,
                  int *sparse_indices,
                  int dims)
{
  return device_compact(
      cond, sparse, dense, sparse_indices, n, dims, 1024, NULL, true);
}

}  // namespace Device
}  // namespace ams

#else

namespace ams
{


PERFFASPECT()
inline void DtoDMemcpy(void *dest, void *src, size_t nBytes)
{
  FATAL(Device, "DtoD Memcpy Not Enabled");
}

PERFFASPECT()
inline void HtoHMemcpy(void *dest, void *src, size_t nBytes)
{
  std::memcpy(dest, src, nBytes);
}

PERFFASPECT()
inline void HtoDMemcpy(void *dest, void *src, size_t nBytes)
{
  FATAL(Device, "HtoD Memcpy Not Enabled");
}

PERFFASPECT()
inline void DtoHMemcpy(void *dest, void *src, size_t nBytes)
{
  FATAL(Device, "DtoH Memcpy Not Enabled");
}


inline void *DeviceAllocate(size_t nBytes)
{
  FATAL(Device, "DtoH Memcpy Not Enabled");
}


PERFFASPECT()
inline void DeviceFree(void *ptr) { FATAL(Device, "DtoH Memcpy Not Enabled"); }

PERFFASPECT()
inline void *DevicePinnedAlloc(size_t nBytes)
{
  FATAL(Device, "Pinned Alloc Not Enabled");
}

PERFFASPECT()
inline void DeviceFreePinned(void *ptr)
{
  FATAL(Device, "Pinned Free Pinned Not Enabled");
}

inline void device_random_uq(int seed,
                             bool *uq_flags,
                             int ndata,
                             double acceptable_error)
{
  FATAL(Device, "Called Device Runtime UQ without enabling Device compilation");
}


inline void deviceCheckErrors(const char *file, int line) { return; }

namespace Device
{
PERFFASPECT()
inline void computePredicate(float *data,
                             bool *predicate,
                             size_t nData,
                             const size_t kneigh,
                             float threshold)
{
  FATAL(Device, "Called device code when CUDA disabled");
  return;
}


template <typename TypeInValue, typename TypeOutValue>
PERFFASPECT()
inline void linearize(TypeOutValue *output,
                      const TypeInValue *const *inputs,
                      size_t dims,
                      size_t elements)
{
  FATAL(Device, "Called device code when CUDA disabled");
  return;
}

template <typename TypeValue>
PERFFASPECT()
inline int pack(bool cond,
                const bool *predicate,
                const size_t n,
                const TypeValue **sparse,
                TypeValue **dense,
                int dims)
{
  FATAL(Device, "Called device code when CUDA disabled");
  return UNDEFINED_FUNC;
}

template <typename TypeValue>
PERFFASPECT()
inline int pack(bool cond,
                const bool *predicate,
                const size_t n,
                TypeValue **sparse,
                TypeValue **dense,
                int *sparse_indices,
                int dims)
{
  FATAL(Device, "Called device code when CUDA disabled");
  return UNDEFINED_FUNC;
}

template <typename TypeValue>
PERFFASPECT()
inline int unpack(bool cond,
                  const bool *predicate,
                  const size_t n,
                  TypeValue **sparse,
                  TypeValue **dense,
                  int dims)
{
  FATAL(Device, "Called device code when CUDA disabled");
  return UNDEFINED_FUNC;
}

template <typename TypeValue>
PERFFASPECT()
inline int unpack(bool cond,
                  const size_t n,
                  TypeValue **sparse,
                  TypeValue **dense,
                  int *sparse_indices,
                  int dims)
{
  FATAL(Device, "Called device code when CUDA disabled");
  return UNDEFINED_FUNC;
}

}  // namespace Device
}  // namespace ams

#endif


#endif
