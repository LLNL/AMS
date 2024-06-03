/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __DEVICE_UTILITIES__
#define __DEVICE_UTILITIES__


#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <iostream>

#include "wf/device.hpp"
#include "wf/resource_manager.hpp"

namespace ams
{
namespace Device
{

const int warpSize = 32;
const unsigned int fullMask = 0xffffffff;

__host__ int divup(int x, int y) { return (x + y - 1) / y; }

__device__ __inline__ int pow2i(int e) { return 1 << e; }

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CUDASAFECALL(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CUDACHECKERROR() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
    fprintf(stderr,
            "cudaSafeCall() failed at %s:%i : %s\n",
            file,
            line,
            cudaGetErrorString(err));

    fprintf(stdout,
            "cudaSafeCall() failed at %s:%i : %s\n",
            file,
            line,
            cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

struct is_true {
  __host__ __device__ bool operator()(const int x) { return x; }
};

struct is_false {
  __host__ __device__ bool operator()(const int x) { return !x; }
};


inline void __cudaCheckError(const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr,
            "cudaCheckError() failed at %s:%i : %s\n",
            file,
            line,
            cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr,
            "cudaCheckError() with sync failed at %s:%i : %s\n",
            file,
            line,
            cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

__global__ void srand_dev(curandState* states, const int total_threads)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < total_threads) {
    int seed = id;  // different seed per thread
    curand_init(seed, id, 0, &states[id]);
  }
}

__global__ void initIndices(int* ind, int length)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < length) ind[id] = id;
}

template <typename T>
__global__ void fillRandom(bool* predicate,
                           const int total_threads,
                           curandState* states,
                           const size_t length,
                           T threshold)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < total_threads) {
    for (int i = id; i < length; i += total_threads) {
      float x = curand_uniform(&states[id]);
      predicate[i] = (x <= threshold);
    }
  }
}

template <typename T>
__global__ void computeBlockCounts(bool cond,
                                   T* d_input,
                                   int length,
                                   int* d_BlockCounts)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < length) {
    int pred = (d_input[idx] == cond);
    int BC = __syncthreads_count(pred);

    if (threadIdx.x == 0) {
      d_BlockCounts[blockIdx.x] =
          BC;  // BC will contain the number of valid elements in all threads of this thread block
    }
  }
}

template <typename T>
__global__ void assignK(T** sparse,
                        T** dense,
                        int* indices,
                        size_t length,
                        int dims,
                        bool isReverse)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < length) {
    int index = indices[idx];
    if (!isReverse) {
      for (int i = 0; i < dims; i++) {
        dense[i][idx] = sparse[i][index];
      }
    } else {
      for (int i = 0; i < dims; i++) {
        sparse[i][index] = dense[i][idx];
      }
    }
  }
}

template <typename T>
__global__ void device_compactK(bool cond,
                                T** d_input,
                                T** d_output,
                                const bool* predicates,
                                const size_t length,
                                int dims,
                                int* d_BlocksOffset,
                                bool reverse)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ int warpTotals[];
  if (idx < length) {
    int pred = (predicates[idx] == cond);
    int w_i = threadIdx.x / warpSize;  //warp index
    int w_l = idx % warpSize;          //thread index within a warp

    // compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
    int t_m = fullMask >> (warpSize - w_l);  //thread mask
#if (CUDART_VERSION < 9000)
    int b = __ballot(pred) & t_m;  //ballot result = number whose ith bit
                                   //is one if the ith's thread pred is true
                                   //masked up to the current index in warp
#else
    int b = __ballot_sync(fullMask, pred) & t_m;
#endif
    int t_u = __popc(
        b);  // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

    // last thread in warp computes total valid counts for the warp
    if (w_l == warpSize - 1) {
      warpTotals[w_i] = t_u + pred;
    }

    // need all warps in thread block to fill in warpTotals before proceeding
    __syncthreads();

    // first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
    int numWarps = blockDim.x / warpSize;
    unsigned int numWarpsMask = fullMask >> (warpSize - numWarps);
    if (w_i == 0 && w_l < numWarps) {
      int w_i_u = 0;
      for (int j = 0; j <= 5; j++) {
#if (CUDART_VERSION < 9000)
        int b_j = __ballot(
            warpTotals[w_l] &
            pow2i(j));  //# of the ones in the j'th digit of the warp offsets
#else
        int b_j = __ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j));
#endif
        w_i_u += (__popc(b_j & t_m)) << j;
      }
      warpTotals[w_l] = w_i_u;
    }

    // need all warps in thread block to wait until prefix sum is calculated in warpTotals
    __syncthreads();

    // if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
    if (pred) {
      if (!reverse) {
        for (int i = 0; i < dims; i++)
          d_output[i][t_u + warpTotals[w_i] + d_BlocksOffset[blockIdx.x]] =
              d_input[i][idx];
      } else {
        for (int i = 0; i < dims; i++)
          d_input[i][idx] =
              d_output[i][t_u + warpTotals[w_i] + d_BlocksOffset[blockIdx.x]];
      }
    }
  }
}


template <typename TypeInValue, typename TypeOutValue>
void __global__ linearizeK(TypeOutValue* output,
                           const TypeInValue* const* inputs,
                           size_t dims,
                           size_t elements)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= elements) return;

  for (int i = 0; i < dims; i++) {
    output[idx * dims + i] = static_cast<TypeOutValue>(inputs[i][idx]);
  }
}


void __global__ compute_predicate(float* data,
                                  bool* predicate,
                                  size_t nData,
                                  const size_t kneigh,
                                  float threshold)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= nData) return;

  int index = idx * kneigh;
  float acc = 0.0f;
  for (int i = 0; i < kneigh; i++) {
    acc += data[index + i];
  }

  acc /= static_cast<float>(kneigh);

  bool pred = acc < threshold ? true : false;

  predicate[idx] = pred;
}

template <typename T>
int device_compact(bool cond,
                   const T** sparse,
                   T** dense,
                   const bool* dPredicate,
                   const size_t length,
                   int dims,
                   int blockSize,
                   bool isReverse)
{
  int numBlocks = divup(length, blockSize);
  auto& rm = ams::ResourceManager::getInstance();
  int* d_BlocksCount = rm.allocate<int>(numBlocks, AMSResourceType::AMS_DEVICE);
  int* d_BlocksOffset =
      rm.allocate<int>(numBlocks, AMSResourceType::AMS_DEVICE);
  // determine number of elements in the compacted list
  int* h_BlocksCount = rm.allocate<int>(numBlocks, AMSResourceType::AMS_HOST);
  int* h_BlocksOffset = rm.allocate<int>(numBlocks, AMSResourceType::AMS_HOST);

  T** d_dense = rm.allocate<T*>(dims, AMSResourceType::AMS_DEVICE);
  T** d_sparse = rm.allocate<T*>(dims, AMSResourceType::AMS_DEVICE);


  rm.copy(dense,
          AMSResourceType::AMS_HOST,
          d_dense,
          AMSResourceType::AMS_DEVICE,
          dims);
  rm.copy(const_cast<T**>(sparse),
          AMSResourceType::AMS_HOST,
          d_sparse,
          AMSResourceType::AMS_DEVICE,
          dims);

  thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
  thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);

  //phase 1: count number of valid elements in each thread block
  computeBlockCounts<<<numBlocks, blockSize>>>(cond,
                                               dPredicate,
                                               length,
                                               d_BlocksCount);

  //phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
  thrust::exclusive_scan(thrust::device,
                         d_BlocksCount,
                         d_BlocksCount + numBlocks,
                         d_BlocksOffset);

  //phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
  device_compactK<<<numBlocks,
                    blockSize,
                    sizeof(int) * (blockSize / warpSize)>>>(cond,
                                                            d_sparse,
                                                            d_dense,
                                                            dPredicate,
                                                            length,
                                                            dims,
                                                            d_BlocksOffset,
                                                            isReverse);
  cudaDeviceSynchronize();
  CUDACHECKERROR();

  rm.copy(d_BlocksCount,
          AMSResourceType::AMS_DEVICE,
          h_BlocksCount,
          AMSResourceType::AMS_HOST,
          numBlocks);
  rm.copy(d_BlocksOffset,
          AMSResourceType::AMS_DEVICE,
          h_BlocksOffset,
          AMSResourceType::AMS_HOST,
          numBlocks);

  int compact_length =
      h_BlocksOffset[numBlocks - 1] + thrustPrt_bCount[numBlocks - 1];

  rm.deallocate(d_BlocksCount, AMSResourceType::AMS_DEVICE);
  rm.deallocate(d_BlocksOffset, AMSResourceType::AMS_DEVICE);

  rm.deallocate(h_BlocksCount, AMSResourceType::AMS_HOST);
  rm.deallocate(h_BlocksOffset, AMSResourceType::AMS_HOST);

  rm.deallocate(d_dense, AMSResourceType::AMS_DEVICE);
  rm.deallocate(d_sparse, AMSResourceType::AMS_DEVICE);

  cudaDeviceSynchronize();
  CUDACHECKERROR();

  return compact_length;
}

template <typename T>
int device_compact(bool cond,
                   T** sparse,
                   T** dense,
                   int* indices,
                   const size_t length,
                   int dims,
                   int blockSize,
                   const bool* dPredicate,
                   bool isReverse)
{
  int numBlocks = divup(length, blockSize);
  size_t sparseElements = length;

  if (!isReverse) {
    initIndices<<<numBlocks, blockSize>>>(indices, length);
    if (cond) {
      auto last = thrust::copy_if(thrust::device,
                                  indices,
                                  indices + sparseElements,
                                  dPredicate,
                                  indices,
                                  is_true());
      sparseElements = last - indices;
    } else {
      auto last = thrust::copy_if(thrust::device,
                                  indices,
                                  indices + sparseElements,
                                  dPredicate,
                                  indices,
                                  is_false());
      sparseElements = last - indices;
    }
  }

  assignK<<<numBlocks, blockSize>>>(
      sparse, dense, indices, sparseElements, dims, isReverse);
  cudaDeviceSynchronize();
  CUDACHECKERROR();

  return sparseElements;
}

template <typename TypeInValue, typename TypeOutValue>
void device_linearize(TypeOutValue* output,
                      const TypeInValue* const* inputs,
                      size_t dims,
                      size_t elements)
{
  // TODO: Fix "magic number".
  const int NT = 256;
  // TODO: We should add a max number of blocks typically this should be around 3K.
  int NB = (elements + NT - 1) / NT;
  DBG(Device,
      "Linearize using %ld blocks %ld threads to transpose %ld, %ld matrix",
      NB,
      NT,
      dims,
      elements);

  linearizeK<<<NB, NT>>>(output, inputs, dims, elements);
  cudaDeviceSynchronize();
  CUDACHECKERROR();
}

template <typename T>
void cuda_rand_init(bool* predicate, const size_t length, T threshold)
{
  static curandState* dev_random = NULL;
  const int TS = 4096;
  const int BS = 128;
  int numBlocks = divup(TS, BS);
  auto& rm = ams::ResourceManager::getInstance();
  if (!dev_random) {
    dev_random = rm.allocate<curandState>(4096, AMSResourceType::AMS_DEVICE);
    srand_dev<<<numBlocks, BS>>>(dev_random, TS);
  }

  DBG(Device,
      "Random Fill using %ld blocks %ld threads to randomly initialize %ld "
      "elements",
      numBlocks,
      BS,
      length);
  fillRandom<<<numBlocks, BS>>>(predicate, TS, dev_random, length, threshold);
  cudaDeviceSynchronize();
  CUDACHECKERROR();
}


void device_compute_predicate(float* data,
                              bool* predicate,
                              size_t nData,
                              const size_t kneigh,
                              float threshold)
{
  const int NT = 256;
  int NB = (nData + NT - 1) / NT;
  DBG(Device,
      "Compute predicate for %d elements with threshold %f",
      nData,
      threshold);
  compute_predicate<<<NB, NT>>>(data, predicate, nData, kneigh, threshold);
  cudaDeviceSynchronize();
  CUDACHECKERROR();
}

__global__ void random_uq_device(int seed,
                                 bool* uq_flags,
                                 int ndata,
                                 double acceptable_error)
{

  /* CUDA's random number library uses curandState_t to keep track of the seed
     value we will store a random state for every thread  */
  curandState_t state;
  int id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id >= ndata) return;

  /* we have to initialize the state */
  curand_init(
      seed +
          id, /* the seed controls the sequence of random values that are produced */
      0,      /* the sequence number is only important with multiple cores */
      0, /* the offset is how much extra we advance in the sequence for each
            call, can be 0 */
      &state);

  float x = curand_uniform(&state);
  uq_flags[id] = (x <= acceptable_error);
}


template <typename scalar_t>
__global__ void computeDeltaUQMeanPredicatesKernel(
    const scalar_t* __restrict__ outputs_stdev,
    bool* __restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold)
{

  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  // Compute mean over columns, strided loop.
  for (size_t i = idx; i < nrows; i += stride) {
    double mean = 0.0;
    for (size_t j = 0; j < ncols; ++j)
      mean += outputs_stdev[j + i * ncols];
    mean /= ncols;

    predicates[i] = (mean < threshold);
  }
}

template <typename scalar_t>
__global__ void computeDeltaUQMaxPredicatesKernel(
    const scalar_t* __restrict__ outputs_stdev,
    bool* __restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold)
{

  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  // Compute max delta uq over columns, strided loop.
  for (size_t i = idx; i < nrows; i += stride) {
    predicates[i] = true;
    for (size_t j = 0; j < ncols; ++j)
      if (outputs_stdev[j + i * ncols] >= threshold) {
        predicates[i] = false;
        break;
      }
  }
}


template <typename TypeValue>
void rand_init(bool* predicate, const size_t n, TypeValue threshold)
{
  cuda_rand_init(predicate, n, threshold);
  return;
}

template <typename scalar_t>
void computeDeltaUQMeanPredicatesDevice(
    const scalar_t* __restrict__ outputs_stdev,
    bool* __restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold)
{
  constexpr int block_size = 256;
  int grid_size = divup(nrows, block_size);
  computeDeltaUQMeanPredicatesKernel<<<grid_size, block_size>>>(
      outputs_stdev, predicates, nrows, ncols, threshold);
  cudaDeviceSynchronize();
  CUDACHECKERROR();
};

template <typename scalar_t>
void computeDeltaUQMaxPredicatesDevice(
    const scalar_t* __restrict__ outputs_stdev,
    bool* __restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold)
{
  constexpr int block_size = 256;
  int grid_size = divup(nrows, block_size);
  computeDeltaUQMaxPredicatesKernel<<<grid_size, block_size>>>(
      outputs_stdev, predicates, nrows, ncols, threshold);
  cudaDeviceSynchronize();
  CUDACHECKERROR();
}


// Specializations

template void computeDeltaUQMaxPredicatesDevice<float>(
    const float* __restrict__ outputs_stdev,
    bool* __restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold);

template void computeDeltaUQMaxPredicatesDevice<double>(
    const double* __restrict__ outputs_stdev,
    bool* __restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold);

template void computeDeltaUQMeanPredicatesDevice<float>(
    const float* __restrict__ outputs_stdev,
    bool* __restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold);

template void computeDeltaUQMeanPredicatesDevice<double>(
    const double* __restrict__ outputs_stdev,
    bool* __restrict__ predicates,
    const size_t nrows,
    const size_t ncols,
    const double threshold);

template void cuda_rand_init<float>(bool* predicate,
                                    const size_t length,
                                    float threshold);

template void cuda_rand_init<double>(bool* predicate,
                                     const size_t length,
                                     double threshold);


template void device_linearize<float, float>(float* output,
                                             const float* const* inputs,
                                             size_t dims,
                                             size_t elements);

template void device_linearize<float, double>(double* output,
                                              const float* const* inputs,
                                              size_t dims,
                                              size_t elements);

template void device_linearize<double, double>(double* output,
                                               const double* const* inputs,
                                               size_t dims,
                                               size_t elements);

template void device_linearize<double, float>(float* output,
                                              const double* const* inputs,
                                              size_t dims,
                                              size_t elements);

template int device_compact<double>(bool cond,
                                    const double** sparse,
                                    double** dense,
                                    const bool* dPredicate,
                                    const size_t length,
                                    int dims,
                                    int blockSize,
                                    bool isReverse);

template int device_compact<float>(bool cond,
                                   const float** sparse,
                                   float** dense,
                                   const bool* dPredicate,
                                   const size_t length,
                                   int dims,
                                   int blockSize,
                                   bool isReverse);

template int device_compact<double>(bool cond,
                                    double** sparse,
                                    double** dense,
                                    int* indices,
                                    const size_t length,
                                    int dims,
                                    int blockSize,
                                    const bool* dPredicate,
                                    bool isReverse);

template int device_compact<float>(bool cond,
                                   float** sparse,
                                   float** dense,
                                   int* indices,
                                   const size_t length,
                                   int dims,
                                   int blockSize,
                                   const bool* dPredicate,
                                   bool isReverse);


template void rand_init<double>(bool* predicate,
                                const size_t n,
                                double threshold);

template void rand_init<float>(bool* predicate,
                               const size_t n,
                               float threshold);

}  // namespace Device


void DtoDMemcpy(void* dest, void* src, size_t nBytes)
{
  cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToDevice);
}

void HtoHMemcpy(void* dest, void* src, size_t nBytes)
{
  std::memcpy(dest, src, nBytes);
}

void HtoDMemcpy(void* dest, void* src, size_t nBytes)
{
  cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice);
};

void DtoHMemcpy(void* dest, void* src, size_t nBytes)
{
  cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost);
}

void* DeviceAllocate(size_t nBytes)
{
  void* devPtr;
  cudaMalloc(&devPtr, nBytes);
  return devPtr;
}

void DeviceFree(void* ptr)
{
  cudaFree(ptr);
  return;
}

void* DevicePinnedAlloc(size_t nBytes)
{
  void* ptr;
  cudaHostAlloc(&ptr, nBytes, cudaHostAllocPortable);
  return ptr;
}

void DeviceFreePinned(void* ptr) { cudaFreeHost(ptr); }


void deviceCheckErrors(const char* file, int line)
{
  ams::Device::__cudaCheckError(file, line);
}

void device_random_uq(int seed,
                      bool* uq_flags,
                      int ndata,
                      double acceptable_error)
{
  size_t block_size = 256;
  size_t blocks = ams::Device::divup(ndata, block_size);
  ams::Device::random_uq_device<<<blocks, block_size>>>(seed,
                                                        uq_flags,
                                                        ndata,
                                                        acceptable_error);
}


}  // namespace ams

#endif
