#ifndef __AMS_DEVICE_HPP__
#define __AMS_DEVICE_HPP__

#include <cstddef>
#include <cstring>
#include <iostream>

#ifdef __ENABLE_CUDA__
#include "cuda/utilities.cuh"
#endif

namespace ams {
namespace Device {

void computePredicate( float *data, bool *predicate, size_t nData, const size_t kneigh, float threshold){
#ifdef __ENABLE_CUDA__
  return device_compute_predicate(data, predicate, nData, kneigh, threshold);
#else
  return;
#endif
}

template<typename TypeInValue, typename TypeOutValue>
void linearize(TypeOutValue *output, const TypeInValue * const *inputs, size_t dims, size_t elements){
#ifdef __ENABLE_CUDA__
  return device_linearize(output, inputs, dims, elements);
#else
  return;
#endif
}

template <typename TypeValue>
int pack(bool cond, const bool* predicate, const size_t n, TypeValue** sparse,
         TypeValue** dense, int dims) {
#ifdef __ENABLE_CUDA__
    return compact(cond, sparse, dense, predicate, n, dims, 1024);
#else
    return 0;
#endif
}

template <typename TypeValue>
int pack(bool cond, const bool* predicate, const size_t n, TypeValue** sparse,
         TypeValue** dense, int* sparse_indices, int dims) {
#ifdef __ENABLE_CUDA__
    return compact(cond, sparse, dense, sparse_indices, n, dims, 1024, predicate);
#else
    return 0;
#endif
}

template <typename TypeValue>
int unpack(bool cond, const bool* predicate, const size_t n, TypeValue** sparse, TypeValue** dense, int dims) {
#ifdef __ENABLE_CUDA__
    return compact(cond, sparse, dense, predicate, n, dims, 1024, true);
#else
    return 0;
#endif
}

template <typename TypeValue>
int unpack(bool cond, const size_t n, TypeValue** sparse, TypeValue** dense, int* sparse_indices, int dims) {
#ifdef __ENABLE_CUDA__
    return compact(cond, sparse, dense, sparse_indices, n, dims, 1024, NULL, true);
#else
    return 0;
#endif
}

template <typename TypeValue>
void rand_init(bool *predicate, const size_t n, TypeValue threshold){
#ifdef __ENABLE_CUDA__
  cuda_rand_init(predicate, n, threshold);
#endif
  return;
}

}  // namespace Device
}  // namespace AMS


#ifdef __ENABLE_CUDA__

#include <curand.h>
#include <curand_kernel.h>
// HB's version of generating random data on device
// should switch to Dino's version
__global__ void random_uq_device(bool *uq_flags, int ndata, double acceptable_error) {

    /* CUDA's random number library uses curandState_t to keep track of the seed value
       we will store a random state for every thread  */
    curandState_t state;

    /* we have to initialize the state */
    curand_init(0, /* the seed controls the sequence of random values that are produced */
                0, /* the sequence number is only important with multiple cores */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);

    for(int i = 0; i < ndata; i++) {
      float x = curand_uniform(&state);
      uq_flags[i] = (x <= acceptable_error);
    }
}


#include <cuda_runtime.h>
inline void DtoDMemcpy(void *dest, void *src, size_t nBytes ){
  cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToDevice);
}

inline void HtoHMemcpy(void *dest, void *src, size_t nBytes ){
  std::memcpy(dest, src, nBytes);
}

inline void HtoDMemcpy(void *dest, void *src, size_t nBytes ){
  cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice);
};

void DtoHMemcpy(void *dest, void *src, size_t nBytes ){
  cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost);
}
#else
inline void DtoDMemcpy(void *dest, void *src, size_t nBytes ){
  std::cerr<< "DtoD Memcpy Not Enabled" << std::endl;
  exit(-1);
}

inline void HtoHMemcpy(void *dest, void *src, size_t nBytes ){
  std::memcpy(dest, src, nBytes);
}

inline void HtoDMemcpy(void *dest, void *src, size_t nBytes ){
  std::cerr<< "HtoD Memcpy Not Enabled" << std::endl;
  exit(-1);
};

void DtoHMemcpy(void *dest, void *src, size_t nBytes ){
  std::cerr<< "DtoH Memcpy Not Enabled" << std::endl;
  exit(-1);
}
#endif

#endif
