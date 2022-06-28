#ifndef __AMS_DEVICE__
#define __AMS_DEVICE__

#include <stddef.h>

#ifdef __ENABLE_CUDA__
#include "cuda/utilities.cuh"
#endif

namespace AMS {
namespace Device {

template <typename TypeValue>
int pack(const bool* predicate, const size_t n, TypeValue** sparse, TypeValue** dense, int dims) {
#ifdef __ENABLE_CUDA__
    return compact(sparse, dense, predicate, n, dims, 1024);
#else
    return 0;
#endif
}

template <typename TypeValue>
int pack(const bool* predicate, const size_t n, TypeValue** sparse, TypeValue** dense,
         int* sparse_indices, int dims) {
#ifdef __ENABLE_CUDA__
    return compact(sparse, dense, sparse_indices, n, dims, 1024, predicate);
#else
    return 0;
#endif
}

template <typename TypeValue>
int unpack(const bool* predicate, const size_t n, TypeValue** sparse, TypeValue** dense, int dims) {
#ifdef __ENABLE_CUDA__
    return compact(sparse, dense, predicate, n, dims, 1024, true);
#else
    return 0;
#endif
}

template <typename TypeValue>
int unpack(const size_t n, TypeValue** sparse, TypeValue** dense, int* sparse_indices, int dims) {
#ifdef __ENABLE_CUDA__
    return compact(sparse, dense, sparse_indices, n, dims, 1024, NULL, true);
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

#endif
