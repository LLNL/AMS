#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cstdint>
#include <random>


double unitrand() { return (double)rand() / RAND_MAX; }


template <typename T>
static inline
T* create_random(const size_t dim, const size_t n) {

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    T *data = new T[dim * n];

    for (size_t i = 0; i < n; i++) {
        for (size_t d = 0; d < dim; d++) {
            data[dim*i + d] = distrib(rng);
            data[dim*i] += i / 1000.;
        }
    }
    return data;
}



#include <curand.h>
#include <curand_kernel.h>
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
      uq_flags[i] = ((double)curand(&state) / RAND_MAX) <= acceptable_error;
    }
}

void random_uq_host(bool *uq_flags, int ndata, double acceptable_error) {

  for(int i = 0; i < ndata; i++) {
      uq_flags[i] = ((double)rand() / RAND_MAX) <= acceptable_error;
  }
}




#endif
