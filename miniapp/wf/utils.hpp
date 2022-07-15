#ifndef __AMS_UTILS_HPP__
#define __AMS_UTILS_HPP__

#include <algorithm>
#include <array>
#include <vector>
#include <random>
#include <iostream>


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#if __cplusplus < 201402L
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
#else
#endif


const int partitionSize = 1 << 24;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

template<typename T>
static void
print_tensor_array(const std::string &label,
                   const T *values,
                   const std::array<int,3> &sz) {

    const int K = sz[0], J = sz[1], I = sz[2];
    if (K == 1) {
        std::cout << "--> printing ["<<J<<" x "<<I<<"] tensor \""<<label<<"\"\n";
        for (int j = 0; j < J; ++j) {
        for (int i = 0; i < I; ++i) {
            int idx = i + I*j;
            std::cout << label << "["<<j<<","<<i<<"] = " << idx << " = " << values[idx] << std::endl;
        }}
    }
    else {
        std::cout << "--> printing ["<<K<<" x "<<J<<" x "<<I<<"] tensor \""<<label<<"\"\n";
        for (int k = 0; k < K; ++k) {
        for (int j = 0; j < J; ++j) {
        for (int i = 0; i < I; ++i) {
            int idx = i + I*(j + k*J);
            std::cout << label << "["<<k<<", "<<j<<","<<i<<"] = " << values[idx] << std::endl;
        }}}
    }
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

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

void random_uq_host(bool *uq_flags, int ndata, double acceptable_error) {

  for(int i = 0; i < ndata; i++) {
      uq_flags[i] = ((double)rand() / RAND_MAX) <= acceptable_error;
  }
}


// -----------------------------------------------------------------------------
#endif
