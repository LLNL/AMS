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


#endif
