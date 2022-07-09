#ifndef __AMS_HDCACHE_RANDOM_HPP__
#define __AMS_HDCACHE_RANDOM_HPP__

#include <cstdint>
#include <vector>
#include <iostream>

#include "utils/utils_data.hpp"
#include "utils/allocator.hpp"
#include "hdcache.hpp"

//! ----------------------------------------------------------------------------
//! An implementation of random HDCache
//! ----------------------------------------------------------------------------
template <typename TypeInValue>
class HDCache_Random : public HDCache<TypeInValue> {

public:
    //! -----------------------------------------------------------------------
    //! constructors
    //! -----------------------------------------------------------------------
    HDCache_Random(uint8_t dim, uint8_t knbrs, bool use_device) :
        HDCache<TypeInValue>(dim, knbrs, use_device, std::string("random")) {}

    inline void
    evaluate(const size_t ndata, const size_t d, TypeInValue *data,
             bool *is_acceptable) const {

        static const TypeInValue acceptable_error = 0.5;

        if (ams::ResourceManager::is_on_device(is_acceptable)) {
#ifdef __ENABLE_CUDA__
            random_uq_device<<<1,1>>>(is_acceptable, ndata, acceptable_error);
#else
            std::cerr << "Data should not be resident on device if CUDA is not available!\n";
            exit(1);
#endif
        }
        else {
            random_uq_host(is_acceptable, ndata, acceptable_error);
        }
    }

    inline void
    evaluate(const size_t ndata, const std::vector<TypeInValue*> &inputs,
             bool *is_acceptable) const {

        return evaluate(ndata, inputs.size(), nullptr, is_acceptable);
    }
};

#endif
