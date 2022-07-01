#ifndef __HDCACHE_HPP__
#define __HDCACHE_HPP__

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <type_traits>

//! ----------------------------------------------------------------------------
//! An abstract definition of the HDCache class
//! ----------------------------------------------------------------------------
template <typename TypeInValue=double>
class HDCache {

    static_assert (std::is_floating_point<TypeInValue>::value,
                  "HDCache supports floating-point values (floats, doubles, and long doubles) only!");

protected:
    const uint8_t m_dim;
    const uint8_t m_knbrs;
    const bool m_use_device;
    const std::string m_type;

    HDCache(uint8_t dim, uint8_t knbrs, bool use_device, const std::string type) :
        m_dim(dim), m_knbrs(knbrs), m_use_device(use_device), m_type(type) {}

public:
    // todo: should be an operator
    virtual inline void
    print() const {
        std::cout << "HDCache_"<<m_type<<" (on device = "<< m_use_device<< ", ";
        if (has_index()) { std::cout << "index = null";          }
        else {             std::cout << "npoints = " << count(); }
        std::cout << ")\n";;
    }

    virtual inline bool has_index() const { return false; }
    virtual inline size_t count() const { return 0;    }

    virtual inline void
    load_cache(const std::string &) {
        std::cerr << "HDCache::load_cache() is a no-op!\n";
    }
    virtual inline void
    save_cache(const std::string &) const {
        std::cerr << "HDCache::save_cache() is a no-op\n";
    }

    virtual inline void
    add(const size_t, const size_t, TypeInValue *) {
        std::cerr << "HDCache::add() is a no-op\n";
    }
    virtual inline void
    add(const size_t, const std::vector<TypeInValue *> &) {
        std::cerr << "HDCache::add() is a no-op\n";
    }

    virtual inline void
    train(const size_t, const size_t, TypeInValue *) {
        std::cerr << "HDCache::train() is a no-op\n";
    }
    virtual inline void
    train(const size_t, const std::vector<TypeInValue *> &) {
        std::cerr << "HDCache::train() is a no-op\n";
    }

    // any child class must implement these!
    virtual inline void
    evaluate(const size_t, const size_t, TypeInValue *, bool *) = 0;

    virtual inline void
    evaluate(const size_t, const std::vector<TypeInValue*> &, bool *) = 0;
};

#endif
