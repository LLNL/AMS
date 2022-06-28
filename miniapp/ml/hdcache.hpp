#ifndef __HDCACHE_HPP__
#define __HDCACHE_HPP__

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

#ifdef __ENABLE_FAISS__
#include <faiss/index_io.h>
#include <faiss/index_factory.h>
#include "faiss/IndexFlat.h"
#endif

#include "utils/data_handler.hpp"

#if __cplusplus < 201402L
  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;
#else

#endif


//! ----------------------------------------------------------------------------
//! An implementation of hdcache
//! the idea is to have a single class that exposes
//! the functionality to compute uncertainty
template <typename TypeInValue=double>
class HDCache {

    static_assert (std::is_floating_point<TypeInValue>::value,
                  "HDCache supports floating-point values (floats, doubles, or long doubles) only!");

    using TypeValue = float;                        // faiss uses floats
    using data_handler = DataHandler<TypeValue>;    // utils to handle float data

#ifdef __ENABLE_FAISS__
    using TypeIndex = faiss::Index::idx_t;          // 64-bit int

    faiss::Index* index;
    const char* index_key = "IVF4096,Flat";
#else
    using TypeIndex = uint64_t;

    void *index;
    const char* index_key = "unknown";
#endif

    const uint8_t dim;
    const uint8_t knbrs;
    bool is_cpu;


#ifdef __ENABLE_FAISS__
    //! -----------------------------------------------------------------------
    //! core faiss functionality. working with linearized TypeValue data
    //! -----------------------------------------------------------------------

    //! add points to index when  (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _add(const size_t ndata, const T *data) {
        index->add(ndata, data);
    }


    //! train an index when (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _train(const size_t ndata, const T *data) {
        if (index != nullptr && index->is_trained)
            throw std::invalid_argument("Trying to re-train an already trained index!");

        index = faiss::index_factory(dim, index_key);
        index->train(ndata, data);

        if (!index->is_trained)
            throw std::runtime_error("Failed to train the index!");
    }


    //! evaluate cache uncertainty when  (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    void
    _evaluate(const size_t ndata, T *data, bool *is_acceptable) const {
        static const TypeInValue acceptable_error = 0.5;
        static const TypeValue ook = 1.0 / TypeValue(knbrs);

        // create memory for the output
        std::vector<TypeIndex> kidxs (ndata*knbrs);
        std::vector<TypeValue> kdists (ndata*knbrs);

        index->search(ndata, data, knbrs, kdists.data(), kidxs.data());

        TypeValue mean_dist = 0;

        auto curr_beg = kdists.begin();
        for (size_t i = 0; i < ndata; ++i, curr_beg += knbrs) {
            mean_dist = ook * std::accumulate(curr_beg, curr_beg+knbrs, 0.);
            is_acceptable[i] = mean_dist < acceptable_error;
        }
    }


    //! -----------------------------------------------------------------------
    //! use the above functionality but also when linearized data is not TypeValue
    //! -----------------------------------------------------------------------

    //! add points to index when (data type != TypeValue)
    template <class T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _add(const size_t ndata, const T *data) {
        TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
        _add(ndata, vdata);
        delete [] vdata;
    }


    //! train an index when (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _train(const size_t ndata, const T *data) {
        TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
        _train(ndata, vdata);
        delete []  vdata;
    }


    //! evaluate cache uncertainty when (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _evaluate(const size_t ndata, T *data, bool *is_acceptable)  const {
        TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
        _evaluate(ndata, data, is_acceptable);
        delete []  vdata;
    }

#endif

public:
    //! -----------------------------------------------------------------------
    //! constructors
    //! -----------------------------------------------------------------------
    HDCache(uint8_t _dim, bool _is_cpu = true, uint8_t _knbrs = 10) :
        dim(_dim), knbrs(_knbrs), is_cpu(_is_cpu), index(nullptr) {
    }

    void load_cache(const std::string &filename) {
#ifdef __ENABLE_FAISS__
        index = faiss::read_index(filename.c_str());
        std::cout << "Loaded hd cache with " << index->ntotal << " points from ("<<filename<<")\n";
#else
        std::cerr << "HDCache::load_cache() is a no-op without Faiss\n";
#endif
    }
    void save_cache(const std::string &filename) {
#ifdef __ENABLE_FAISS__
        std::cout << "Saving hd cache with " << index->ntotal << " points to ("<<filename<<")\n";
        faiss::write_index(index, filename.c_str());
#else
        std::cerr << "HDCache::save_cache() is a no-op without Faiss\n";
#endif
    }

    //! -----------------------------------------------------------------------
#ifdef __ENABLE_FAISS__
    inline bool has_index() const { return index != nullptr && index->is_trained; }
    inline bool count() const {     return has_index() ? index->ntotal : 0;       }
#else
    inline bool has_index() const { return false; }
    inline bool count() const {     return 0;     }
#endif

    //! -----------------------------------------------------------------------
    //! add points to the cache
    //! -----------------------------------------------------------------------

    //! add the data that comes as linearized features
    void add(const size_t ndata, const size_t d, TypeInValue *data) {
#ifdef __ENABLE_FAISS__
        std::cout << "Adding " << ndata << " " << d << "-dim points!\n";

        if (d != dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (!has_index())
            throw std::invalid_argument("Does not have a valid and trained index!");

        _add(ndata, data);
        std::cout << "Successfully added! Cache has " << index->ntotal << " points!\n";
#else
        std::cerr << "HDCache::add() is a no-op without Faiss\n";
#endif
    }

    //! add the data that comes as separate features (a vector of pointers)
    void add(const size_t ndata, const std::vector<const TypeInValue *> &data) {
#ifdef __ENABLE_FAISS__
        std::cout << "Adding " << ndata << " " << data.size() << "-dim points!\n";

        if (data.size() != dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (!has_index())
            throw std::invalid_argument("Does not have a valid and trained index!");

        std::vector<TypeValue> lin_data = data_handler::linearize_features(ndata, data);
        _add(ndata, lin_data.data());
        lin_data.clear();

        std::cout << "Successfully added! Cache has " << index->ntotal << " points!\n";
#else
        std::cerr << "HDCache::add() is a no-op without Faiss\n";
#endif
    }

    //! -----------------------------------------------------------------------
    //! train a cache
    //! -----------------------------------------------------------------------

    //! train on data that comes as linearized features
    void train(const size_t ndata, const size_t d, TypeInValue *data) {
#ifdef __ENABLE_FAISS__
        std::cout << "Training a " << int(dim) << "-dim cache "
                  << "using " << ndata << " " << d << "-dim points!\n";

        if (d != dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (has_index())
            throw std::invalid_argument("Already have a valid and trained index!");

        _train(ndata, data);
        std::cout << "Successfully trained " << int(dim) << "-dim faiss index!\n";
#else
        std::cerr << "HDCache::train() is a no-op without Faiss\n";
#endif
    }

    //! train on data that comes separate features (a vector of pointers)
    void train(const size_t ndata, const std::vector<const TypeInValue *> &data) {
#ifdef __ENABLE_FAISS__
        std::cout << "Training a " << int(dim) << "-dim cache "
                  << "using " << ndata << " " << data.size() << "-dim points!\n";

        if (data.size() != dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (has_index())
            throw std::invalid_argument("Already have a valid and trained index!");

        std::vector<TypeValue> lin_data = data_handler::linearize_features(ndata, data);
        _train(ndata, lin_data.data());
        lin_data.clear();

        std::cout << "Successfully trained " << int(dim) << "-dim faiss index!\n";
#else
        std::cerr << "HDCache::train() is a no-op without Faiss\n";
#endif
    }


    //! -----------------------------------------------------------------------
    //! evaluate uncertainty using the cache
    //! -----------------------------------------------------------------------
    //! train on data that comes as linearized features
    void evaluate(const size_t ndata, const size_t d, TypeInValue *data,
                  bool *is_acceptable) const {
#ifdef __ENABLE_FAISS__
        std::cout << "Evaluating a " << int(dim) << "-dim cache "
                  << "using " << ndata << " " << d << "-dim points!\n";

        if (d != dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        _evaluate(ndata, data, is_acceptable);
        std::cout << "Successfully evaluated " << ndata << " " << d << "-dim points!\n";
#else
        static const TypeInValue acceptable_error = 0.5;
        for(int i = 0; i < ndata; i++) {
          is_acceptable[i] = ((TypeInValue)rand() / RAND_MAX) <= acceptable_error;
        }
#endif
   }

    //! train on data that comes separate features (a vector of pointers)
    void evaluate(const size_t ndata, const std::vector<TypeInValue*> &inputs,
                  bool *is_acceptable) const {
#ifdef __ENABLE_FAISS__
        std::cout << "Evaluating a " << int(dim) << "-dim cache "
                  << "using " << ndata << " " << inputs.size() << "-dim points!\n";

        if (inputs.size() != dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        std::vector<TypeValue> data = data_handler::linearize_features<TypeInValue>(ndata, inputs);
        _evaluate(ndata, data.data(), is_acceptable);
        data.clear();
        std::cout << "Successfully evaluated " << ndata << " " << inputs.size() << "-dim points!\n";
#else
        static const TypeInValue acceptable_error = 0.5;
        if ( is_cpu ){
        for(int i = 0; i < ndata; i++) {
          is_acceptable[i] = ((TypeInValue)rand() / RAND_MAX) <= acceptable_error;
        }
        }
        else{
          AMS::Device::rand_init(is_acceptable, ndata, acceptable_error);
        }
#endif
    }

    //! -----------------------------------------------------------------------
};

#endif
