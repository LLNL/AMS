#ifndef _HDCACHE_HPP_
#define _HDCACHE_HPP_

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

#if __cplusplus < 201402L
  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;
#else

#endif


//! ----------------------------------------------------------------------------
//! An implementation of hdcache
//! the idea is to have a single class that exposes
//! the functionality to compute uncertainty
//template <typename TypeValue=double>
class HDCache {

    using TypeInValue = double;
    static_assert (std::is_floating_point<TypeInValue>::value,
                  "HDCache supports floating-point values (floats, doubles, or long doubles) only!");

#ifdef __ENABLE_FAISS__
    using TypeIndex = faiss::Index::idx_t;      // 64-bit int
#else
    using TypeIndex = uint64_t;
#endif
    using TypeValue = float;                    // faiss uses floats

    //! faiss index
    const char* index_key = "IVF4096,Flat";
#ifdef __ENABLE_FAISS__
    faiss::Index* index;
#else
    void *index;
#endif

    const uint8_t dim;
    const uint8_t knbrs;


    //! -----------------------------------------------------------------------
    //! cast an array into TypeValue
    //! -----------------------------------------------------------------------

    //! when  (data type = TypeValue)
    template <class T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    static inline
    TypeValue*
    _cast_to_typevalue(const size_t n, T *data) {
        return data;
    }

    //! when  (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    static inline
    TypeValue*
    _cast_to_typevalue(const size_t n, T *data) {
        TypeValue *fdata = new TypeValue[n];
        std::transform(data, data+n, fdata,
                       [&](const T& v) { return static_cast<TypeValue>(v); });
        return fdata;
    }

#ifdef __ENABLE_FAISS__
    //! -----------------------------------------------------------------------
    //! add points to cache
    //! -----------------------------------------------------------------------

    //! when  (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _add(const size_t ndata, const T *data) {
        index->add(ndata, data);
    }

    //! when  (data type != TypeValue)
    template <class T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _add(const size_t ndata, const T *data) {
        TypeValue *vdata = _cast_to_typevalue(ndata, data);
        index->add(ndata, vdata);
        delete [] vdata;
    }

    //! ------------------------------------------------------------------------
    //! train on data that comes already linearized
    //! ------------------------------------------------------------------------

    //! when  (data type = TypeValue)
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

    //! when  (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _train(const size_t ndata, const T *data) {
        if (index != nullptr && index->is_trained)
            throw std::invalid_argument("Trying to re-train an already trained index!");

        TypeValue *vdata = _cast_to_typevalue(ndata, data);
        index = faiss::index_factory(dim, index_key);
        index->train(ndata, vdata);
        delete [] vdata;

        if (!index->is_trained)
            throw std::runtime_error("Failed to train the index!");
    }


    //! ------------------------------------------------------------------------
    //! compute mean distance to k nearest neighbors
    //! ------------------------------------------------------------------------
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    inline
    void
    _mean_dist_to_knn(const size_t ndata, const T *data, const uint8_t k,
                      std::vector<TypeValue> &mean_dists) const {

        static const TypeValue ook = 1.0 / TypeValue(k);
        if (mean_dists.size() != ndata)
            mean_dists.resize(ndata);

        // create memory for the output
        std::vector<TypeIndex> kidxs (ndata*k);

        // if k = 1, this is the distance we need
        if (k == 1) {
            index->search(ndata, data, k, mean_dists.data(), kidxs.data());
        }

        else {
            std::vector<TypeValue> kdists (ndata*k);
            index->search(ndata, data, k, kdists.data(), kidxs.data());

            auto curr_beg = kdists.begin();
            for (size_t i = 0; i < ndata; i++) {
                mean_dists[i] = ook * std::accumulate(curr_beg, curr_beg+knbrs, 0.);
                curr_beg += k;
            }
        }
    }

    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline
    void
    _mean_dist_to_knn(const size_t ndata, const T *data, const uint8_t k,
                      std::vector<TypeValue> &mean_dists) const {

        TypeValue *vdata = _cast_to_typevalue(ndata, data);
        _mean_dist_to_knn(ndata, data, k, mean_dists);
        delete []  vdata;
    }
#endif

    //! -----------------------------------------------------------------------
    //! linearize a set of features (vector of pointers) into
    //! a single vector of TypeValue
    static inline
    std::vector<TypeValue>
    linearize_features(const size_t ndata, const std::vector<const TypeInValue *> &features) {

        // combine all features into a single linearized vector
        const size_t nfeatures = features.size();
        std::vector<TypeValue> data (ndata*nfeatures);
        for (size_t i = 0; i < ndata; i++) {
            for (size_t d = 0; d < nfeatures; d++) {
                data[i*nfeatures + d] = static_cast<TypeValue>(features[d][i]);
            }
        }
        return data;
    }


public:
    //! -----------------------------------------------------------------------
    HDCache(uint8_t _dim, uint8_t _knbrs = 10) :
        dim(_dim), knbrs(_knbrs), index(nullptr) {
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

    //! add the data that comes as separate features (a vector of pointers)
    void add(const size_t ndata, const std::vector<const TypeInValue *> &data) {
#ifdef __ENABLE_FAISS__
        std::cout << "Adding " << ndata << " " << data.size() << "-dim points!\n";

        if (data.size() != dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (!has_index())
            throw std::invalid_argument("Does not have a valid and trained index!");

        std::vector<TypeValue> lin_data = linearize_features(ndata, data);
        _add(ndata, lin_data.data());
        lin_data.clear();

        std::cout << "Successfully added! Cache has " << index->ntotal << " points!\n";
#else
        std::cerr << "HDCache::add() is a no-op without Faiss\n";
#endif
    }

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

    //! -----------------------------------------------------------------------
    //! train a cache
    //! -----------------------------------------------------------------------

    //! train on data that comes separate features (a vector of pointers)
    void train(const size_t ndata, const std::vector<const TypeInValue *> &data) {
#ifdef __ENABLE_FAISS__
        std::cout << "Training a " << int(dim) << "-dim cache "
                  << "using " << ndata << " " << data.size() << "-dim points!\n";

        if (data.size() != dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (has_index())
            throw std::invalid_argument("Already have a valid and trained index!");

        std::vector<TypeValue> lin_data = linearize_features(ndata, data);
        _train(ndata, lin_data.data());
        lin_data.clear();

        std::cout << "Successfully trained " << int(dim) << "-dim faiss index!\n";
#else
        std::cerr << "HDCache::train() is a no-op without Faiss\n";
#endif
    }

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


    //! -----------------------------------------------------------------------
    //! evaluate uncertainty using the cache (inputs are separate pointers)
    //! -----------------------------------------------------------------------
    //! this function works on the "inputs"
    //! to see how good are our error estimates in the input space
    //! i.e., we should call this *before* ML inference
    void Eval(const int length,
              const TypeInValue *density,
              const TypeInValue *energy,
              bool *is_acceptable)  const {

      static const TypeInValue acceptable_error = 0.5;

#ifdef __ENABLE_FAISS__
        // keep static. can we afford to hold the memory??
        static std::vector<TypeValue> mean_dists (length);

        std::vector<TypeValue> data = linearize_features(length, {density, energy});
        _mean_dist_to_knn(length, data.data(), knbrs, mean_dists);

        std::transform(mean_dists.begin(), mean_dists.end(), is_acceptable,
                       [](const TypeValue& v) { return v <= acceptable_error; });
#else

        for(int i = 0; i < length; i++) {
          is_acceptable[i] = ((TypeInValue)rand() / RAND_MAX) <= acceptable_error;
        }
#endif
   }


   //! this function can use both "inputs" and "outputs"
   //! to estimate uncertainity in either or both
   //! i.e., we should call this *after* ML inference
   void Eval(const int length,
             const TypeInValue *density,
             const TypeInValue *energy,
             TypeInValue *pressure,
             TypeInValue *soundspeed2,
             TypeInValue *bulkmod,
             bool *is_acceptable)  const {

       return Eval(length, density, energy, is_acceptable);
   }

    //! -----------------------------------------------------------------------
};

#endif
