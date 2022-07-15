#ifndef __AMS_HDCACHE_HPP__
#define __AMS_HDCACHE_HPP__

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
  #include <faiss/IndexFlat.h>

  #ifdef __ENABLE_CUDA__
    #include <faiss/gpu/GpuAutoTune.h>
    #include <faiss/gpu/GpuCloner.h>
    #include <faiss/gpu/GpuIndexIVFPQ.h>
    #include <faiss/gpu/StandardGpuResources.h>
  #endif
#endif

#include "wf/resource_manager.hpp"
#include "wf/data_handler.hpp"

//! ----------------------------------------------------------------------------
//! An implementation of FAISS-based HDCache
//! ----------------------------------------------------------------------------
template <typename TypeInValue>
class HDCache {

    static_assert (std::is_floating_point<TypeInValue>::value,
                  "HDCache supports floating-point values (floats, doubles, and long doubles) only!");

#ifdef __ENABLE_FAISS__
    using Index = faiss::Index;
    using TypeIndex = faiss::Index::idx_t;            // 64-bit int
    using TypeValue = float;                          // faiss uses floats
#else
    using Index = void;
    using TypeIndex = uint64_t;
    using TypeValue = TypeInValue;
#endif
    using data_handler = ams::DataHandler<TypeValue>; // utils to handle float data

    Index *m_index = nullptr;
    const uint8_t m_dim;

    const bool m_use_random;
    const bool m_use_device;
    const uint8_t m_knbrs;


#ifdef __ENABLE_FAISS__
    const char* index_key = "IVF4096,Flat";
    //const char* index_key = "IndexFlatL2";
    //const char* index_key = "IndexFlatL2";
    //const char* index_key = "GpuIndexFlatL2";

    //faiss::gpu::StandardGpuResources resources;
    //faiss::gpu::GpuIndexIVFPQConfig config;
    //faiss::IndexIVFPQ* index_cpu;
    //faiss::gpu::GpuIndexIVFPQ *index_gpu;
#endif

public:
    //! ------------------------------------------------------------------------
    //! constructors
    //! ------------------------------------------------------------------------
    HDCache(uint8_t dim, uint8_t knbrs, bool use_device) :
        m_index(nullptr), m_dim(dim), m_use_random(true),
        m_knbrs(knbrs), m_use_device(use_device) {
        if (use_device) {
            throw std::invalid_argument("HDCache is not functional on device!");
        }
        print();
    }

#ifdef __ENABLE_FAISS__
    HDCache(const std::string &cache_path, uint8_t knbrs, bool use_device) :
        m_index(load_cache(cache_path)), m_dim(m_index->d), m_use_random(false),
        m_knbrs(knbrs), m_use_device(use_device) {
        if (use_device) {
            throw std::invalid_argument("HDCache is not functional on device!");
        }
        print();
    }
#else
    HDCache(const std::string &cache_path, uint8_t knbrs, bool use_device) :
        m_index(nullptr), m_dim(0), m_use_random(true),
        m_knbrs(knbrs), m_use_device(use_device) {
        if (use_device) {
            throw std::invalid_argument("HDCache is not functional on device!");
        }
        std::cerr << "WARNING: Ignoring cache path because FAISS is not available!\n";
        print();
    }
#endif

    //! ------------------------------------------------------------------------
    //! simple queries
    //! ------------------------------------------------------------------------
    inline void
    print() const {
        std::cout << "HDCache (on_device = "<< m_use_device<< ", "
                  << "random = " << m_use_random << ", ";
        if (has_index()) { std::cout << "npoints = " << count();  }
        else {             std::cout << "index = null";           }
        std::cout << ")\n";
    }

    inline bool
    has_index() const {
#ifdef __ENABLE_FAISS__
        if (!m_use_random)
            return m_index != nullptr && m_index->is_trained;
#endif
        return true;
    }

    inline size_t
    count() const {
#ifdef __ENABLE_FAISS__
        if (!m_use_random)
            return m_index->ntotal;
#endif
        return 0;
    }

    inline uint8_t
    dim() const {
        return m_dim;
    }

    //! ------------------------------------------------------------------------
    //! load/save faiss cache
    //! ------------------------------------------------------------------------
    static inline
    Index*
    load_cache(const std::string &filename) {
#ifdef __ENABLE_FAISS__
        std::cout << "Loading HDCache from ("<<filename<<")\n";
        return faiss::read_index(filename.c_str());
#else
        return nullptr;
#endif
    }

    inline void
    save_cache(const std::string &filename) const {
#ifdef __ENABLE_FAISS__
        print();
        std::cout << "Saving to ("<<filename<<")\n";
        faiss::write_index(m_index, filename.c_str());
#endif
    }

    //! -----------------------------------------------------------------------
    //! add points to the faiss cache
    //! -----------------------------------------------------------------------
    //! add the data that comes as linearized features
    void add(const size_t ndata, const size_t d, TypeInValue *data) {
        if (m_use_random)
            return;

        std::cout << "Adding " << ndata << " " << d << "-dim points!\n";
        if (d != m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (!has_index())
            throw std::invalid_argument("HDCache does not have a valid and trained index!");

        _add(ndata, data);
        std::cout << "Successfully added!";
        print();
    }

    //! add the data that comes as separate features (a vector of pointers)
    void add(const size_t ndata, const std::vector<TypeInValue *> &inputs) {
        if (m_use_random)
            return;

        std::cout << "Adding " << ndata << " " << inputs.size() << "-dim points!\n";

        if (inputs.size() != m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (!has_index())
            throw std::invalid_argument("HDCache does not have a valid and trained index!");

        TypeValue* lin_data = data_handler::linearize_features_hd(ndata, inputs);
        _add(ndata, lin_data);
        delete [] lin_data;

        std::cout << "Successfully added!";
        print();
    }

    //! -----------------------------------------------------------------------
    //! train a faiss cache
    //! -----------------------------------------------------------------------
    //! train on data that comes as linearized features
    void train(const size_t ndata, const size_t d, TypeInValue *data) {
        if (m_use_random)
            return;

        std::cout << "Training a " << int(m_dim) << "-dim cache "
                  << "using " << ndata << " " << d << "-dim points!\n";

        if (d != m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (has_index())
            throw std::invalid_argument("Already have a valid and trained index!");

        _train(ndata, data);
        std::cout << "Successfully trained " << int(m_dim) << "-dim faiss index!\n";
    }

    //! train on data that comes separate features (a vector of pointers)
    void train(const size_t ndata, const std::vector<TypeInValue *> &inputs) {
        if (m_use_random)
            return;

        std::cout << "Training a " << int(m_dim) << "-dim cache "
                  << "using " << ndata << " " << inputs.size() << "-dim points!\n";

        if (inputs.size() != m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (has_index())
            throw std::invalid_argument("Already have a valid and trained index!");

        TypeValue* lin_data = data_handler::linearize_features_hd(ndata, inputs);
        _train(ndata, lin_data);
        delete [] lin_data;

        std::cout << "Successfully trained " << int(m_dim) << "-dim faiss index!\n";
    }

    //! ------------------------------------------------------------------------
    //! evaluate uncertainty using the cache
    //! ------------------------------------------------------------------------
    //! train on data that comes as linearized features
    //! it looks like faiss can work directly on torch tensor
    //! https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#passing-in-pytorch-tensors
    //! so, we should use Dino's code to linearize data into torch tensor and then pass it here
    void evaluate(const size_t ndata, const size_t d, TypeInValue *data,
                  bool *is_acceptable) const {

        if (!has_index()) {
            throw std::invalid_argument("HDCache does not have a valid and trained index!");
        }

        std::cout << "Evaluating a " << int(m_dim) << "-dim cache "
                  << "for " << ndata << " " << d << "-dim points!\n";

        if (d != m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (m_use_random) {
            _evaluate(ndata, is_acceptable);
        }
        else {
            _evaluate(ndata, data, is_acceptable);
        }

        std::cout << "Successfully evaluated " << ndata << " " << int(m_dim) << "-dim points!\n";
    }

    //! train on data that comes separate features (a vector of pointers)
    void evaluate(const size_t ndata, const std::vector<const TypeInValue*> &inputs,
                  bool *is_acceptable) const {

        if (!has_index()) {
            throw std::invalid_argument("HDCache does not have a valid and trained index!");
        }

        std::cout << "Evaluating a " << int(m_dim) << "-dim cache "
                  << "for " << ndata << " " << inputs.size() << "-dim points!\n";

        if (inputs.size() != m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (m_use_random) {
            _evaluate(ndata, is_acceptable);
        }
        else {
            TypeValue* lin_data = data_handler::linearize_features(ndata, inputs);
            _evaluate(ndata, lin_data, is_acceptable);
            ams::ResourceManager::deallocate(lin_data);
        }

        std::cout << "Successfully evaluated " << ndata << " " << int(m_dim) << "-dim points!\n";
    }

    //! -----------------------------------------------------------------------

private:

#ifdef __ENABLE_FAISS__
    //! ------------------------------------------------------------------------
    //! core faiss functionality.
    //! ------------------------------------------------------------------------

    inline uint8_t
    _dim() const {
        return  (m_index != nullptr) ? m_index->d : 0;
    }

    //! add points to index when  (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _add(const size_t ndata, const T *data) {
        m_index->add(ndata, data);
    }

    //! add points to index when (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _add(const size_t ndata, const T *data) {
        TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
        _add(ndata, vdata);
        delete [] vdata;
    }


    //! train an index when (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _train(const size_t ndata, const T *data) {

        if (m_index != nullptr && m_index->is_trained)
            throw std::invalid_argument("Trying to re-train an already trained index!");

        m_index = faiss::index_factory(m_dim, index_key);
        m_index->train(ndata, data);

        if (!m_index->is_trained)
            throw std::runtime_error("Failed to train the index!");
    }

    //! train an index when (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _train(const size_t ndata, const T *data) {
        TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
        _train(ndata, vdata);
        delete []  vdata;
    }


    // -------------------------------------------------------------------------
    //! evaluate cache uncertainty when  (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    void
    _evaluate(const size_t ndata, T *data, bool *is_acceptable) const {

        const size_t knbrs = static_cast<size_t>(m_knbrs);
        static const TypeValue acceptable_error = 0.5;
        static const TypeValue ook = 1.0 / TypeValue(knbrs);

        const bool data_on_device = ams::ResourceManager::is_on_device(data);

        // index on host
        if (!m_use_device) {

            TypeValue* kdists = new TypeValue[ndata*knbrs];
            TypeIndex* kidxs = new TypeIndex[ndata*knbrs];

            // host copies!
            bool *h_is_acceptable = nullptr;
            TypeValue *h_data = nullptr;

            if (data_on_device) {
                h_is_acceptable = new bool [ndata];

                h_data = new TypeValue[ndata];
                DtoHMemcpy(h_data, data, ndata*sizeof(TypeValue));
            }
            else {
                h_is_acceptable = is_acceptable;
                h_data = data;
            }

            // query faiss
            m_index->search(ndata, h_data, knbrs, kdists, kidxs);

            // compute means
            TypeValue total_dist = 0;
            for (size_t i = 0; i < ndata; ++i) {
                total_dist = std::accumulate(kdists + i*knbrs, kdists + (i+1)*knbrs, 0.);
                h_is_acceptable[i] = (ook * total_dist) < acceptable_error;
            }

            // move output back to device
            if (data_on_device) {
                HtoDMemcpy(is_acceptable, h_is_acceptable, ndata*sizeof(bool));
                delete [] h_is_acceptable;
                delete [] h_data;
            }
            delete [] kdists;
            delete [] kidxs;
        }
        else {
            // https://github.com/kyamagu/faiss-wheels/issues/54
            throw std::invalid_argument("FAISS evaluation on device is not implemented\n");
        }
    }

    //! evaluate cache uncertainty when (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _evaluate(const size_t ndata, T *data, bool *is_acceptable)  const {
        TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
        _evaluate(ndata, data, is_acceptable);
        delete []  vdata;
    }

#else
    // -------------------------------------------------------------------------
    // fucntionality for randomized cache
    // -------------------------------------------------------------------------
    inline uint8_t _dim() const { return 0; }

    template<typename T>
    inline void _add(const size_t, const T *) {}

    template<typename T>
    inline void _train(const size_t, const T *) {}

    template<typename T>
    inline void _evaluate(const size_t, T *, bool*) const {}
#endif

    inline void
    _evaluate(const size_t ndata, bool *is_acceptable) const {

        static const TypeInValue acceptable_error = 0.5;
        const bool data_on_device = ams::ResourceManager::is_on_device(is_acceptable);

#if 1
        if (data_on_device) {
#ifdef USE_CUDA
            random_uq_device<<<1,1>>>(is_acceptable, ndata, acceptable_error);
#endif
        }
        else {
            random_uq_host(is_acceptable, ndata, acceptable_error);
        }
#else
        // use host for computation
        if (!m_use_device) {
            bool *h_is_acceptable = nullptr;
            if (data_on_device) {
                h_is_acceptable = ams::ResourceManager::allocate<bool>(ndata, ams::ResourceManager::ResourceType::HOST);
            }
            else {
                h_is_acceptable = is_acceptable;
            }

            random_uq_host(h_is_acceptable, ndata, acceptable_error);
            if (data_on_device) {
              HtoDMemcpy(is_acceptable, h_is_acceptable, ndata*sizeof(bool));
              ams::ResourceManager::deallocate(h_is_acceptable);
            }
        }

        // compute random flags directly on host
        else {
          throw std::invalid_argument("HDCache is not functional on device!");
        }
#endif
    }
    // -------------------------------------------------------------------------
};
#endif
