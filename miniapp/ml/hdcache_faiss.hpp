#ifdef __ENABLE_FAISS__
#ifndef __AMS_HDCACHE_FAISS_HPP__
#define __AMS_HDCACHE_FAISS_HPP__

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include <faiss/index_io.h>
#include <faiss/index_factory.h>
#include <faiss/IndexFlat.h>

#ifdef __ENABLE_CUDA__
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

#include "hdcache.hpp"
#include "utils/utils_data.hpp"
#include "utils/allocator.hpp"

//! ----------------------------------------------------------------------------
//! An implementation of FAISS-based HDCache
//! ----------------------------------------------------------------------------
template <typename TypeInValue>
class HDCache_Faiss : public HDCache<TypeInValue> {

    using TypeIndex = faiss::Index::idx_t;          // 64-bit int
    using TypeValue = float;                        // faiss uses floats
    using data_handler = ams::DataHandler<TypeValue>;    // utils to handle float data

    faiss::Index *index;  // old
    const char* index_key = "IVF4096,Flat";
    //const char* index_key = "IndexFlatL2";
    //const char* index_key = "IndexFlatL2";
    //const char* index_key = "GpuIndexFlatL2";

    //faiss::gpu::StandardGpuResources resources;
    //faiss::gpu::GpuIndexIVFPQConfig config;
    //faiss::IndexIVFPQ* index_cpu;
    //faiss::gpu::GpuIndexIVFPQ *index_gpu;

public:
    HDCache_Faiss(uint8_t dim, uint8_t knbrs, bool use_device) :
        HDCache<TypeInValue>(dim, knbrs, use_device, std::string("faiss")) {
        index = nullptr;

        if (use_device) {
          throw std::invalid_argument("FAISS on device is not functional!");
        }
    }

    //todo: dim = 2 is wrong!
    HDCache_Faiss(const std::string &cache_path, uint8_t knbrs, bool use_device) :
        HDCache<TypeInValue>(2, knbrs, use_device, std::string("faiss")) {

        if (use_device) {
          throw std::invalid_argument("FAISS on device is not functional!");
        }
        load_cache(cache_path);
    }

    inline bool has_index() const { return index != nullptr && index->is_trained; }
    inline size_t count() const { return has_index() ? index->ntotal : 0;      }


    //! -----------------------------------------------------------------------
    //! load/save cache
    //! -----------------------------------------------------------------------
    inline void load_cache(const std::string &filename) {
        std::cout << "Loading HDCache from ("<<filename<<")\n";
        index = faiss::read_index(filename.c_str());
        this->print();
    }
    inline void save_cache(const std::string &filename) const {
        this->print();
        std::cout << "Saving to ("<<filename<<")\n";
        faiss::write_index(index, filename.c_str());
    }


    //! -----------------------------------------------------------------------
    //! add points to the cache
    //! -----------------------------------------------------------------------
    //! add the data that comes as linearized features
    void add(const size_t ndata, const size_t d, TypeInValue *data) {

        std::cout << "Adding " << ndata << " " << d << "-dim points!\n";

        if (d != this->m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (!has_index())
            throw std::invalid_argument("Does not have a valid and trained index!");

        _add(ndata, data);
        std::cout << "Successfully added!";
        this->print();
    }

    //! add the data that comes as separate features (a vector of pointers)
    void add(const size_t ndata, const std::vector<TypeInValue *> &inputs) {

        std::cout << "Adding " << ndata << " " << inputs.size() << "-dim points!\n";

        if (inputs.size() != this->m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (!has_index())
            throw std::invalid_argument("Does not have a valid and trained index!");

        TypeValue* lin_data = data_handler::linearize_features_hd(ndata, inputs);
        _add(ndata, lin_data);
        delete [] lin_data;

        std::cout << "Successfully added!";
        this->print();
    }


    //! -----------------------------------------------------------------------
    //! train a cache
    //! -----------------------------------------------------------------------
    //! train on data that comes as linearized features
    void train(const size_t ndata, const size_t d, TypeInValue *data) {
        std::cout << "Training a " << int(this->m_dim) << "-dim cache "
                  << "using " << ndata << " " << d << "-dim points!\n";

        if (d != this->m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (has_index())
            throw std::invalid_argument("Already have a valid and trained index!");

        _train(ndata, data);
        std::cout << "Successfully trained " << int(this->m_dim) << "-dim faiss index!\n";
    }

    //! train on data that comes separate features (a vector of pointers)
    void train(const size_t ndata, const std::vector<TypeInValue *> &inputs) {
        std::cout << "Training a " << int(this->m_dim) << "-dim cache "
                  << "using " << ndata << " " << inputs.size() << "-dim points!\n";

        if (inputs.size() != this->m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        if (has_index())
            throw std::invalid_argument("Already have a valid and trained index!");

        TypeValue* lin_data = data_handler::linearize_features_hd(ndata, inputs);
        _train(ndata, lin_data);
        delete [] lin_data;

        std::cout << "Successfully trained " << int(this->m_dim) << "-dim faiss index!\n";
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

        if (!this->has_index()) {
            throw std::invalid_argument("HDCache does not have a valid index!");
        }
        static const TypeInValue acceptable_error = 0.5;

        std::cout << "Evaluating a " << int(this->m_dim) << "-dim cache "
                  << "using " << ndata << " " << d << "-dim points!\n";

        if (d != this->m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        _evaluate(ndata, data, is_acceptable);
        std::cout << "Successfully evaluated " << ndata << " " << d << "-dim points!\n";
    }

    //! train on data that comes separate features (a vector of pointers)
    void evaluate(const size_t ndata, const std::vector<TypeInValue*> &inputs,
                  bool *is_acceptable) const {

        if (!this->has_index()) {
            throw std::invalid_argument("HDCache does not have a valid index!");
        }
        static const TypeInValue acceptable_error = 0.5;

        std::cout << "Evaluating a " << int(this->m_dim) << "-dim cache "
                  << "using " << ndata << " " << inputs.size() << "-dim points!\n";

        if (inputs.size() != this->m_dim)
            throw std::invalid_argument("Mismatch in data dimensionality!");

        TypeValue* lin_data = data_handler::linearize_features_hd(ndata, inputs);
        _evaluate(ndata, lin_data, is_acceptable);
        //delete [] lin_data;
        std::cout << "Successfully evaluated " << ndata << " " << inputs.size() << "-dim points!\n";
    }

    //! -----------------------------------------------------------------------

private:

    //! ------------------------------------------------------------------------
    //! core faiss functionality. working with linearized TypeValue data
    //! ------------------------------------------------------------------------

    //! add points to index when  (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    inline void
    _add(const size_t ndata, const T *data) {
        index->add(ndata, data);
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

        if (index != nullptr && index->is_trained)
            throw std::invalid_argument("Trying to re-train an already trained index!");

        index = faiss::index_factory(this->m_dim, index_key);
        index->train(ndata, data);

        if (!index->is_trained)
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


    //! evaluate cache uncertainty when  (data type = TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    void
    _evaluate(const size_t ndata, T *data, bool *is_acceptable) const {

        const size_t knbrs = static_cast<size_t>(this->m_knbrs);
        static const TypeValue acceptable_error = 0.5;
        static const TypeValue ook = 1.0 / TypeValue(knbrs);

        const bool data_on_device = ams::ResourceManager::is_on_device(data);

        // index on host
        if (!this->m_use_device) {

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
            index->search(ndata, h_data, knbrs, kdists, kidxs);

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
            std::cerr << "   - evaluating on device is not implemented\n";
            exit(1);
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

  // ---------------------------------------------------------------------------
};
#endif
#endif
