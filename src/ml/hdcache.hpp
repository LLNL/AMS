// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __AMS_HDCACHE_HPP__
#define __AMS_HDCACHE_HPP__

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __ENABLE_FAISS__
#include <faiss/IndexFlat.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#ifdef __ENABLE_CUDA__
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif
#endif

#include "AMS.h"
#include "wf/data_handler.hpp"
#include "wf/resource_manager.hpp"

//! ----------------------------------------------------------------------------
//! An implementation of FAISS-based HDCache
//! ----------------------------------------------------------------------------
template <typename TypeInValue>
class HDCache
{

  static_assert(std::is_floating_point<TypeInValue>::value,
                "HDCache supports floating-point values (floats, doubles, and "
                "long doubles) only!");

#ifdef __ENABLE_FAISS__
  using Index = faiss::Index;
  using TypeIndex = faiss::Index::idx_t;  // 64-bit int
  using TypeValue = float;                // faiss uses floats
#ifdef __ENABLE_CUDA__
  faiss::gpu::StandardGpuResources res;
  faiss::gpu::GpuClonerOptions copyOptions;
#endif
#else
  using Index = void;
  using TypeIndex = uint64_t;
  using TypeValue = TypeInValue;
#endif
  using data_handler =
      ams::DataHandler<TypeValue>;  // utils to handle float data

  Index *m_index = nullptr;
  const uint8_t m_dim;

  const bool m_use_random;
  const bool m_use_device;
  const int m_knbrs = 0;
  const AMSUQPolicy m_policy = AMSUQPolicy::FAISSMean;

  AMSResourceType defaultRes;

  const TypeValue acceptable_error;


#ifdef __ENABLE_FAISS__
  const char *index_key = "IVF4096,Flat";
  // const char* index_key = "IndexFlatL2";
  // const char* index_key = "IndexFlatL2";
  // const char* index_key = "GpuIndexFlatL2";

  // faiss::gpu::StandardGpuResources resources;
  // faiss::gpu::GpuIndexIVFPQConfig config;
  // faiss::IndexIVFPQ* index_cpu;
  // faiss::gpu::GpuIndexIVFPQ *index_gpu;
#endif

public:
  //! ------------------------------------------------------------------------
  //! constructors
  //! ------------------------------------------------------------------------
  HDCache(bool use_device,
          TypeInValue threshold = 0.5)
      : m_index(nullptr),
        m_dim(0),
        m_use_random(true),
        m_knbrs(-1),
        m_use_device(use_device),
        acceptable_error(threshold)
  {
    defaultRes =
        (m_use_device) ? AMSResourceType::DEVICE : AMSResourceType::HOST;
    print();
  }

#ifdef __ENABLE_FAISS__
  HDCache(const std::string &cache_path,
          bool use_device,
          const AMSUQPolicy uqPolicy,
          int knbrs,
          TypeInValue threshold = 0.5)
      : m_index(load_cache(cache_path)),
        m_dim(m_index->d),
        m_use_random(false),
        m_knbrs(knbrs),
        m_policy(uqPolicy),
        m_use_device(use_device),
        acceptable_error(threshold)
  {
    defaultRes =
        (m_use_device) ? AMSResourceType::DEVICE : AMSResourceType::HOST;
#ifdef __ENABLE_CUDA__
    // Copy index to device side
    if (use_device) {
      faiss::gpu::GpuClonerOptions copyOptions;
      faiss::gpu::ToGpuCloner cloner(&res, 0, copyOptions);
      m_index = cloner.clone_Index(m_index);
    }
#endif
    print();
  }
#else
  HDCache(const std::string &cache_path,
          int knbrs,
          bool use_device,
          TypeInValue threshold = 0.5)
      : m_index(nullptr),
        m_dim(0),
        m_use_random(true),
        m_knbrs(knbrs),
        m_use_device(use_device),
        acceptable_error(threshold)
  {
    defaultRes =
        (m_use_device) ? AMSResourceType::DEVICE : AMSResourceType::HOST;
    WARNING(UQModule, "Ignoring cache path because FAISS is not available")
    print();
  }
#endif

  //! ------------------------------------------------------------------------
  //! simple queries
  //! ------------------------------------------------------------------------
  inline void print() const
  {
    std::string info("index = null");
    if ( has_index() ) {
      info =  "npoints = " + std::to_string(count());
    }
    DBG(UQModule, "HDCache (on_device = %d random = %d %s)",
        m_use_device, m_use_random, info.c_str());
  }

  inline bool has_index() const
  {
#ifdef __ENABLE_FAISS__
    if (!m_use_random) return m_index != nullptr && m_index->is_trained;
#endif
    return true;
  }

  inline size_t count() const
  {
#ifdef __ENABLE_FAISS__
    if (!m_use_random) return m_index->ntotal;
#endif
    return 0;
  }

  inline uint8_t dim() const { return m_dim; }

  //! ------------------------------------------------------------------------
  //! load/save faiss cache
  //! ------------------------------------------------------------------------
  static inline Index *load_cache(const std::string &filename)
  {
#ifdef __ENABLE_FAISS__
    DBG(UQModule, "Loading HDCache: %s", filename.c_str());
    return faiss::read_index(filename.c_str());
#else
    return nullptr;
#endif
  }

  inline void save_cache(const std::string &filename) const
  {
#ifdef __ENABLE_FAISS__
    print();
    DBG(UQModule, "Saving HDCache to: %s", filename.c_str());
    faiss::write_index(m_index, filename.c_str());
#endif
  }

  //! -----------------------------------------------------------------------
  //! add points to the faiss cache
  //! -----------------------------------------------------------------------
  //! add the data that comes as linearized features
PERFFASPECT()
  void add(const size_t ndata, const size_t d, TypeInValue *data)
  {
    if (m_use_random) return;

    DBG(UQModule, "Add %ld %ld points to HDCache", ndata, d);
    CFATAL(UQModule, d != m_dim, "Mismatch in data dimensionality!")
    CFATAL(UQModule, !has_index(), "HDCache does not have a valid and trained index!")

    _add(ndata, data);
  }

  //! add the data that comes as separate features (a vector of pointers)
PERFFASPECT()
  void add(const size_t ndata, const std::vector<TypeInValue *> &inputs)
  {
    if (m_use_random) return;

    if (inputs.size() != m_dim)
    CFATAL(UQModule, inputs.size() != m_dim, "Mismatch in data dimensionality")
    CFATAL(UQModule, !has_index(), "HDCache does not have a valid and trained index!")

    TypeValue *lin_data = data_handler::linearize_features(ndata, inputs);
    _add(ndata, lin_data);
    ams::ResourceManager::deallocate(lin_data);
  }

  //! -----------------------------------------------------------------------
  //! train a faiss cache
  //! -----------------------------------------------------------------------
  //! train on data that comes as linearized features
PERFFASPECT()
  void train(const size_t ndata, const size_t d, TypeInValue *data)
  {
    if (m_use_random) return;
    DBG(UQModule, "Add %ld %ld points to HDCache", ndata, d);
    CFATAL(UQModule, d != m_dim, "Mismatch in data dimensionality!")
    CFATAL(UQModule, !has_index(), "HDCache does not have a valid and trained index!")

    _train(ndata, data);
    DBG(UQModule, "Successfully Trained HDCache");
  }

  //! train on data that comes separate features (a vector of pointers)
PERFFASPECT()
  void train(const size_t ndata, const std::vector<TypeInValue *> &inputs)
  {
    if (m_use_random) return;
    TypeValue *lin_data = data_handler::linearize_features(ndata, inputs);
    _train(ndata, lin_data);
    ams::ResourceManager::deallocate(lin_data);
  }

  //! ------------------------------------------------------------------------
  //! evaluate uncertainty using the cache
  //! ------------------------------------------------------------------------
  //! train on data that comes as linearized features
  //! it looks like faiss can work directly on torch tensor
  //! https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#passing-in-pytorch-tensors
  //! so, we should use Dino's code to linearize data into torch tensor and then
  //! pass it here
PERFFASPECT()
  void evaluate(const size_t ndata,
                const size_t d,
                TypeInValue *data,
                bool *is_acceptable) const
  {

    CFATAL(UQModule, !has_index(), "HDCache does not have a valid and trained index!")
    DBG(UQModule, "Evaluating %ld %ld points using HDCache", ndata, d);

    CFATAL(UQModule, (!m_use_random) && (d != m_dim), "Mismatch in data dimensionality!")

    if (m_use_random) {
      _evaluate(ndata, is_acceptable);
    } else {
      _evaluate(ndata, data, is_acceptable);
    }

  }

  //! train on data that comes separate features (a vector of pointers)
PERFFASPECT()
  void evaluate(const size_t ndata,
                const std::vector<const TypeInValue *> &inputs,
                bool *is_acceptable) const
  {

    CFATAL(UQModule, !has_index(), "HDCache does not have a valid and trained index!")
    DBG(UQModule, "Evaluating %ld %ld points using HDCache configured with %d neighbors, %f threshold, %d policy",
        ndata, inputs.size(), m_knbrs, acceptable_error, m_policy);
    CFATAL(UQModule, ((!m_use_random) && inputs.size() != m_dim), "Mismatch in data dimensionality!")

    if (m_use_random) {
      _evaluate(ndata, is_acceptable);
    } else {
      TypeValue *lin_data = data_handler::linearize_features(ndata, inputs);
      _evaluate(ndata, lin_data, is_acceptable);
      ams::ResourceManager::deallocate(lin_data);
    }
  }

private:
#ifdef __ENABLE_FAISS__
  //! ------------------------------------------------------------------------
  //! core faiss functionality.
  //! ------------------------------------------------------------------------

  inline uint8_t _dim() const { return (m_index != nullptr) ? m_index->d : 0; }

  //! add points to index when  (data type = TypeValue)
  template <typename T,
            std::enable_if_t<std::is_same<TypeValue, T>::value> * = nullptr>
PERFFASPECT()
  inline void _add(const size_t ndata, const T *data)
  {
    m_index->add(ndata, data);
  }

  //! add points to index when (data type != TypeValue)
  template <typename T,
            std::enable_if_t<!std::is_same<TypeValue, T>::value> * = nullptr>
PERFFASPECT()
  inline void _add(const size_t ndata, const T *data)
  {
    TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
    _add(ndata, vdata);
    delete[] vdata;
  }


  //! train an index when (data type = TypeValue)
  template <typename T,
            std::enable_if_t<std::is_same<TypeValue, T>::value> * = nullptr>
PERFFASPECT()
  inline void _train(const size_t ndata, const T *data)
  {

    if (m_index != nullptr && m_index->is_trained)
      throw std::invalid_argument(
          "!");

    CFATAL(UQModule,
        (m_index != nullptr && m_index->is_trained),
        "Trying to re-train an already trained index")

    m_index = faiss::index_factory(m_dim, index_key);
    m_index->train(ndata, data);

    CFATAL(UQModule,
        ((!m_index->is_trained)),
        "Failed to train index")
  }

  //! train an index when (data type != TypeValue)
  template <typename T,
            std::enable_if_t<!std::is_same<TypeValue, T>::value> * = nullptr>
PERFFASPECT()
  inline void _train(const size_t ndata, const T *data)
  {
    TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
    _train(ndata, vdata);
    delete[] vdata;
  }

  // -------------------------------------------------------------------------
  //! evaluate cache uncertainty when  (data type = TypeValue)
  template <typename T,
            std::enable_if_t<std::is_same<TypeValue, T>::value> * = nullptr>
PERFFASPECT()
  void _evaluate(const size_t ndata, T *data, bool *is_acceptable) const
  {

    const size_t knbrs = static_cast<size_t>(m_knbrs);
    static const TypeValue ook = 1.0 / TypeValue(knbrs);

    const bool input_on_device = ams::ResourceManager::is_on_device(data);
    const bool output_on_device =
        ams::ResourceManager::is_on_device(is_acceptable);

    if (input_on_device != output_on_device) {
      WARNING(UQModule, "Input is ( on_device: %d)"
                        " Output is ( on_device: %d)"
                        " on different devices",
                        input_on_device, output_on_device)
    }

    TypeValue *kdists =
        ams::ResourceManager::allocate<TypeValue>(ndata * knbrs, defaultRes);
    TypeIndex *kidxs =
        ams::ResourceManager::allocate<TypeIndex>(ndata * knbrs, defaultRes);

    // query faiss
    // TODO: This is a HACK. When searching more than 65535
    // items in the GPU case, faiss is throwing an exception.
    const unsigned int MAGIC_NUMBER = 65535;
    for (int start = 0; start < ndata; start += MAGIC_NUMBER) {
      unsigned int nElems =
          ((ndata - start) < MAGIC_NUMBER) ? ndata - start : MAGIC_NUMBER;
      m_index->search(
          nElems, &data[start], knbrs, &kdists[start*knbrs], &kidxs[start*knbrs]);
    }

    // compute means
    if (defaultRes == AMSResourceType::HOST) {
      TypeValue total_dist = 0;
      for (size_t i = 0; i < ndata; ++i) {
        CFATAL(UQModule, m_policy==AMSUQPolicy::DeltaUQ, "DeltaUQ is not supported yet");
        if ( m_policy == AMSUQPolicy::FAISSMean ) {
          total_dist =
              std::accumulate(kdists + i * knbrs, kdists + (i + 1) * knbrs, 0.);
          is_acceptable[i] = (ook * total_dist) < acceptable_error;
        }
        else if ( m_policy == AMSUQPolicy::FAISSMax ) {
          // Take the furtherst cluster as the distance metric
          total_dist = kdists[i*knbrs + knbrs -1];
          is_acceptable[i] = (total_dist) < acceptable_error;
        }
      }
    } else {
      CFATAL(UQModule, (m_policy==AMSUQPolicy::DeltaUQ) || (m_policy==AMSUQPolicy::FAISSMax),
          "DeltaUQ is not supported yet");

      ams::Device::computePredicate(
          kdists, is_acceptable, ndata, knbrs, acceptable_error);
    }

    ams::ResourceManager::deallocate(kdists);
    ams::ResourceManager::deallocate(kidxs);
  }

  //! evaluate cache uncertainty when (data type != TypeValue)
  template <typename T,
            std::enable_if_t<!std::is_same<TypeValue, T>::value> * = nullptr>
  inline void _evaluate(const size_t ndata, T *data, bool *is_acceptable) const
  {
    TypeValue *vdata = data_handler::cast_to_typevalue(ndata, data);
    _evaluate(ndata, data, is_acceptable);
    delete[] vdata;
  }

#else
  // -------------------------------------------------------------------------
  // fucntionality for randomized cache
  // -------------------------------------------------------------------------
  inline uint8_t _dim() const { return 0; }

  template <typename T>
PERFFASPECT()
  inline void _add(const size_t, const T *)
  {
  }

  template <typename T>
PERFFASPECT()
  inline void _train(const size_t, const T *)
  {
  }

  template <typename T>
PERFFASPECT()
  inline void _evaluate(const size_t, T *, bool *) const
  {
  }
#endif
PERFFASPECT()
  inline void _evaluate(const size_t ndata, bool *is_acceptable) const
  {
    const bool data_on_device =
        ams::ResourceManager::is_on_device(is_acceptable);

    if (data_on_device) {
#ifdef __ENABLE_CUDA__
      random_uq_device<<<1, 1>>>(is_acceptable, ndata, acceptable_error);
#endif
    } else {
      random_uq_host(is_acceptable, ndata, acceptable_error);
    }
  }
  // -------------------------------------------------------------------------
};
#endif
