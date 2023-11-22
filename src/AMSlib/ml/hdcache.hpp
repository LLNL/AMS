/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_HDCACHE_HPP__
#define __AMS_HDCACHE_HPP__

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
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
#include "wf/utils.hpp"

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
  const int m_knbrs = 0;
  const AMSUQPolicy m_policy = AMSUQPolicy::FAISS_Mean;

  AMSResourceType cache_location;

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

protected:
  // A mechanism to keep track of all unique HDCaches
  static std::unordered_map<std::string, std::shared_ptr<HDCache<TypeInValue>>>
      instances;

  //! ------------------------------------------------------------------------
  //! constructors
  //! ------------------------------------------------------------------------
  HDCache(AMSResourceType resource, TypeInValue threshold = 0.5)
      : m_index(nullptr),
        m_dim(0),
        m_use_random(true),
        m_knbrs(-1),
        cache_location(resource),
        acceptable_error(threshold)
  {
    print();
  }

#ifdef __ENABLE_FAISS__
  HDCache(const std::string &cache_path,
          AMSResourceType resource,
          const AMSUQPolicy uqPolicy,
          int knbrs,
          TypeInValue threshold = 0.5)
      : m_index(load_cache(cache_path)),
        m_dim(m_index->d),
        m_use_random(false),
        m_knbrs(knbrs),
        m_policy(uqPolicy),
        cache_location(resource),
        acceptable_error(threshold)
  {
#ifdef __ENABLE_CUDA__
    // Copy index to device side
    if (cache_location == AMSResourceType::DEVICE) {
      faiss::gpu::GpuClonerOptions copyOptions;
      faiss::gpu::ToGpuCloner cloner(&res, 0, copyOptions);
      m_index = cloner.clone_Index(m_index);
    }
#endif
    print();
  }
#else  // Disabled FAISS
  HDCache(const std::string &cache_path,
          AMSResourceType resource,
          const AMSUQPolicy uqPolicy,
          int knbrs,
          TypeInValue threshold = 0.5)
      : m_index(load_cache(cache_path)),
        m_dim(0),
        m_use_random(false),
        m_knbrs(knbrs),
        m_policy(uqPolicy),
        cache_location(resource),
        acceptable_error(threshold)
  {
    WARNING(UQModule, "Ignoring cache path because FAISS is not available")
    print();
  }
#endif

public:
  static std::shared_ptr<HDCache<TypeInValue>> find_cache(
      const std::string &cache_path,
      AMSResourceType resource,
      const AMSUQPolicy uqPolicy,
      int knbrs,
      TypeInValue threshold = 0.5)
  {
    auto model = HDCache<TypeInValue>::instances.find(cache_path);

    if (model != instances.end()) {
      // Model Found
      auto cache = model->second;
      if (resource != cache->cache_location)
        throw std::runtime_error(
            "Currently we do not support loading the same index on different "
            "devices.");

      if (uqPolicy != cache->m_policy)
        throw std::runtime_error(
            "We do not support caches of different policies.");

      if (knbrs != cache->m_knbrs)
        throw std::runtime_error(
            "We do not support caches of different number of neighbors.");

      // FIXME: Here we need to cast both to float. FAISS index only works for
      // single precision and we shoehorn FAISS inability to support arbitary real
      // types by forcing TypeValue to be 'float'. In our case this results in having
      // cases where input data are of type(TypeInValue) double. Thus here, threshold can
      // be of different type than 'acceptable_error' and at compile time we cannot decide
      // which overloaded function to pick.
      if (!is_real_equal(static_cast<float>(threshold),
                         static_cast<float>(cache->acceptable_error)))
        throw std::runtime_error(
            "We do not support caches of different thresholds");

      return cache;
    }
    return nullptr;
  }

  static std::shared_ptr<HDCache<TypeInValue>> getInstance(
      const std::string &cache_path,
      AMSResourceType resource,
      const AMSUQPolicy uqPolicy,
      int knbrs,
      TypeInValue threshold = 0.5)
  {

    // Cache does not exist. We need to create one
    //
    std::shared_ptr<HDCache<TypeInValue>> cache =
        find_cache(cache_path, resource, uqPolicy, knbrs, threshold);
    if (cache) {
      DBG(UQModule, "Returning existing cache under (%s)", cache_path.c_str())
      return cache;
    }

    if (uqPolicy != AMSUQPolicy::FAISS_Mean &&
        uqPolicy != AMSUQPolicy::FAISS_Max)
      THROW(std::invalid_argument,
            "Invalid UQ policy for hdcache" + std::to_string(uqPolicy));

    DBG(UQModule, "Generating new cache under (%s)", cache_path.c_str())
    std::shared_ptr<HDCache<TypeInValue>> new_cache =
        std::shared_ptr<HDCache<TypeInValue>>(new HDCache<TypeInValue>(
            cache_path, resource, uqPolicy, knbrs, threshold));

    instances.insert(std::make_pair(cache_path, new_cache));
    return new_cache;
  }

  static std::shared_ptr<HDCache<TypeInValue>> getInstance(
      AMSResourceType resource,
      float threshold = 0.5)
  {
    static std::string random_path("random");
    std::shared_ptr<HDCache<TypeInValue>> cache = find_cache(
        random_path, resource, AMSUQPolicy::FAISS_Mean, -1, threshold);
    if (cache) {
      DBG(UQModule, "Returning existing cache under (%s)", random_path.c_str())
      return cache;
    }

    DBG(UQModule,
        "Generating new cache under (%s, threshold:%f)",
        random_path.c_str(),
        threshold)
    std::shared_ptr<HDCache<TypeInValue>> new_cache =
        std::shared_ptr<HDCache<TypeInValue>>(
            new HDCache<TypeInValue>(resource, threshold));

    instances.insert(std::make_pair(random_path, new_cache));
    return new_cache;
  }

  ~HDCache()
  {
    DBG(UQModule, "Deleting UQ-Module");
#ifdef __ENABLE_FAISS__
    if (m_index) {
      DBG(UQModule, "Deleting HD-Cache");
      /// TODO: Deleting the cache on device can, and does
      /// result in C++ destructor.
      if (cache_location != AMSResourceType::DEVICE) {
        m_index->reset();
        delete m_index;
      }
    }
#endif
  }

  //! ------------------------------------------------------------------------
  //! simple queries
  //! ------------------------------------------------------------------------
  inline void print() const
  {
    std::string info("index = null");
    if (has_index()) {
      info = "npoints = " + std::to_string(count());
    }
    DBG(UQModule,
        "HDCache (on_device = %d random = %d %s)",
        cache_location,
        m_use_random,
        info.c_str());
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
    CFATAL(UQModule,
           !has_index(),
           "HDCache does not have a valid and trained index!")

    _add(ndata, data);
  }

  //! add the data that comes as separate features (a vector of pointers)
  PERFFASPECT()
  void add(const size_t ndata, const std::vector<TypeInValue *> &inputs)
  {
    if (m_use_random) return;

    if (inputs.size() != m_dim)
      CFATAL(UQModule,
             inputs.size() != m_dim,
             "Mismatch in data dimensionality")
    CFATAL(UQModule,
           !has_index(),
           "HDCache does not have a valid and trained index!")

    TypeValue *lin_data =
        data_handler::linearize_features(cache_location, ndata, inputs);
    _add(ndata, lin_data);
    ams::ResourceManager::deallocate(lin_data, cache_location);
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
    CFATAL(UQModule,
           !has_index(),
           "HDCache does not have a valid and trained index!")

    _train(ndata, data);
    DBG(UQModule, "Successfully Trained HDCache");
  }

  //! train on data that comes separate features (a vector of pointers)
  PERFFASPECT()
  void train(const size_t ndata, const std::vector<TypeInValue *> &inputs)
  {
    if (m_use_random) return;
    TypeValue *lin_data =
        data_handler::linearize_features(cache_location, ndata, inputs);
    _train(ndata, lin_data);
    ams::ResourceManager::deallocate(lin_data, cache_location);
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

    CFATAL(UQModule,
           !has_index(),
           "HDCache does not have a valid and trained index!")
    DBG(UQModule, "Evaluating %ld %ld points using HDCache", ndata, d);

    CFATAL(UQModule,
           (!m_use_random) && (d != m_dim),
           "Mismatch in data dimensionality!")

    if (m_use_random) {
      _evaluate(ndata, is_acceptable);
    } else {
      _evaluate(ndata, data, is_acceptable);
    }

    if (cache_location == AMSResourceType::DEVICE) {
      deviceCheckErrors(__FILE__, __LINE__);
    }

    DBG(UQModule, "Done with evalution of uq")
  }

  //! train on data that comes separate features (a vector of pointers)
  PERFFASPECT()
  void evaluate(const size_t ndata,
                const std::vector<const TypeInValue *> &inputs,
                bool *is_acceptable) const
  {

    CFATAL(UQModule,
           !has_index(),
           "HDCache does not have a valid and trained index!")
    DBG(UQModule,
        "Evaluating %ld %ld points using HDCache configured with %d neighbors, "
        "%f threshold, %d policy",
        ndata,
        inputs.size(),
        m_knbrs,
        acceptable_error,
        m_policy);
    CFATAL(UQModule,
           ((!m_use_random) && inputs.size() != m_dim),
           "Mismatch in data dimensionality!")

    if (m_use_random) {
      _evaluate(ndata, is_acceptable);
    } else {
      TypeValue *lin_data =
          data_handler::linearize_features(cache_location, ndata, inputs);
      _evaluate(ndata, lin_data, is_acceptable);
      ams::ResourceManager::deallocate(lin_data, cache_location);
    }
    DBG(UQModule, "Done with evalution of uq");
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
    TypeValue *vdata =
        data_handler::cast_to_typevalue(cache_location, ndata, data);
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
      throw std::invalid_argument("!");

    CFATAL(UQModule,
           (m_index != nullptr && m_index->is_trained),
           "Trying to re-train an already trained index")

    m_index = faiss::index_factory(m_dim, index_key);
    m_index->train(ndata, data);

    CFATAL(UQModule, ((!m_index->is_trained)), "Failed to train index")
  }

  //! train an index when (data type != TypeValue)
  template <typename T,
            std::enable_if_t<!std::is_same<TypeValue, T>::value> * = nullptr>
  PERFFASPECT()
  inline void _train(const size_t ndata, const T *data)
  {
    TypeValue *vdata =
        data_handler::cast_to_typevalue(cache_location, ndata, data);
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

    TypeValue *kdists =
        ams::ResourceManager::allocate<TypeValue>(ndata * knbrs,
                                                  cache_location);
    TypeIndex *kidxs =
        ams::ResourceManager::allocate<TypeIndex>(ndata * knbrs,
                                                  cache_location);

    // query faiss
    // TODO: This is a HACK. When searching more than 65535
    // items in the GPU case, faiss is throwing an exception.
    const unsigned int MAGIC_NUMBER = 65535;
    for (int start = 0; start < ndata; start += MAGIC_NUMBER) {
      unsigned int nElems =
          ((ndata - start) < MAGIC_NUMBER) ? ndata - start : MAGIC_NUMBER;
      DBG(UQModule, "Running for %d elements %d %d", nElems, start, m_dim);
      m_index->search(nElems,
                      &data[start * m_dim],
                      knbrs,
                      &kdists[start * knbrs],
                      &kidxs[start * knbrs]);
    }
#ifdef __ENABLE_CUDA__
    faiss::gpu::synchronizeAllDevices();
#endif

    // compute means
    if (cache_location == AMSResourceType::HOST) {
      for (size_t i = 0; i < ndata; ++i) {
        if (m_policy == AMSUQPolicy::FAISS_Mean) {
          TypeValue mean_dist = std::accumulate(kdists + i * knbrs,
                                                kdists + (i + 1) * knbrs,
                                                0.) *
                                ook;
          is_acceptable[i] = mean_dist < acceptable_error;
        } else if (m_policy == AMSUQPolicy::FAISS_Max) {
          // Take the furtherst cluster as the distance metric
          TypeValue max_dist =
              *std::max_element(&kdists[i * knbrs],
                                &kdists[i * knbrs + knbrs - 1]);
          is_acceptable[i] = (max_dist) < acceptable_error;
        }
      }
    } else {
      CFATAL(UQModule,
             m_policy == AMSUQPolicy::FAISS_Max,
             "FAISS Max on device is not supported yet");

      ams::Device::computePredicate(
          kdists, is_acceptable, ndata, knbrs, acceptable_error);
    }

    ams::ResourceManager::deallocate(kdists, cache_location);
    ams::ResourceManager::deallocate(kidxs, cache_location);
  }

  //! evaluate cache uncertainty when (data type != TypeValue)
  template <typename T,
            std::enable_if_t<!std::is_same<TypeValue, T>::value> * = nullptr>
  inline void _evaluate(const size_t ndata, T *data, bool *is_acceptable) const
  {
    TypeValue *vdata =
        data_handler::cast_to_typevalue(cache_location, ndata, data);
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
    if (cache_location == AMSResourceType::DEVICE) {
#ifdef __ENABLE_CUDA__
      random_uq_device<<<1, 1>>>(is_acceptable, ndata, acceptable_error);
#else
      THROW(std::runtime_error,
            "Random-uq is not configured to use device allocations");
#endif
    } else {
      random_uq_host(is_acceptable, ndata, acceptable_error);
    }
  }
  // -------------------------------------------------------------------------
};

template <typename T>
std::unordered_map<std::string, std::shared_ptr<HDCache<T>>>
    HDCache<T>::instances;

#endif
