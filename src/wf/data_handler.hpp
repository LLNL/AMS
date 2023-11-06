/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_UTILS_DATA_HPP__
#define __AMS_UTILS_DATA_HPP__

#include <algorithm>
#include <random>
#include <vector>

#include "wf/device.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

namespace ams
{
/**
 * @brief A "utility" class that transforms data into
 * various formats. For example moving from sparse to dense
 * representations
 */
template <typename TypeValue>
class DataHandler
{

public:
  /* @brief Casts C-vector from one type to another type.
   *
   * This function uses tempalte metaprogramming. When both
   * the class Templated 'TypeValue' and the functions template
   * 'TypeInValue' have the same type, we return directly
   * the same memory.
   *
   * @tparam TypeInValue Type of the source value.
   * @param[in] n The number of elements of the vector.
   * @param[in] data A pointer pointing to the C-vector to be casted.
   * @return A pointer to a C-vector containing the casted values.
   *
   */
  template <
      class TypeInValue,
      std::enable_if_t<std::is_same<TypeValue, TypeInValue>::value>* = nullptr>
  static inline TypeValue* cast_to_typevalue(AMSResourceType resource, const size_t n, TypeInValue* data)
  {
    return data;
  }

  /* @brief Casts C-vector from one type to another type.
   *
   * This function uses tempalte metaprogramming. Both
   * the class Templated 'TypeValue' and the functions template
   * 'TypeInValue' have a different type, thus we allocate a new
   * vector of 'TypeInValue' type and we cast each element of the vector
   * to the desired ('TypeValue') type.
   *
   * @tparam TypeInValue Type of the source value.
   * @param[in] n The number of elements of the vector.
   * @param[in] data A pointer pointing to the C-vector to be casted.
   * @return A pointer to a C-vector containing the casted values.
   *
   */
  template <
      typename TypeInValue,
      std::enable_if_t<!std::is_same<TypeValue, TypeInValue>::value>* = nullptr>
  static inline TypeValue* cast_to_typevalue(AMSResourceType resource, const size_t n, TypeInValue* data)
  {
    TypeValue* fdata = ams::ResourceManager::allocate<TypeValue>(resource, n);
    std::transform(data, data + n, fdata, [&](const TypeInValue& v) {
      return static_cast<TypeValue>(v);
    });
    return fdata;
  }

  /* @brief Casts all elements of a C-vector from one type to
   * the other type and stores them to the 'dest' vector.
   *
   * This function uses tempalte metaprogramming. In this function
   * template datatypes match, thus we just copy data from
   * one vector to another.
   *
   * @tparam TypeInValue Type of the source value.
   * @param[in] n The number of elements of the vectors.
   * @param[out] dest The destination vector.
   * @param[in] src The source vector.
   * @return A pointer to a C-vector containing the casted values.
   */
  template <
      typename TypeInValue,
      std::enable_if_t<std::is_same<TypeValue, TypeInValue>::value>* = nullptr>
  static inline void cast_from_typevalue(const size_t n,
                                         TypeInValue* dest,
                                         TypeValue* src)
  {
    std::transform(src, src + n, dest, [&](const TypeInValue& v) { return v; });
  }

  /* @brief Casts all elements of a C-vector from one type to
   * the other type and stores them to the 'dest' vector.
   *
   * This function uses tempalte metaprogramming. In this function
   * template datatypes do not match, thus we cast each element
   * and store it to destination vector
   *
   * @tparam TypeInValue Type of the source value.
   * @param[in] n The number of elements of the vectors.
   * @param[out] dest The destination vector.
   * @param[in] src The source vector.
   * @return A pointer to a C-vector containing the casted values.
   */
  template <
      typename TypeInValue,
      std::enable_if_t<!std::is_same<TypeValue, TypeInValue>::value>* = nullptr>
  static inline void cast_from_typevalue(const size_t n,
                                         TypeInValue* dest,
                                         TypeValue* src)
  {
    std::transform(src, src + n, dest, [&](const TypeInValue& v) {
      return static_cast<TypeInValue>(v);
    });
  }

  /* @brief linearize all elements of a vector of C-vectors
   * in a single C-vector. Data are transposed.
   *
   * @tparam TypeInValue Type of the source value.
   * @param[in] n The number of elements of the vectors.
   * @param[in] features A vector containing C-vector of feature values.
   * @return A pointer to a C-vector containing the linearized values. The
   * C-vector is_same resident in the same device as the input feature pointers.
   */
  template <typename TypeInValue>
PERFFASPECT()
  static inline TypeValue* linearize_features(
      AMSResourceType resource,
      const size_t n,
      const std::vector<const TypeInValue*>& features)
  {

    const size_t nfeatures = features.size();
    const size_t nvalues = n * nfeatures;

    TypeValue* data = ams::ResourceManager::allocate<TypeValue>(nvalues, resource);

    if (resource == AMSResourceType::HOST) {
      for (size_t d = 0; d < nfeatures; d++) {
        for (size_t i = 0; i < n; i++) {
          data[i * nfeatures + d] = static_cast<TypeValue>(features[d][i]);
        }
      }
    } else {
      ams::Device::linearize(data, features.data(), nfeatures, n);
    }
    return data;
  }

  /* @brief The function stores all elements of the sparse
   * vector in the dense vector if the respective index
   * of the predicate vector is equal to 'denseVal.
   *
   * @param[in] dataLocation Location of the data
   * @param[in] predicate A boolean vector storing which elements in the vector
   * should be dropped.
   * @param[in] n The number of elements of the C-vectors.
   * @param[in] sparse A vector containing C-vectors whose elements will be
   * dropped
   * @param[out] dense A vector containing C-vectors with the remaining elements
   * @param[in] denseVal The condition the predicate needs to meet for the index
   * to be stored in the dense vector
   * @return Total number of elements stored in the dense vector
   * */
PERFFASPECT()
  static inline size_t pack(AMSResourceType dataLocation, const bool* predicate,
                            const size_t n,
                            std::vector<const TypeValue*>& sparse,
                            std::vector<TypeValue*>& dense,
                            bool denseVal = false)
  {
    if (sparse.size() != dense.size())
      throw std::invalid_argument("Packing arrays size mismatch");

    size_t npacked = 0;
    size_t dims = sparse.size();

    if (dataLocation != AMSResourceType::DEVICE) {
      for (size_t i = 0; i < n; i++) {
        if (predicate[i] == denseVal) {
          for (size_t j = 0; j < dims; j++)
            dense[j][npacked] = sparse[j][i];
          npacked++;
        }
      }
    } else {
      npacked = ams::Device::pack(denseVal,
                                  predicate,
                                  n,
                                  static_cast<const TypeValue**>(sparse.data()),
                                  dense.data(),
                                  dims);
    }
    return npacked;
  }

  /* @brief The function stores all elements from the dense
   * vector to the sparse vector.
   *
   * @param[in] dataLocation Location of the data
   * @param[in] predicate A boolean vector storing which elements in the vector
   * should be kept.
   * @param[in] n The number of elements of the C-vectors.
   * dropped
   * @param[in] dense A vector containing C-vectors with elements
   * to be stored in the sparse vector
   * @param[out] sparse A vector containing C-vectors whose elements will be
   * @param[in] denseVal The condition the predicate needs to meet for the index
   * to be copied to the sparse vectors.
   * */
PERFFASPECT()
  static inline void unpack(AMSResourceType dataLocation, const bool* predicate,
                            const size_t n,
                            std::vector<TypeValue*>& dense,
                            std::vector<TypeValue*>& sparse,
                            bool denseVal = false)
  {

    if (sparse.size() != dense.size())
      throw std::invalid_argument("Packing arrays size mismatch");

    size_t npacked = 0;
    size_t dims = sparse.size();
    if (dataLocation != AMSResourceType::DEVICE) {
      for (size_t i = 0; i < n; i++) {
        if (predicate[i] == denseVal) {
          for (size_t j = 0; j < dims; j++)
            sparse[j][i] = dense[j][npacked];
          npacked++;
        }
      }
    } else {
      npacked = ams::Device::unpack(denseVal,
                                    predicate,
                                    n,
                                    sparse.data(),
                                    dense.data(),
                                    dims);
    }
    return;
  }

  /* @brief The function stores all elements of the sparse
   * vector in the dense vector if the respective index
   * of the predicate vector is equal to 'denseVal.
   *
   * @param[in] dataLocation Location of the data
   * @param[in] predicate A boolean vector storing which elements in the vector
   * @param[out] sparse_indices A vector storing the mapping from dense elements
   * to sparse elements.
   * @param[in] n The number of elements of the C-vectors.
   * @param[in] sparse A vector containing C-vectors whose elements will be
   * dropped
   * @param[out] dense A vector containing C-vectors with the remaining elements
   * @param[in] denseVal The condition the predicate needs to meet for the index
   * to be stored in the dense vector
   * @return Total number of elements stored in the dense vector
   * */
PERFFASPECT()
  static inline size_t pack(AMSResourceType dataLocation, const bool* predicate,
                            int* sparse_indices,
                            const size_t n,
                            std::vector<const TypeValue*>& sparse,
                            std::vector<TypeValue*>& dense,
                            bool denseVal = false)
  {

    if (sparse.size() != dense.size())
      throw std::invalid_argument("Packing arrays size mismatch");

    size_t npacked = 0;
    int dims = sparse.size();

    if (dataLocation != AMSResourceType::DEVICE) {
      for (size_t i = 0; i < n; i++) {
        if (predicate[i] == denseVal) {
          for (size_t j = 0; j < dims; j++)
            dense[j][npacked] = sparse[j][i];
          sparse_indices[npacked++] = i;
        }
      }
    } else {
      npacked = ams::Device::pack(denseVal,
                                  predicate,
                                  n,
                                  sparse.data(),
                                  dense.data(),
                                  sparse_indices,
                                  dims);
    }

    return npacked;
  }

  /* @brief The function copies all elements from the dense
   * vector to the sparse vector.
   *
   * @param[in] dataLocation Location of the data
   * @param[in] sparse_indices A vector storing the mapping from sparse to
   * dense.
   * @param[in] n The number of elements of the C-vectors.
   * dropped
   * @param[in] dense A vector containing C-vectors with elements
   * to be stored in the sparse vector
   * @param[out] sparse A vector containing C-vectors whose elements will be
   * @param[in] denseVal The condition the predicate needs to meet for the index
   * to be copied to the sparse vectors.
   * */
PERFFASPECT()
  static inline void unpack(AMSResourceType dataLocation, int* sparse_indices,
                            const size_t nPacked,
                            std::vector<TypeValue*>& dense,
                            std::vector<TypeValue*>& sparse,
                            bool denseVal = false)
  {

    if (sparse.size() != dense.size())
      throw std::invalid_argument("Packing arrays size mismatch");

    int dims = sparse.size();

    if (dataLocation != AMSResourceType::DEVICE) {
      for (size_t i = 0; i < nPacked; i++)
        for (size_t j = 0; j < dims; j++)
          sparse[j][sparse_indices[i]] = dense[j][i];
    } else {
      ams::Device::unpack(
          denseVal, nPacked, sparse.data(), dense.data(), sparse_indices, dims);
    }

    return;
  }

};
}  // namespace ams

// -----------------------------------------------------------------------------
#endif
