#ifndef __AMS_UTILS_DATA_HPP__
#define __AMS_UTILS_DATA_HPP__

#include <algorithm>
#include <vector>
#include <random>

#include "wf/device.hpp"
#include "wf/utils.hpp"
#include "wf/resource_manager.hpp"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
namespace ams {

template<typename TypeValue>
class DataHandler {

   public:
    //! -----------------------------------------------------------------------
    //! cast an array into TypeValue
    //! -----------------------------------------------------------------------

    //! when  (data type = TypeValue)
    template <class T, std::enable_if_t<std::is_same<TypeValue, T>::value>* = nullptr>
    static inline TypeValue* cast_to_typevalue(const size_t n, T* data) {
        return data;
    }

    //! when  (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue, T>::value>* = nullptr>
    static inline TypeValue* cast_to_typevalue(const size_t n, T* data) {
        //TypeValue* fdata = static_cast<TypeValue *> (AMS::utilities::allocate(n * sizeof(TypeValue)));
        TypeValue* fdata = ams::ResourceManager::allocate<TypeValue>(n);
        std::transform(data, data + n, fdata,
                       [&](const T& v) { return static_cast<TypeValue>(v); });
        return fdata;
    }

    //! when  (data type == TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue, T>::value>* = nullptr>
    static inline void cast_from_typevalue(const size_t n, T* dest, TypeValue* src) {
        std::transform(src, src + n, dest, [&](const T& v) { return v; });
    }

    //! when  (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue, T>::value>* = nullptr>
    static inline void cast_from_typevalue(const size_t n, T* dest, TypeValue* src) {
        std::transform(src, src + n, dest, [&](const T& v) { return static_cast<T>(v); });
    }

    //! -----------------------------------------------------------------------
    //! linearize a set of features (vector of pointers) into
    //! a single vector of TypeValue (input can be another datatype)
    template<typename T>
    static inline
    TypeValue*
    linearize_features(const size_t ndata, const std::vector<const T*> &features) {

        const size_t nfeatures = features.size();
        const size_t nvalues = ndata*nfeatures;

        // output!
        TypeValue *data = ams::ResourceManager::allocate<TypeValue>(nvalues);

        // features are on host
        const bool features_on_device = ams::ResourceManager::is_on_device(features[0]);
        if (!features_on_device) {

            for (size_t d = 0; d < nfeatures; d++) {
            for (size_t i = 0; i < ndata; i++) {
                data[i*nfeatures + d] = static_cast<TypeValue>(features[d][i]);
            }}
        }

        // features are on device
        else {

            // move data to host, linearize, and move back
            // TODO: linearize directly on device as this is inefficient

            // host copies!
            TypeValue *hdata = ams::ResourceManager::allocate<TypeValue>(nvalues, ams::ResourceManager::ResourceType::HOST);
            T* hfeature = ams::ResourceManager::allocate<T>(ndata, ams::ResourceManager::ResourceType::HOST);

            for (size_t d = 0; d < nfeatures; d++) {
                DtoHMemcpy(hfeature, const_cast<T*>(features[d]), ndata*sizeof(T));
                for (size_t i = 0; i < ndata; i++) {
                    hdata[i*nfeatures + d] = static_cast<TypeValue>(hfeature[i]);
                }
            }
            HtoDMemcpy(data, hdata, ndata*sizeof(TypeValue));
            ams::ResourceManager::deallocate(hdata);
            ams::ResourceManager::deallocate(hfeature);
        }
        return data;
    }

    //! -----------------------------------------------------------------------
    //! packing code for pointers based on boolean predicates
    //! -----------------------------------------------------------------------
    //! since boolean predicate is likely to be sparse
    //! we pack the data based on the predicate value
    static inline size_t pack(const bool* predicate, const size_t n,
                              std::vector<const TypeValue*>& sparse, std::vector<TypeValue*>& dense,
                              bool denseVal = false) {
        if (sparse.size() != dense.size())
            throw std::invalid_argument("Packing arrays size mismatch");

        size_t npacked = 0;
        size_t dims = sparse.size();

        if ( !ams::ResourceManager::isDeviceExecution() ){
          for (size_t i = 0; i < n; i++) {
              if (predicate[i] == denseVal) {
                  for (size_t j = 0; j < dims; j++)
                      dense[j][npacked] = sparse[j][i];
                  npacked++;
              }
          }
        }
        else {
          npacked = ams::Device::pack(denseVal, predicate, n, const_cast<TypeValue**>(sparse.data()), dense.data(), dims);
        }
        return npacked;
    }

    //! -----------------------------------------------------------------------
    //! unpacking code for pointers based on boolean predicates
    //! -----------------------------------------------------------------------
    //! Reverse packing. From the dense representation we copy data
    //! back to the sparse one based on the value of the predeicate.
    static inline void unpack(const bool* predicate, const size_t n, std::vector<TypeValue*>& dense,
                              std::vector<TypeValue*>& sparse, bool denseVal = false) {

        if (sparse.size() != dense.size())
            throw std::invalid_argument("Packing arrays size mismatch");

        size_t npacked = 0;
        size_t dims = sparse.size();
        if ( !ams::ResourceManager::isDeviceExecution() ){
          for (size_t i = 0; i < n; i++) {
              if (predicate[i] == denseVal) {
                  for (size_t j = 0; j < dims; j++)
                      sparse[j][i] = dense[j][npacked];
                  npacked++;
              }
          }
        }
        else{
          npacked = ams::Device::unpack(denseVal, predicate, n, sparse.data(), dense.data(), dims);
        }
        return;
    }

    //! -----------------------------------------------------------------------
    //! packing code for pointers based on boolean predicates
    //! -----------------------------------------------------------------------
    //! since boolean predicate is likely to be sparse
    //! we pack the data based on the predicate
    //! to allow chunking, pack n elements and store
    //! reverse mapping into sparse_indices pointer.
    static inline size_t pack(const bool* predicate, int* sparse_indices, const size_t n,
                              std::vector<TypeValue*>& sparse, std::vector<TypeValue*>& dense,
                              bool denseVal = false) {

        if (sparse.size() != dense.size())
            throw std::invalid_argument("Packing arrays size mismatch");

        size_t npacked = 0;
        int dims = sparse.size();

        if ( !ams::ResourceManager::isDeviceExecution() ){
          for (size_t i = 0; i < n; i++) {
              if (predicate[i] == denseVal) {
                  for (size_t j = 0; j < dims; j++)
                      dense[j][npacked] = sparse[j][i];
                  sparse_indices[npacked++] = i;
              }
          }
        } else {
          npacked = ams::Device::pack(denseVal, predicate, n, sparse.data(), dense.data(), sparse_indices, dims);
        }

        return npacked;
    }

    //! -----------------------------------------------------------------------
    //! unpacking code for pointers based on pre-computed sparse reverse indices
    //! -----------------------------------------------------------------------
    //! We unpack data values from a dense (packed) representation to an
    //! sparse representation. We use "sparse_indices" to map indices from the
    //! dense representation to the sparse one
    static inline void unpack(int* sparse_indices, const size_t nPacked,
                              std::vector<TypeValue*>& dense, std::vector<TypeValue*>& sparse,
                              bool denseVal = false) {

        if (sparse.size() != dense.size())
            throw std::invalid_argument("Packing arrays size mismatch");

        int dims = sparse.size();

        if ( !ams::ResourceManager::isDeviceExecution() ){
          for (size_t i = 0; i < nPacked; i++)
              for (size_t j = 0; j < dims; j++)
                  sparse[j][sparse_indices[i]] = dense[j][i];
        }
        else{
          ams::Device::unpack(denseVal, nPacked, sparse.data(), dense.data(), sparse_indices, dims);
        }

        return;
    }

    static inline int computePartitionSize(int numIFeatures, int numOFeatures,
                                           bool includeReIndex = true,
                                           const int pSize = partitionSize) {
        int singleElementBytes = sizeof(TypeValue) * (numIFeatures + numOFeatures);
        // We require the re-index vector
        if (includeReIndex)
            return pSize / (singleElementBytes + sizeof(int));
        else
            return pSize / (singleElementBytes);
    }
};
}   // end of namespace

// -----------------------------------------------------------------------------
#endif
