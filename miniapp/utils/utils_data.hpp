#ifndef __AMS_UTILS_DATA_HPP__
#define __AMS_UTILS_DATA_HPP__

#include <algorithm>
#include <vector>
#include <random>

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/dtensor.hpp"
#include "utils/allocator.hpp"
#include "utils/utils_cuda.hpp"

#include "umpire/ResourceManager.hpp"


const int partitionSize = 1 << 24;

using mfem::ForallWrap;

#if __cplusplus < 201402L
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
#else

#endif



// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double unitrand() { return (double)rand() / RAND_MAX; }


template <typename T>
static inline
T* create_random(const size_t dim, const size_t n) {

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    T *data = new T[dim * n];

    for (size_t i = 0; i < n; i++) {
        for (size_t d = 0; d < dim; d++) {
            data[dim*i + d] = distrib(rng);
            data[dim*i] += i / 1000.;
        }
    }
    return data;
}
void random_uq_host(bool *uq_flags, int ndata, double acceptable_error) {

  for(int i = 0; i < ndata; i++) {
      uq_flags[i] = ((double)rand() / RAND_MAX) <= acceptable_error;
  }
}

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
#if 0
    template<typename T>
    static inline
    TypeValue*
    linearize_features(const size_t ndata, const std::vector<const T*> &features) {

        const size_t nfeatures = features.size();

        TypeValue *data = ams::ResourceManager::allocate<TypeValue>(ndata*nfeatures);

        //auto &rm = umpire::ResourceManager::getInstance();
        //auto dataAllocator = rm.getAllocator(AMS::ResourceManager::getHostAllocatorName());
        //TypeValue *data = static_cast<TypeValue*> (dataAllocator.allocate(ndata*nfeatures*sizeof(TypeValue)));

        for (size_t i = 0; i < ndata; i++) {
        for (size_t d = 0; d < nfeatures; d++) {
            data[i*nfeatures + d] = static_cast<TypeValue>(features[d][i]);
        }}
        return data;
    }


    // TODO: merge this with the above function (make a single linearize?)
    template<typename T>
    static inline
    TypeValue*
    linearize_features_device(const size_t ndata, const std::vector<T*> &features) {

        size_t nfeatures = features.size();

        //auto &rm = umpire::ResourceManager::getInstance();
        //auto dataAllocator = rm.getAllocator(AMS::utilities::getDefaultAllocatorName());
        //TypeValue *data = static_cast<TypeValue*> (dataAllocator.allocate(ndata*nfeatures*sizeof(TypeValue)));

        TypeValue *data = ams::ResourceManager::allocate<TypeValue>(ndata*nfeatures);

        std::cerr << "WARNING: linearize_features_device has incorrect logic!\n";
        /*for (size_t i = 0; i < ndata; i++) {
        for (size_t d = 0; d < nfeatures; d++) {
            data[i*nfeatures + d] = static_cast<TypeValue>(features[d][i]);
        }}*/

        // TODO: this is incorrect ordering!
        // but lets get the data movement working
        for(size_t i = 0; i < nfeatures; i++) {
          //HtoDMemcpy(static_cast<void*>(data + i*ndata),
          //           static_cast<void*>(features[i]),
          //           ndata*sizeof(T));
        }

        //auto found_allocator = rm.getAllocator(data);
        //std::cout << " created linearized: " << found_allocator << "\n";
        return data;
    }
#endif

    template<typename T>
    static inline
    TypeValue*
    linearize_features(const size_t ndata, const std::vector<const T*> &features) {

        std::cout << " lineraize_hd\n";
        for(int i = 0; i < ndata; i++) {
            std::cout << "["<<i<<"] = " << features[0][i] << " : " << features[1][i] << "\n";
        }

        const size_t nfeatures = features.size();
        const size_t nvalues = ndata*nfeatures;

        TypeValue *data = nullptr;

        // features are on host
        const bool features_on_device = ams::ResourceManager::is_on_device(features[0]);
        if (!features_on_device) {

            data = ams::ResourceManager::allocate<TypeValue>(nvalues);

            for (size_t d = 0; d < nfeatures; d++) {
            for (size_t i = 0; i < ndata; i++) {
                data[i*nfeatures + d] = static_cast<TypeValue>(features[d][i]);
            }}
        }

        // features are on device
        else {

            // move data to host, linearize, and move back
            // TODO: linearize directly on device as this is inefficient

            data = ams::ResourceManager::allocate<TypeValue>(nvalues);

            T* hfeature = new T[ndata];
            for (size_t d = 0; d < nfeatures; d++) {
                DtoHMemcpy(hfeature, const_cast<T*>(features[d]), ndata*sizeof(T));

                for (size_t i = 0; i < ndata; i++) {
                    data[i*nfeatures + d] = static_cast<TypeValue>(hfeature[i]);
                }
            }
        }

        /*
        // clean this and merge!
        if (ams::ResourceManager::is_on_device(features[0])) {
            std::cout << " on device!\n";
            return linearize_features_device(ndata, features);
        }
        else {
            std::cout << " on host!\n";
            return linearize_features(ndata, features);
        }*/

        std::cout << " final\n";
        for(int i = 0; i < nvalues; i++) {
            std::cout << "["<<i<<"] = " << data[i] << "\n";
        }

        return data;

    }

    //! -----------------------------------------------------------------------
    //! packing code for mfem tensors (i,j,k)
    //! -----------------------------------------------------------------------
    //! for us, index j is sparse wrt k
    //!     i.e., for a given k, certain j are inactive
    //! so we pack a sparse (i,j,k) tensor into a dense (i,j*) tensor
    //!     for a given k
    //!     where j* is the linearized "dense" index of all sparse indices "j"

    template <typename T>
    using dt1 = mfem::DeviceTensor<1, T>;
    template <typename T>
    using dt2 = mfem::DeviceTensor<2, T>;
    template <typename T>
    using dt3 = mfem::DeviceTensor<3, T>;

    template <typename Tin, typename Tout>
    static inline void pack_ij(const int k, const int sz_i, const int sz_sparse_j,
                               const int offset_sparse_j, const dt1<int>& sparse_j_indices,
                               const dt3<Tin>& a3, const dt2<Tout>& a2, const dt3<Tin>& b3,
                               const dt2<Tout>& b2) {

        MFEM_FORALL(j, sz_sparse_j, {
            const int sparse_j = sparse_j_indices[offset_sparse_j + j];
            for (int i = 0; i < sz_i; ++i) {
                a2(i, j) = a3(i, sparse_j, k);
                b2(i, j) = b3(i, sparse_j, k);
            }
        });
    }

    template <typename Tin, typename Tout>
    static inline void unpack_ij(const int k, const int sz_i, const int sz_sparse_j,
                                 const int offset_sparse_j, const dt1<int>& sparse_j_indices,
                                 const dt2<Tin>& a2, const dt3<Tout>& a3, const dt2<Tin>& b2,
                                 const dt3<Tout>& b3, const dt2<Tin>& c2, const dt3<Tout>& c3,
                                 const dt2<Tin>& d2, const dt3<Tout>& d3) {

        MFEM_FORALL(j, sz_sparse_j, {
            const int sparse_j = sparse_j_indices[offset_sparse_j + j];
            for (int i = 0; i < sz_i; ++i) {
                a3(i, sparse_j, k) = a2(i, j);
                b3(i, sparse_j, k) = b2(i, j);
                c3(i, sparse_j, k) = c2(i, j);
                d3(i, sparse_j, k) = d2(i, j);
            }
        });
    }

    //! -----------------------------------------------------------------------
    //! packing code for pointers based on boolean predicates
    //! -----------------------------------------------------------------------
    //! since boolean predicate is likely to be sparse
    //! we pack the data based on the predicate value
    static inline size_t pack(const bool* predicate, const size_t n,
                              std::vector<const TypeValue*>& sparse, std::vector<TypeValue*>& dense,
                              bool denseVal = true) {
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
          npacked = ams::Device::pack(predicate, n, sparse.data(), dense.data(), dims);
        }
        return npacked;
    }

    //! -----------------------------------------------------------------------
    //! unpacking code for pointers based on boolean predicates
    //! -----------------------------------------------------------------------
    //! Reverse packing. From the dense representation we copy data
    //! back to the sparse one based on the value of the predeicate.
    static inline void unpack(const bool* predicate, const size_t n, std::vector<TypeValue*>& dense,
                              std::vector<TypeValue*>& sparse, bool denseVal = true) {

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
          npacked = ams::Device::unpack(predicate, n, sparse.data(), dense.data(), dims);
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
                              bool denseVal = true) {

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
          npacked = ams::Device::pack(predicate, n, sparse.data(), dense.data(), sparse_indices, dims);
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
                              bool denseVal = true) {

        if (sparse.size() != dense.size())
            throw std::invalid_argument("Packing arrays size mismatch");

        int dims = sparse.size();

        if ( !ams::ResourceManager::isDeviceExecution() ){
          for (size_t i = 0; i < nPacked; i++)
              for (size_t j = 0; j < dims; j++)
                  sparse[j][sparse_indices[i]] = dense[j][i];
        }
        else{
          ams::Device::unpack(nPacked, sparse.data(), dense.data(), sparse_indices, dims);
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
