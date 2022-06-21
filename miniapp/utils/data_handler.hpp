#ifndef __DATA_HANDLER_HPP__
#define __DATA_HANDLER_HPP__

#include <algorithm>
#include <vector>

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/dtensor.hpp"

const int partitionSize = 1 << 24;

using mfem::ForallWrap;

#if __cplusplus < 201402L
template <bool B, typename T = void> using enable_if_t = typename std::enable_if<B, T>::type;
#else

#endif

template <typename TypeValue> class DataHandler {

  public:
    //! -----------------------------------------------------------------------
    //! cast an array into TypeValue
    //! -----------------------------------------------------------------------

    //! when  (data type = TypeValue)
    template <class T, std::enable_if_t<std::is_same<TypeValue, T>::value> * = nullptr>
    static inline TypeValue *cast_to_typevalue(const size_t n, T *data) {
        return data;
    }

    //! when  (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue, T>::value> * = nullptr>
    static inline TypeValue *cast_to_typevalue(const size_t n, T *data) {
        TypeValue *fdata = new TypeValue[n];
        std::transform(data, data + n, fdata,
                       [&](const T &v) { return static_cast<TypeValue>(v); });
        return fdata;
    }

    //! when  (data type == TypeValue)
    template <typename T, std::enable_if_t<std::is_same<TypeValue, T>::value> * = nullptr>
    static inline void cast_from_typevalue(const size_t n, T *dest, TypeValue *src) {
        std::transform(src, src + n, dest, [&](const T &v) { return v; });
    }

    //! when  (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue, T>::value> * = nullptr>
    static inline void cast_from_typevalue(const size_t n, T *dest, TypeValue *src) {
        std::transform(src, src + n, dest, [&](const T &v) { return static_cast<T>(v); });
    }

    //! -----------------------------------------------------------------------
    //! linearize a set of features (vector of pointers) into
    //! a single vector of TypeValue (input can be another datatype)

    template <typename T>
    static inline std::vector<TypeValue> linearize_features(const size_t ndata,
                                                            const std::vector<T *> &features) {

        const size_t nfeatures = features.size();
        std::vector<TypeValue> ldata(ndata * nfeatures);

        for (size_t i = 0; i < ndata; i++) {
            for (size_t d = 0; d < nfeatures; d++) {
                ldata[i * nfeatures + d] = static_cast<TypeValue>(features[d][i]);
            }
        }
        return ldata;
    }

    //! -----------------------------------------------------------------------
    //! packing code for mfem tensors (i,j,k)
    //! -----------------------------------------------------------------------
    //! for us, index j is sparse wrt k
    //!     i.e., for a given k, certain j are inactive
    //! so we pack a sparse (i,j,k) tensor into a dense (i,j*) tensor
    //!     for a given k
    //!     where j* is the linearized "dense" index of all sparse indices "j"

    template <typename T> using dt1 = mfem::DeviceTensor<1, T>;
    template <typename T> using dt2 = mfem::DeviceTensor<2, T>;
    template <typename T> using dt3 = mfem::DeviceTensor<3, T>;

    template <typename Tin, typename Tout>
    static inline void pack_ij(const int k, const int sz_i, const int sz_sparse_j,
                               const int offset_sparse_j, const dt1<int> &sparse_j_indices,
                               const dt3<Tin> &a3, const dt2<Tout> &a2, const dt3<Tin> &b3,
                               const dt2<Tout> &b2) {

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
                                 const int offset_sparse_j, const dt1<int> &sparse_j_indices,
                                 const dt2<Tin> &a2, const dt3<Tout> &a3, const dt2<Tin> &b2,
                                 const dt3<Tout> &b3, const dt2<Tin> &c2, const dt3<Tout> &c3,
                                 const dt2<Tin> &d2, const dt3<Tout> &d3) {

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
    //! we pack the data based on the predicate
    //! to allow chunking, pack n elements from a given offset
    static inline size_t pack(const bool *predicate, const size_t offset, const size_t n,
                              int *sparse_indices, const TypeValue *a, TypeValue *pa,
                              const TypeValue *b, TypeValue *pb) {

        size_t npacked = 0;
        for (size_t i = 0; i < n; i++) {
            if (predicate[offset + i]) {
                pa[npacked] = a[offset + i];
                pb[npacked] = b[offset + i];
                sparse_indices[npacked++] = offset + i;
            }
        }
        return npacked;
    }

    static inline void unpack(const int *sparse_indices, const size_t n, const TypeValue *pa,
                              TypeValue *a, const TypeValue *pb, TypeValue *b, const TypeValue *pc,
                              TypeValue *c, const TypeValue *pd, TypeValue *d) {

        for (size_t i = 0; i < n; i++) {
            auto sidx = sparse_indices[i];
            a[sidx] = pa[i];
            b[sidx] = pb[i];
            c[sidx] = pc[i];
            d[sidx] = pd[i];
        }
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

#endif
