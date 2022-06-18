#ifndef __DATA_HANDLER_HPP__
#define __DATA_HANDLER_HPP__

#include <vector>
#include <algorithm>


#if __cplusplus < 201402L
  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;
#else

#endif


template<typename TypeValue>
class DataHandler {

public:

    //! -----------------------------------------------------------------------
    //! cast an array into TypeValue
    //! -----------------------------------------------------------------------

    //! when  (data type = TypeValue)
    template <class T, std::enable_if_t<std::is_same<TypeValue,T>::value>* = nullptr>
    static inline
    TypeValue*
    cast_to_typevalue(const size_t n, T *data) {
        return data;
    }

    //! when  (data type != TypeValue)
    template <typename T, std::enable_if_t<!std::is_same<TypeValue,T>::value>* = nullptr>
    static inline
    TypeValue*
    cast_to_typevalue(const size_t n, T *data) {
        TypeValue *fdata = new TypeValue[n];
        std::transform(data, data+n, fdata,
                       [&](const T& v) { return static_cast<TypeValue>(v); });
        return fdata;
    }

    //! -----------------------------------------------------------------------
    //! linearize a set of features (vector of pointers) into
    //! a single vector of TypeValue (input can be another datatype)

    template<typename T>
    static inline
    std::vector<TypeValue>
    linearize_features(const size_t ndata, const std::vector<T*> &features) {

        const size_t nfeatures = features.size();
        std::vector<TypeValue> ldata (ndata*nfeatures);

        for (size_t i = 0; i < ndata; i++) {
        for (size_t d = 0; d < nfeatures; d++) {
                ldata[i*nfeatures + d] = static_cast<TypeValue>(features[d][i]);
        }}
        return ldata;
    }

    //! -----------------------------------------------------------------------
};


#endif
