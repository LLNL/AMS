#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "AMSTypes.hpp"
#include "ArrayRef.hpp"
#include "SmallVector.hpp"

namespace ams
{
class AMSTensor
{
public:
  using IntDimType = long int;
  using LifetimeToken = std::shared_ptr<void>;

private:
  uint8_t* _data = nullptr;
  IntDimType _elements = 0;
  IntDimType _element_size = 0;
  SmallVector<IntDimType> _shape;
  SmallVector<IntDimType> _strides;
  AMSDType _dType = AMS_UNKNOWN_TYPE;
  AMSResourceType _location = AMS_UNKNOWN;
  bool _contiguous = false;
  bool _writable = false;
  bool _valid = false;
  size_t _bytes = 0;
  size_t _storage_bytes = 0;
  LifetimeToken _lifetime;

  AMSTensor(uint8_t* data,
            ArrayRef<IntDimType> shapes,
            ArrayRef<IntDimType> strides,
            AMSDType dType,
            AMSResourceType location,
            bool writable,
            LifetimeToken lifetime = {});
  void requireValid() const;
  void requireDataType(AMSDType requested) const;
  void resetMovedFrom() noexcept;

  template <typename T>
  static constexpr AMSDType dtypeFor()
  {
    using U = std::remove_cv_t<T>;
    static_assert(std::is_same_v<U, float> || std::is_same_v<U, double> ||
                      std::is_same_v<U, int32_t> || std::is_same_v<U, int64_t>,
                  "Unsupported AMS scalar type");
    if constexpr (std::is_same_v<U, float>) return AMS_SINGLE;
    if constexpr (std::is_same_v<U, double>) return AMS_DOUBLE;
    if constexpr (std::is_same_v<U, int32_t>) return AMS_INT32;
    return AMS_INT64;
  }

public:
  template <typename ScalarType>
  static AMSTensor create(ArrayRef<IntDimType> shapes,
                          ArrayRef<IntDimType> strides,
                          AMSResourceType location);

  /** Borrow memory. The caller remains responsible for its lifetime/capacity. */
  template <typename ScalarType>
  static AMSTensor view(ScalarType* data,
                        ArrayRef<IntDimType> shapes,
                        ArrayRef<IntDimType> strides,
                        AMSResourceType location);

  /** View foreign memory and retain the supplied foreign-storage owner. */
  template <typename ScalarType>
  static AMSTensor view(ScalarType* data,
                        ArrayRef<IntDimType> shapes,
                        ArrayRef<IntDimType> strides,
                        AMSResourceType location,
                        LifetimeToken lifetime);

  static AMSTensor view(AMSTensor& tensor);
  static AMSTensor view(const AMSTensor& tensor);

  ~AMSTensor() = default;
  AMSTensor(const AMSTensor&) = delete;
  AMSTensor& operator=(const AMSTensor&) = delete;
  AMSTensor(AMSTensor&& other) noexcept;
  AMSTensor& operator=(AMSTensor&& other) noexcept;

  IntDimType elements() const
  {
    requireValid();
    return _elements;
  }
  IntDimType element_size() const
  {
    requireValid();
    return _element_size;
  }
  size_t nbytes() const
  {
    requireValid();
    return _bytes;
  }
  size_t storage_nbytes() const
  {
    requireValid();
    return _storage_bytes;
  }
  size_t dim() const
  {
    requireValid();
    return _shape.size();
  }
  AMSDType dtype() const
  {
    requireValid();
    return _dType;
  }
  AMSResourceType location() const
  {
    requireValid();
    return _location;
  }
  ArrayRef<IntDimType> strides() const
  {
    requireValid();
    return _strides;
  }
  ArrayRef<IntDimType> shape() const
  {
    requireValid();
    return _shape;
  }
  ArrayRef<IntDimType> sizes() const { return shape(); }
  bool contiguous() const
  {
    requireValid();
    return _contiguous;
  }
  bool writable() const
  {
    requireValid();
    return _writable;
  }
  bool valid() const noexcept { return _valid; }
  LifetimeToken lifetimeToken() const
  {
    requireValid();
    return _lifetime;
  }

  template <typename T>
  T* data()
  {
    requireValid();
    requireDataType(dtypeFor<T>());
    if constexpr (!std::is_const_v<T>) {
      if (!_writable) throw std::logic_error("AMSTensor is read-only");
    }
    return reinterpret_cast<T*>(_data);
  }

  template <typename T>
  const std::remove_cv_t<T>* data() const
  {
    requireValid();
    requireDataType(dtypeFor<T>());
    return reinterpret_cast<const std::remove_cv_t<T>*>(_data);
  }

  void* data_ptr()
  {
    requireValid();
    if (!_writable) throw std::logic_error("AMSTensor is read-only");
    return _data;
  }
  const void* data_ptr() const
  {
    requireValid();
    return _data;
  }

  AMSTensor transpose(IntDimType axis1 = 0, IntDimType axis2 = 1);
  AMSTensor transpose(IntDimType axis1 = 0, IntDimType axis2 = 1) const;
  AMSTensor clone() const;
  static AMSTensor concat(ArrayRef<AMSTensor> tensors, AMSDType inputDType);
};

extern template AMSTensor AMSTensor::create<float>(
    ArrayRef<AMSTensor::IntDimType>,
    ArrayRef<AMSTensor::IntDimType>,
    AMSResourceType);
extern template AMSTensor AMSTensor::create<double>(
    ArrayRef<AMSTensor::IntDimType>,
    ArrayRef<AMSTensor::IntDimType>,
    AMSResourceType);
extern template AMSTensor AMSTensor::create<int32_t>(
    ArrayRef<AMSTensor::IntDimType>,
    ArrayRef<AMSTensor::IntDimType>,
    AMSResourceType);
extern template AMSTensor AMSTensor::create<int64_t>(
    ArrayRef<AMSTensor::IntDimType>,
    ArrayRef<AMSTensor::IntDimType>,
    AMSResourceType);
}  // namespace ams
